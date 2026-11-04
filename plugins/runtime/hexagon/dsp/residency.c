// Copyright 2026 RooflineAI GmbH
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "hexagon/dsp/residency.h"

#include <stdlib.h>

#include "AEEStdErr.h"
#include "HAP_farf.h"
#include "HAP_mem.h"
#include "qurt.h"

/// a mapped allocation
typedef struct hexa_residency_entry_s {
  int fd;
  uint64_t size;
  uint8_t *vaddr;
  /// number of acquires not released yet
  uint32_t ref_count;
  /// value of the use counter at the last acquire, for LRU
  uint64_t last_use;
} hexa_residency_entry_t;

/// cache state, guarded by mutex
static struct {
  qurt_mutex_t mutex;
  /// set by the host when it opens a session
  uint64_t budget;
  /// sum of the sizes of all entries
  uint64_t mapped_bytes;
  /// sum of the sizes of all entries with ref_count > 0
  uint64_t in_use_bytes;
  /// incremented on each acquire
  uint64_t use_counter;
  hexa_residency_entry_t *entries;
  uint32_t num_entries;
  uint32_t capacity;
} g_residency = {.mutex = QURT_MUTEX_INIT};

static hexa_residency_entry_t *hexa_residency_find(int fd) {
  for (uint32_t idx = 0; idx < g_residency.num_entries; ++idx) {
    if (g_residency.entries[idx].fd == fd) {
      return &g_residency.entries[idx];
    }
  }
  return NULL;
}

/// Perform a cache operation on a whole mapped allocation.
/// need to hold the mutex, so another cant unmap while this is in progress
static int hexa_residency_cache_op(const hexa_residency_entry_t *entry,
                                   qurt_mem_cache_op_t op) {
  int err = qurt_mem_cache_clean((qurt_addr_t)entry->vaddr, entry->size, op,
                                 QURT_MEM_DCACHE);
  // according to doc, the only error is QURT_EVAL - invalid cache type
  return err == QURT_EOK ? AEE_SUCCESS : AEE_EFAILED;
}

/// Unmap an entry (ref_count must be 0) and remove it from the cache.
/// need to hold the mutex
static void hexa_residency_unmap_entry(hexa_residency_entry_t *entry) {
  // Publish what was written, and leave no cache line at the addresses another
  // allocation may be mapped at next.
  hexa_residency_cache_op(entry, QURT_MEM_CACHE_FLUSH_INVALIDATE);
  int err = HAP_munmap2(entry->vaddr, entry->size);
  if (err != AEE_SUCCESS) {
    FARF(ERROR, "HEXAGON-RUNTIME-ERROR: HAP_munmap2 of fd %d failed: 0x%x",
         entry->fd, err);
  }
  g_residency.mapped_bytes -= entry->size;
  // remove, by moving the last entry in its place
  *entry = g_residency.entries[--g_residency.num_entries];
}

/// Unmap the least recently used entry not in use.
/// @return false if there is none
/// need to hold the mutex
static bool hexa_residency_evict_one(void) {
  hexa_residency_entry_t *lru = NULL;
  for (uint32_t idx = 0; idx < g_residency.num_entries; ++idx) {
    hexa_residency_entry_t *entry = &g_residency.entries[idx];
    if (entry->ref_count == 0 && (!lru || entry->last_use < lru->last_use)) {
      lru = entry;
    }
  }
  // nothing to evict
  if (!lru) {
    return false;
  }
  hexa_residency_unmap_entry(lru);
  return true;
}

int hexagon_dsp_residency_set_budget(uint64_t budget_bytes) {
  if (budget_bytes == 0) {
    return AEE_EBADPARM;
  }
  qurt_mutex_lock(&g_residency.mutex);
  g_residency.budget = budget_bytes;
  qurt_mutex_unlock(&g_residency.mutex);
  return AEE_SUCCESS;
}

/// hexagon_dsp_residency_acquire with the mutex held
static int hexa_residency_acquire_locked(int fd, uint64_t alloc_size,
                                         uint8_t **out_vaddr) {
  hexa_residency_entry_t *entry = hexa_residency_find(fd);
  if (entry) {
    if (entry->ref_count++ == 0) {
      // re-use existing entry
      g_residency.in_use_bytes += entry->size;
    }
    entry->last_use = ++g_residency.use_counter;
    *out_vaddr = entry->vaddr;
    return AEE_SUCCESS;
  }
  if (alloc_size == 0) {
    return AEE_EBADPARM;
  }

  // Make room within the budget.
  while (g_residency.mapped_bytes + alloc_size > g_residency.budget) {
    if (!hexa_residency_evict_one()) {
      // cant evict more: everything is in use
      if (g_residency.in_use_bytes > 0) {
        // might fit the budget if we could release more
        return AEE_EWOULDBLOCK;
      } else {
        // will never fit budget
        return AEE_ENOMEMORY;
      }
    }
  }

  // make room for another entry in the list if list is full
  if (g_residency.num_entries == g_residency.capacity) {
    uint32_t capacity = g_residency.capacity ? 2 * g_residency.capacity : 64;
    hexa_residency_entry_t *entries = (hexa_residency_entry_t *)realloc(
        g_residency.entries, capacity * sizeof(*entries));
    if (!entries) {
      return AEE_ENOMEMORY;
    }
    g_residency.entries = entries;
    g_residency.capacity = capacity;
  }

  // The budget is an estimate, the address space may still be exhausted (or
  // fragmented): unmap more and retry.
  void *vaddr = NULL;
  while (true) {
    vaddr =
        HAP_mmap2(NULL, alloc_size, HAP_PROT_READ | HAP_PROT_WRITE, 0, fd, 0);
    if (vaddr != (void *)-1 && vaddr != NULL) {
      // success
      break;
    }
    // try evicting more due to fragmentation
    if (!hexa_residency_evict_one()) {
      FARF(ERROR,
           "HEXAGON-RUNTIME-ERROR: HAP_mmap2 of fd %d (%llu bytes) failed "
           "with %llu bytes mapped, %llu bytes in use",
           fd, (unsigned long long)alloc_size,
           (unsigned long long)g_residency.mapped_bytes,
           (unsigned long long)g_residency.in_use_bytes);
      return g_residency.in_use_bytes > 0 ? AEE_EWOULDBLOCK : AEE_ENOMEMORY;
    }
  }

  entry = &g_residency.entries[g_residency.num_entries++];
  *entry = (hexa_residency_entry_t){
      .fd = fd,
      .size = alloc_size,
      .vaddr = (uint8_t *)vaddr,
      .ref_count = 1,
      .last_use = ++g_residency.use_counter,
  };
  g_residency.mapped_bytes += alloc_size;
  g_residency.in_use_bytes += alloc_size;
  *out_vaddr = entry->vaddr;
  return AEE_SUCCESS;
}

int hexagon_dsp_residency_acquire(int fd, uint64_t alloc_size,
                                  uint8_t **out_vaddr) {
  qurt_mutex_lock(&g_residency.mutex);
  int err = hexa_residency_acquire_locked(fd, alloc_size, out_vaddr);
  qurt_mutex_unlock(&g_residency.mutex);
  return err;
}

void hexagon_dsp_residency_release(int fd) {
  qurt_mutex_lock(&g_residency.mutex);
  hexa_residency_entry_t *entry = hexa_residency_find(fd);
  if (entry && entry->ref_count > 0 && --entry->ref_count == 0) {
    g_residency.in_use_bytes -= entry->size;
    // no unmap
  }
  // else entry->ref_count was decreased
  // or no-op on releasing an unknown buf
  qurt_mutex_unlock(&g_residency.mutex);
}

int hexagon_dsp_residency_invalidate_if_mapped(int fd, uint64_t offset,
                                               uint64_t length, bool flush) {
  int err = AEE_SUCCESS;
  qurt_mutex_lock(&g_residency.mutex);
  hexa_residency_entry_t *entry = hexa_residency_find(fd);
  if (entry && length > 0) {
    // according to doc, the only error is QURT_EVAL - invalid cache type
    err = qurt_mem_cache_clean((qurt_addr_t)(entry->vaddr + offset), length,
                               flush ? QURT_MEM_CACHE_FLUSH_INVALIDATE
                                     : QURT_MEM_CACHE_INVALIDATE,
                               QURT_MEM_DCACHE) == QURT_EOK
              ? AEE_SUCCESS
              : AEE_EFAILED;
  }
  qurt_mutex_unlock(&g_residency.mutex);
  return err;
}

void hexagon_dsp_residency_forget(int fd, bool *out_in_use) {
  *out_in_use = false;
  qurt_mutex_lock(&g_residency.mutex);
  hexa_residency_entry_t *entry = hexa_residency_find(fd);
  if (entry && entry->ref_count > 0) {
    FARF(ERROR,
         "HEXAGON-RUNTIME-ERROR: host frees fd %d while %u commands use it", fd,
         (unsigned)entry->ref_count);
    *out_in_use = true;
  } else if (entry) {
    hexa_residency_unmap_entry(entry);
  }
  qurt_mutex_unlock(&g_residency.mutex);
}

void hexagon_dsp_residency_clear(void) {
  qurt_mutex_lock(&g_residency.mutex);
  while (hexa_residency_evict_one()) {
  }
  // Only entries in use are left. Another session may still use them, but with
  // a single session this means an acquire was never released.
  if (g_residency.in_use_bytes > 0) {
    FARF(ERROR,
         "HEXAGON-RUNTIME-ERROR: %u buffers (%llu bytes) still in use when "
         "clearing the residency cache",
         (unsigned)g_residency.num_entries,
         (unsigned long long)g_residency.in_use_bytes);
  }
  qurt_mutex_unlock(&g_residency.mutex);
}

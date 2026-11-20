// Copyright 2026 RooflineAI GmbH
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "hexagon/dsp/residency.h"

#include <assert.h>
#include <stdlib.h>

#include "AEEStdErr.h"
#include "HAP_farf.h"
#include "HAP_mem.h"
#include "qurt.h"

/// fd_table value of an fd that is not mapped
#define HEXA_RESIDENCY_FD_UNMAPPED UINT32_MAX

/// a mapped allocation
typedef struct hexa_residency_entry_s {
  int fd;
  uint64_t size;
  uint8_t *vaddr;
  /// number of acquires not released yet
  uint32_t ref_count;
  /// value of the use counter at the last acquire, for LRU
  uint64_t last_use;
  /// what the data cache may hold of the allocation
  hexagon_dsp_cache_state_t cache;
} hexa_residency_entry_t;

/// cache state, guarded by mutex
struct hexagon_dsp_residency_s {
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
  /// index in entries of the mapped allocation of each fd below
  /// HEXAGON_DSP_RESIDENCY_FD_TABLE_SIZE, HEXA_RESIDENCY_FD_UNMAPPED if not
  /// mapped
  uint32_t fd_table[HEXAGON_DSP_RESIDENCY_FD_TABLE_SIZE];
  /// an fd did not fit fd_table, and that was logged
  bool fd_overflow_warned;
};

static hexa_residency_entry_t *
hexa_residency_find(hexagon_dsp_residency_t *residency, int fd) {
  if (fd < HEXAGON_DSP_RESIDENCY_FD_TABLE_SIZE) {
    uint32_t idx = residency->fd_table[fd];
    return idx != HEXA_RESIDENCY_FD_UNMAPPED ? &residency->entries[idx] : NULL;
  }
  // fds that do not fit the table: search the list
  for (uint32_t idx = 0; idx < residency->num_entries; ++idx) {
    if (residency->entries[idx].fd == fd) {
      return &residency->entries[idx];
    }
  }
  return NULL;
}

/// Record that fd is mapped at entries[idx].
static void hexa_residency_index_fd(hexagon_dsp_residency_t *residency, int fd,
                                    uint32_t idx) {
  if (fd < HEXAGON_DSP_RESIDENCY_FD_TABLE_SIZE) {
    residency->fd_table[fd] = idx;
    return;
  }
#if HEXAGON_DSP_RESIDENCY_WARN_FD_OVERFLOW
  if (!residency->fd_overflow_warned) {
    residency->fd_overflow_warned = true;
    FARF(RUNTIME_HIGH,
         "HEXAGON-RUNTIME-WARNING: fd %d does not fit the residency cache's fd "
         "table of %d entries, such fds are searched linearly",
         fd, HEXAGON_DSP_RESIDENCY_FD_TABLE_SIZE);
  }
#endif
}

/// Flush (HEXAGON_DSP_CACHE_OP_FLUSH), or flush and invalidate (any other
/// kind), the whole data cache. Invalidating alone would discard dirty lines
/// of other allocations. Their cache states are not updated.
static int hexa_residency_whole_cache_op(hexagon_dsp_cache_op_kind_t kind) {
  if (kind != HEXAGON_DSP_CACHE_OP_FLUSH) {
    kind = HEXAGON_DSP_CACHE_OP_FLUSH_INVALIDATE;
  }
  int err = qurt_mem_cache_clean(0, 0,
                                 kind == HEXAGON_DSP_CACHE_OP_FLUSH
                                     ? QURT_MEM_CACHE_FLUSH_ALL
                                     : QURT_MEM_CACHE_FLUSH_INVALIDATE_ALL,
                                 QURT_MEM_DCACHE);
  // according to doc, the only error is QURT_EVAL - invalid cache type
  return err == QURT_EOK ? AEE_SUCCESS : AEE_EFAILED;
}

/// Perform a cache operation on a range of a mapped allocation, or on the whole
/// data cache if that is cheaper.
/// need to hold the mutex, so another cant unmap while this is in progress
static int hexa_residency_cache_op(const hexa_residency_entry_t *entry,
                                   hexagon_dsp_cache_op_t op) {
  if (op.kind == HEXAGON_DSP_CACHE_OP_NONE || op.offset >= entry->size) {
    return AEE_SUCCESS;
  }
  if (op.length > entry->size - op.offset) {
    op.length = entry->size - op.offset;
  }
  if (hexagon_dsp_cache_op_uses_whole_cache(&op)) {
    return hexa_residency_whole_cache_op(op.kind);
  }
  qurt_mem_cache_op_t qurt_op = QURT_MEM_CACHE_FLUSH_INVALIDATE;
  switch (op.kind) {
  case HEXAGON_DSP_CACHE_OP_FLUSH:
    qurt_op = QURT_MEM_CACHE_FLUSH;
    break;
  case HEXAGON_DSP_CACHE_OP_INVALIDATE:
    qurt_op = QURT_MEM_CACHE_INVALIDATE;
    break;
  case HEXAGON_DSP_CACHE_OP_FLUSH_INVALIDATE:
    break;
  case HEXAGON_DSP_CACHE_OP_NONE:
    assert(false && "not reachable: returned early above");
    break;
  }
  int err =
      qurt_mem_cache_clean((qurt_addr_t)(entry->vaddr + op.offset),
                           (qurt_size_t)op.length, qurt_op, QURT_MEM_DCACHE);
  // according to doc, the only error is QURT_EVAL - invalid cache type
  return err == QURT_EOK ? AEE_SUCCESS : AEE_EFAILED;
}

/// Unmap an entry (ref_count must be 0) and remove it from the cache.
/// need to hold the mutex
static void hexa_residency_unmap_entry(hexagon_dsp_residency_t *residency,
                                       hexa_residency_entry_t *entry) {
  // Publish what was written, and leave no cache line at the addresses another
  // allocation may be mapped at next.
  hexa_residency_cache_op(
      entry, hexagon_dsp_cache_state_unmap(&entry->cache, entry->size));
  int err = HAP_munmap2(entry->vaddr, entry->size);
  if (err != AEE_SUCCESS) {
    FARF(ERROR, "HEXAGON-RUNTIME-ERROR: HAP_munmap2 of fd %d failed: 0x%x",
         entry->fd, err);
  }
  residency->mapped_bytes -= entry->size;
  // remove, by moving the last entry in its place
  if (entry->fd < HEXAGON_DSP_RESIDENCY_FD_TABLE_SIZE) {
    residency->fd_table[entry->fd] = HEXA_RESIDENCY_FD_UNMAPPED;
  }
  uint32_t idx = (uint32_t)(entry - residency->entries);
  *entry = residency->entries[--residency->num_entries];
  if (idx < residency->num_entries) {
    hexa_residency_index_fd(residency, entry->fd, idx);
  }
}

/// Unmap the entry not in use whose next use is furthest in the future
/// (Belady), the least recently used one among equals. Without a future (NULL),
/// all next uses are equal, so the least recently used one.
/// @return false if there is none
/// need to hold the mutex
static bool
hexa_residency_evict_one(hexagon_dsp_residency_t *residency,
                         const hexagon_dsp_residency_future_t *future) {
  hexa_residency_entry_t *victim = NULL;
  uint64_t victim_next_use = 0;
  for (uint32_t idx = 0; idx < residency->num_entries; ++idx) {
    hexa_residency_entry_t *entry = &residency->entries[idx];
    if (entry->ref_count != 0) {
      continue;
    }
    uint64_t next_use =
        future ? future->next_use(future->user_data, entry->fd) : 0;
    // no overflow check necessary, even at 1 increment per nanosecond, we would
    // have 580 years before overflow
    if (!victim || next_use > victim_next_use ||
        (next_use == victim_next_use && entry->last_use < victim->last_use)) {
      victim = entry;
      victim_next_use = next_use;
    }
  }
  // nothing to evict
  if (!victim) {
    return false;
  }
  hexa_residency_unmap_entry(residency, victim);
  return true;
}

int hexagon_dsp_residency_create(hexagon_dsp_residency_t **out_residency) {
  hexagon_dsp_residency_t *residency =
      (hexagon_dsp_residency_t *)calloc(1, sizeof(*residency));
  if (!residency) {
    return AEE_ENOMEMORY;
  }
  for (uint32_t fd = 0; fd < HEXAGON_DSP_RESIDENCY_FD_TABLE_SIZE; ++fd) {
    residency->fd_table[fd] = HEXA_RESIDENCY_FD_UNMAPPED;
  }
  qurt_mutex_init(&residency->mutex);
  *out_residency = residency;
  return AEE_SUCCESS;
}

void hexagon_dsp_residency_destroy(hexagon_dsp_residency_t *residency) {
  if (!residency) {
    return;
  }
  qurt_mutex_lock(&residency->mutex);
  while (hexa_residency_evict_one(residency, NULL)) {
  }
  // Only entries in use are left: an acquire was never released. Leave them
  // mapped, unmapping could crash a command still using them.
  if (residency->in_use_bytes > 0) {
    FARF(ERROR,
         "HEXAGON-RUNTIME-ERROR: %u buffers (%llu bytes) still in use when "
         "destroying the residency cache",
         (unsigned)residency->num_entries,
         (unsigned long long)residency->in_use_bytes);
  }
  qurt_mutex_unlock(&residency->mutex);
  qurt_mutex_destroy(&residency->mutex);
  free(residency->entries);
  free(residency);
}

int hexagon_dsp_residency_set_budget(hexagon_dsp_residency_t *residency,
                                     uint64_t budget_bytes) {
  if (budget_bytes == 0) {
    return AEE_EBADPARM;
  }
  qurt_mutex_lock(&residency->mutex);
  residency->budget = budget_bytes;
  qurt_mutex_unlock(&residency->mutex);
  return AEE_SUCCESS;
}

/// Release one hold of an entry.
/// need to hold the mutex
static void hexa_residency_release_entry(hexagon_dsp_residency_t *residency,
                                         hexa_residency_entry_t *entry) {
  if (entry->ref_count > 0 && --entry->ref_count == 0) {
    residency->in_use_bytes -= entry->size;
    // no unmap
  }
}

/// Map an allocation if not mapped yet and hold it
///  no cache maintenance necessary
/// need to hold the mutex
static int
hexa_residency_map_locked(hexagon_dsp_residency_t *residency, int fd,
                          uint64_t alloc_size,
                          const hexagon_dsp_residency_future_t *future,
                          hexa_residency_entry_t **out_entry) {
  hexa_residency_entry_t *entry = hexa_residency_find(residency, fd);
  if (entry) {
    if (entry->ref_count++ == 0) {
      // re-use existing entry
      residency->in_use_bytes += entry->size;
    }
    entry->last_use = ++residency->use_counter;
    *out_entry = entry;
    return AEE_SUCCESS;
  }
  if (alloc_size == 0) {
    return AEE_EBADPARM;
  }

  // Make room within the budget.
  while (residency->mapped_bytes + alloc_size > residency->budget) {
    if (!hexa_residency_evict_one(residency, future)) {
      // cant evict more: everything is in use
      if (residency->in_use_bytes > 0) {
        // might fit the budget if we could release more
        return AEE_EWOULDBLOCK;
      } else {
        // will never fit budget
        return AEE_ENOMEMORY;
      }
    }
  }

  // make room for another entry in the list if list is full
  if (residency->num_entries == residency->capacity) {
    uint32_t capacity = residency->capacity ? 2 * residency->capacity : 64;
    hexa_residency_entry_t *entries = (hexa_residency_entry_t *)realloc(
        residency->entries, capacity * sizeof(*entries));
    if (!entries) {
      return AEE_ENOMEMORY;
    }
    residency->entries = entries;
    residency->capacity = capacity;
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
    if (!hexa_residency_evict_one(residency, future)) {
      FARF(ERROR,
           "HEXAGON-RUNTIME-ERROR: HAP_mmap2 of fd %d (%llu bytes) failed "
           "with %llu bytes mapped, %llu bytes in use",
           fd, (unsigned long long)alloc_size,
           (unsigned long long)residency->mapped_bytes,
           (unsigned long long)residency->in_use_bytes);
      return residency->in_use_bytes > 0 ? AEE_EWOULDBLOCK : AEE_ENOMEMORY;
    }
  }

  hexa_residency_index_fd(residency, fd, residency->num_entries);
  entry = &residency->entries[residency->num_entries++];
  *entry = (hexa_residency_entry_t){
      .fd = fd,
      .size = alloc_size,
      .vaddr = (uint8_t *)vaddr,
      .ref_count = 1,
      .last_use = ++residency->use_counter,
  };
  hexagon_dsp_cache_state_init(&entry->cache);
  residency->mapped_bytes += alloc_size;
  residency->in_use_bytes += alloc_size;
  *out_entry = entry;
  return AEE_SUCCESS;
}

int hexagon_dsp_residency_acquire(hexagon_dsp_residency_t *residency, int fd,
                                  uint64_t alloc_size,
                                  const hexagon_dsp_residency_future_t *future,
                                  const hexagon_dsp_residency_access_t *access,
                                  uint8_t **out_vaddr) {
  assert(fd >= 0);
  qurt_mutex_lock(&residency->mutex);
  hexa_residency_entry_t *entry = NULL;
  int err =
      hexa_residency_map_locked(residency, fd, alloc_size, future, &entry);
  if (err == AEE_SUCCESS) {
    // perform cache maintainance if necessary
    [[maybe_unused]] int cache_err = hexa_residency_cache_op(
        entry, hexagon_dsp_cache_state_access(&entry->cache, access->kind,
                                              access->offset, access->length));
    // only fails for an invalid cache type
    assert(cache_err == AEE_SUCCESS);
    *out_vaddr = entry->vaddr;
  }
  qurt_mutex_unlock(&residency->mutex);
  return err;
}

void hexagon_dsp_residency_release(hexagon_dsp_residency_t *residency, int fd) {
  assert(fd >= 0);
  qurt_mutex_lock(&residency->mutex);
  hexa_residency_entry_t *entry = hexa_residency_find(residency, fd);
  // no-op on releasing an unknown buf
  if (entry) {
    hexa_residency_release_entry(residency, entry);
  }
  qurt_mutex_unlock(&residency->mutex);
}

int hexagon_dsp_residency_host_sync(hexagon_dsp_residency_t *residency, int fd,
                                    uint64_t offset, uint64_t length) {
  assert(fd >= 0);
  int err = AEE_SUCCESS;
  qurt_mutex_lock(&residency->mutex);
  hexa_residency_entry_t *entry = hexa_residency_find(residency, fd);
  if (entry) {
    err = hexa_residency_cache_op(entry, hexagon_dsp_cache_state_host_sync(
                                             &entry->cache, offset, length));
  } // else: no need for cache op, unmapped mem cant be in cache
  qurt_mutex_unlock(&residency->mutex);
  return err;
}

int hexagon_dsp_residency_after_bypass_write(hexagon_dsp_residency_t *residency,
                                             int fd, const uint8_t *vaddr,
                                             uint64_t length) {
  assert(fd >= 0);
  int err = AEE_SUCCESS;
  qurt_mutex_lock(&residency->mutex);
  hexa_residency_entry_t *entry = hexa_residency_find(residency, fd);
  if (entry && vaddr >= entry->vaddr) {
    uint64_t offset = (uint64_t)(vaddr - entry->vaddr);
    err = hexa_residency_cache_op(
        entry, hexagon_dsp_cache_state_after_bypass_write(&entry->cache, offset,
                                                          length));
  }
  qurt_mutex_unlock(&residency->mutex);
  return err;
}

int hexagon_dsp_residency_end_execution(hexagon_dsp_residency_t *residency) {
  int err = AEE_SUCCESS;
  qurt_mutex_lock(&residency->mutex);
  // The host may read anything, and one flush of the whole cache saves
  // tracking which allocations it reads.
  bool any_dirty = false;
  for (uint32_t idx = 0; idx < residency->num_entries; ++idx) {
    any_dirty =
        hexagon_dsp_cache_state_end_execution(&residency->entries[idx].cache) ||
        any_dirty;
  }
  if (any_dirty) {
    err = hexa_residency_whole_cache_op(HEXAGON_DSP_CACHE_OP_FLUSH);
  }
  qurt_mutex_unlock(&residency->mutex);
  return err;
}

void hexagon_dsp_residency_forget(hexagon_dsp_residency_t *residency, int fd,
                                  bool *out_in_use) {
  assert(fd >= 0);
  *out_in_use = false;
  qurt_mutex_lock(&residency->mutex);
  hexa_residency_entry_t *entry = hexa_residency_find(residency, fd);
  if (entry && entry->ref_count > 0) {
    FARF(ERROR,
         "HEXAGON-RUNTIME-ERROR: host frees fd %d while %u commands use it", fd,
         (unsigned)entry->ref_count);
    *out_in_use = true;
  } else if (entry) {
    hexa_residency_unmap_entry(residency, entry);
  }
  qurt_mutex_unlock(&residency->mutex);
}

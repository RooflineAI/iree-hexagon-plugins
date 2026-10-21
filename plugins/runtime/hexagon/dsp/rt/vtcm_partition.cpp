// Copyright 2026 RooflineAI GmbH
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "hexagon/dsp/rt/vtcm_partition.h"

#include <stdint.h>
#include <stdlib.h>

#include "HAP_farf.h"
#include "HexagonAPI.h"
#include "VTCMPool.h"
#include "hexagon/dsp/rt/runtime_state.h"
#include "hexagon/dsp/rt/vtcm_allocation.h"

// Allocation granularities of the partition, shared with the compiler that
// sizes the partition requirement. The front one is also hexagon-mlir's
// VtcmPool's, see VtcmPool::Allocate.
#define VTCM_FRONT_LARGE_GRANULE ((uintptr_t)HEXAGON_DSP_VTCM_LARGE_ALIGNMENT)
#define VTCM_BACK_SMALL_GRANULE                                                \
  ((uintptr_t)HEXAGON_DSP_VTCM_MIN_ALLOCATION_BYTES)

// alignment helper
// round up to the nearest multiple of alignment
inline uintptr_t align_up(uintptr_t value, uintptr_t alignment) {
  return (value + alignment - 1) & ~(alignment - 1);
}

// round down to the nearest multiple of alignment
inline uintptr_t align_down(uintptr_t value, uintptr_t alignment) {
  return value & ~(alignment - 1);
}

/// Rounds a partition request to the size actually taken from the pool.
static size_t partition_bytes(VtcmPool *vtcm_pool, size_t nbytes) {
  if (nbytes == HEXAGON_RT_VTCM_PARTITION_ALL) {
    // full pool
    // VtcmAllocatedBytes is the size of the pool (the name is misleading imho)
    return align_down(vtcm_pool->VtcmAllocatedBytes(),
                      VTCM_FRONT_LARGE_GRANULE);
  }
  return align_up(nbytes, VTCM_FRONT_LARGE_GRANULE);
}

void hexagon_rt_acquire_vtcm_partition(
    size_t nbytes, const char *dispatch_name,
    hexagon_rt_vtcm_partition_t *out_partition) {
  *out_partition = hexagon_rt_vtcm_partition_t{};
  out_partition->dispatch_name = dispatch_name;
  if (nbytes == 0) {
    return;
  }
  // VtcmPool allocates sizes that are a multiple of 2 KiB from the front of a
  // free block, which keeps partitions 2 KiB aligned.
  VtcmPool *vtcm_pool = HexagonAPI::Global()->getVtcmPool();
  nbytes = partition_bytes(vtcm_pool, nbytes);
  out_partition->base = static_cast<char *>(vtcm_pool->Allocate(nbytes));
  if (!out_partition->base) {
    // VtcmPool::Allocate has already reported the failure.
    abort();
  }
  out_partition->size = nbytes;
}

bool hexagon_rt_try_acquire_vtcm_partition(
    size_t nbytes, const char *dispatch_name,
    hexagon_rt_vtcm_partition_t *out_partition) {
  *out_partition = hexagon_rt_vtcm_partition_t{};
  out_partition->dispatch_name = dispatch_name;
  if (nbytes == 0) {
    return true;
  }
  VtcmPool *vtcm_pool = HexagonAPI::Global()->getVtcmPool();
  nbytes = partition_bytes(vtcm_pool, nbytes);
  out_partition->size = nbytes;
  // Allocate reports every failure, but not fitting yet is expected here. A
  // partition is a multiple of the 2 KiB large-allocation alignment, which
  // VtcmPool takes best-fit from any free block, so it fits exactly if the
  // largest free block is large enough. Only the master thread allocates and
  // frees partitions, so nothing changes the pool in between.
  if (vtcm_pool->getLargestFreeBlock() < nbytes) {
    return false;
  }
  out_partition->base = static_cast<char *>(vtcm_pool->Allocate(nbytes));
  return out_partition->base != NULL;
}

size_t hexagon_rt_vtcm_pool_size(void) {
  return partition_bytes(HexagonAPI::Global()->getVtcmPool(),
                         HEXAGON_RT_VTCM_PARTITION_ALL);
}

void hexagon_rt_release_vtcm_partition(hexagon_rt_vtcm_partition_t *partition) {
  if (partition->base) {
    HexagonAPI::Global()->getVtcmPool()->Free(partition->base, partition->size);
  }
  *partition = hexagon_rt_vtcm_partition_t{};
}

/// Reports a dispatch using more VTCM than it declared and aborts.
static void partition_overflow(const hexagon_rt_vtcm_partition_t *partition,
                               size_t nbytes) {
  FARF(ERROR,
       "HEXAGON-DISPATCH-RESOURCES-VIOLATION name=%s declared=%llu "
       "in_use=%llu requested=%llu",
       partition->dispatch_name ? partition->dispatch_name : "(unnamed)",
       (unsigned long long)partition->size,
       (unsigned long long)(partition->front + partition->back),
       (unsigned long long)nbytes);
  abort();
}

void *hexagon_rt_vtcm_partition_alloc(hexagon_rt_vtcm_partition_t *partition,
                                      size_t nbytes, uint64_t alignment) {
  if (nbytes == 0) {
    return NULL; // nothing to do
  }
  if (!partition) {
    FARF(ERROR, "VTCM allocation without a VTCM partition");
    abort();
  }

  if (alignment < VTCM_BACK_SMALL_GRANULE) {
    alignment = VTCM_BACK_SMALL_GRANULE;
  }
  if ((alignment & (alignment - 1)) != 0) { // power of 2?
    FARF(ERROR, "VTCM allocation with unsupported alignment %llu in %s",
         (unsigned long long)alignment,
         partition->dispatch_name ? partition->dispatch_name : "(unnamed)");
    abort();
  }
  if (alignment > partition->size) {
    partition_overflow(partition, nbytes);
  }

  // Offset of the lowest byte handed out from the back, relative to base.
  uintptr_t base = (uintptr_t)partition->base;
  size_t back_begin = partition->size - partition->back;

  if (alignment >= VTCM_FRONT_LARGE_GRANULE) {
    // The front stays a multiple of 2 KiB and base is 2 KiB aligned, so only
    // alignments beyond 2 KiB pad.
    size_t size = align_up(nbytes, VTCM_FRONT_LARGE_GRANULE);
    // add padding if necessary
    size_t begin = align_up(base + partition->front, alignment) // align address
                   - base; // convert to offset from base again
    if (size < nbytes || begin > back_begin || size > back_begin - begin) {
      partition_overflow(partition, nbytes);
    }
    partition->front = begin + size;
    return partition->base + begin;
  }

  // base is 2 KiB aligned, so aligning the offset aligns the address. The back
  // stays a multiple of 128 bytes, so only alignments beyond 128 bytes pad.
  size_t size = align_up(nbytes, VTCM_BACK_SMALL_GRANULE);
  if (size < nbytes || size > back_begin) {
    partition_overflow(partition, nbytes);
  }
  // align address
  size_t begin = align_down(back_begin - size, alignment);
  if (begin < partition->front) {
    partition_overflow(partition, nbytes);
  }
  partition->back = partition->size - begin;
  return partition->base + begin;
}

void *hexagon_runtime_vtcm_alloc(const hexagon_rt_state_t *runtime_state,
                                 size_t nbytes, uint64_t alignment) {
  return hexagon_rt_vtcm_partition_alloc(runtime_state->vtcm_partition, nbytes,
                                         alignment);
}

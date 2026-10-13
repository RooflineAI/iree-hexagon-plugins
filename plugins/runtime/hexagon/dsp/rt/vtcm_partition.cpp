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

// Allocation granularities of hexagon-mlir's VtcmPool, see VtcmPool::Allocate.
#define VTCM_FRONT_GRANULE ((uintptr_t)0x800)
#define VTCM_BACK_GRANULE ((uintptr_t)0x80)

// alignment helper
inline uintptr_t align_up(uintptr_t value, uintptr_t alignment) {
  return (value + alignment - 1) & ~(alignment - 1);
}

inline uintptr_t align_down(uintptr_t value, uintptr_t alignment) {
  return value & ~(alignment - 1);
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
  if (nbytes == HEXAGON_RT_VTCM_PARTITION_ALL) {
    // full pool
    // VtcmAllocatedBytes is the size of the pool (the name is misleading imho)
    nbytes = align_down(vtcm_pool->VtcmAllocatedBytes(), VTCM_FRONT_GRANULE);
  } else {
    nbytes = align_up(nbytes, VTCM_FRONT_GRANULE);
  }
  out_partition->base = static_cast<char *>(vtcm_pool->Allocate(nbytes));
  out_partition->size = nbytes;
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

  size_t available = partition->size - partition->front - partition->back;

  if (alignment == VTCM_FRONT_GRANULE) {
    size_t size = align_up(nbytes, VTCM_FRONT_GRANULE);
    if (size < nbytes || size > available) {
      partition_overflow(partition, nbytes);
    }
    char *ptr = partition->base + partition->front;
    partition->front += size;
    return ptr;
  }
  if (alignment == VTCM_BACK_GRANULE) {
    size_t size = align_up(nbytes, VTCM_BACK_GRANULE);
    if (size < nbytes || size > available) {
      partition_overflow(partition, nbytes);
    }
    partition->back += size;
    return partition->base + partition->size - partition->back;
  }
  FARF(ERROR, "VTCM allocation with unsupported alignment %llu in %s",
       (unsigned long long)alignment,
       partition->dispatch_name ? partition->dispatch_name : "(unnamed)");
  abort();
}

void *hexagon_runtime_vtcm_alloc(const hexagon_rt_state_t *runtime_state,
                                 size_t nbytes, uint64_t alignment) {
  return hexagon_rt_vtcm_partition_alloc(runtime_state->vtcm_partition, nbytes,
                                         alignment);
}

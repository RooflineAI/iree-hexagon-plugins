// Copyright 2026 RooflineAI GmbH
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef HEXAGON_DSP_RT_VTCM_PARTITION_H
#define HEXAGON_DSP_RT_VTCM_PARTITION_H

#include "hexagon/dsp/rt/runtime_state_fwd_decl.h"

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

/**
 * @brief Contiguous piece of VTCM the runtime carves out of the global VTCM
 *        pool for a single dispatch.
 *
 * A dispatch only allocates VTCM from its own partition, so dispatches never
 * share (and fragment) the global pool. Allocations are bumped from the two
 * ends of the partition and never freed individually: the runtime returns the
 * whole partition to the global pool once the dispatch is done. Running out of
 * space means the dispatch declared less VTCM than it uses, which aborts.
 */
typedef struct hexagon_rt_vtcm_partition_s {
  /// start of the partition, NULL if size is 0
  char *base;
  /// size of the partition in bytes
  size_t size;
  /// bytes handed out from the front (2 KiB granularity)
  size_t front;
  /// bytes handed out from the back (128 byte granularity)
  size_t back;
  /// name of the owning dispatch, for diagnostics only
  const char *dispatch_name;
} hexagon_rt_vtcm_partition_t;

/// Requests all of the global VTCM pool in hexagon_rt_acquire_vtcm_partition.
#define HEXAGON_RT_VTCM_PARTITION_ALL SIZE_MAX

/**
 * @brief Carve a dispatch's partition out of the global VTCM pool.
 *
 * Must only be called by the thread that owns the global pool. Aborts if the
 * pool has no contiguous free block of the requested size.
 * @param[in] nbytes size of the partition in bytes, rounded up to 2 KiB;
 *                   HEXAGON_RT_VTCM_PARTITION_ALL for all of the pool. A size
 *                   of 0 leaves the pool untouched and yields an empty
 *                   partition.
 * @param[in] dispatch_name name of the owning dispatch for diagnostics, may be
 *                          NULL
 * @param[out] out_partition partition to initialize
 */
void hexagon_rt_acquire_vtcm_partition(
    size_t nbytes, const char *dispatch_name,
    hexagon_rt_vtcm_partition_t *out_partition);

/**
 * @brief Return a partition to the global VTCM pool.
 *
 * Must only be called by the thread that owns the global pool.
 * @param[in,out] partition partition from hexagon_rt_acquire_vtcm_partition,
 *                empty afterwards
 */
void hexagon_rt_release_vtcm_partition(hexagon_rt_vtcm_partition_t *partition);

/**
 * @brief Allocate VTCM from a partition.
 *
 * Requests with an alignment of 2 KiB are rounded up to 2 KiB and taken from
 * the front, all others are rounded up to 128 bytes and taken from the back.
 * Aborts if the partition cannot accommodate the new allocation anymore
 * @param[in,out] partition partition to allocate from
 * @param[in] nbytes size of the memory to allocate in bytes
 * @param[in] alignment required alignment in bytes, 128 or 2048; any other
 *                      value aborts
 * @return pointer to allocated memory, NULL if nbytes == 0, aborts if out of
 * mem
 */
void *hexagon_rt_vtcm_partition_alloc(hexagon_rt_vtcm_partition_t *partition,
                                      size_t nbytes, uint64_t alignment);

/**
 * @brief Runtime entry point for VTCM allocations issued by generated code.
 * Overload of hexagon_rt_vtcm_partition_alloc, takes a hexagon_rt_state_t*
 * instead of the hexagon_rt_vtcm_partition_t* directly Results in easier
 * codegen
 */
void *hexagon_runtime_vtcm_alloc(const hexagon_rt_state_t *runtime_state,
                                 size_t nbytes, uint64_t alignment);

#ifdef __cplusplus
} // extern "C"
#endif // __cplusplus

#endif // #ifndef HEXAGON_DSP_RT_VTCM_PARTITION_H

// Copyright 2026 RooflineAI GmbH
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef HEXAGON_DSP_RT_VTCM_PARTITION_H
#define HEXAGON_DSP_RT_VTCM_PARTITION_H

#include "hexagon/dsp/rt/runtime_state_fwd_decl.h"

#include <stdbool.h>
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
  /// bytes handed out from the front (2 KiB granularity), including padding
  size_t front;
  /// bytes handed out from the back (128 byte granularity), including padding
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
 * @brief Like hexagon_rt_acquire_vtcm_partition, but fails instead of aborting
 *        if the pool has no contiguous free block of the requested size.
 *
 * Must only be called by the thread that owns the global pool.
 * @param[in] nbytes see hexagon_rt_acquire_vtcm_partition
 * @param[in] dispatch_name see hexagon_rt_acquire_vtcm_partition
 * @param[out] out_partition partition to initialize; on failure base is NULL
 *                           and size holds the (rounded) size that did not fit
 * @return true if the partition was acquired
 */
bool hexagon_rt_try_acquire_vtcm_partition(
    size_t nbytes, const char *dispatch_name,
    hexagon_rt_vtcm_partition_t *out_partition);

/**
 * @brief Size of the global VTCM pool in bytes, i.e. the size of a
 *        HEXAGON_RT_VTCM_PARTITION_ALL partition.
 *
 * Must only be called by the thread that owns the global pool. The first call
 * makes the calling thread the owner of the pool (and of the HMX lock).
 */
size_t hexagon_rt_vtcm_pool_size(void);

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
 * Requests aligned to at least 2 KiB are rounded up to 2 KiB and taken from
 * the front, all others are rounded up to 128 bytes and taken from the back.
 * Each allocation takes at most its rounded size plus its alignment minus the
 * granularity of its end, i.e. at most align_up(nbytes, 128) + alignment - 128
 * from the back and align_up(nbytes, 2048) + alignment - 2048 from the front.
 * Aborts if the partition cannot accommodate the new allocation anymore
 * @param[in,out] partition partition to allocate from
 * @param[in] nbytes size of the memory to allocate in bytes
 * @param[in] alignment required alignment in bytes, a power of two; values
 *                      below 128 (including 0) are raised to 128
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

// Copyright 2026 RooflineAI GmbH
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef HEXAGON_DSP_RESIDENCY_H
#define HEXAGON_DSP_RESIDENCY_H

#include <stdbool.h>
#include <stdint.h>

#include "hexagon/dsp/cache_state.h"

/**
 * Residency cache: maps RPCmem allocations into the DSP address space when
 * commands use them and keeps them mapped until the space is needed.
 *
 * The host registers every allocation once (FASTRPC_MAP_FD_DELAYED), which
 * does not use any DSP address space. Only about 3.75 GiB can be mapped at
 * once, and a mapping always covers the whole allocation, so the cache maps
 * whole allocations (HAP_mmap2) on first use and unmaps unused ones to stay
 * below a budget. A caller that knows which allocations it uses next passes a
 * future, then the cache unmaps the unused allocation used furthest in the
 * future (Belady), least recently used first among equals. Without a future,
 * it unmaps the least recently used one.
 * The eviction mechanism is not aware of fragmentation and will evict more,
 * until it fits
 *
 * The cache also tracks what the data cache may hold of each mapped allocation
 * and does the cache maintenance its accesses need (see cache_state.h). An
 * allocation that may have lines in the data cache is flushed and invalidated
 * before it is unmapped
 * Therefore, no cache invalidation on mapping is needed.
 *
 * Each RPC session owns one cache. The address space belongs to the DSP
 * process, which all sessions of a host process share, so the budgets of all
 * sessions together have to stay below what the DSP can map.
 */

/// File descriptors below this are looked up in a table indexed by the fd
/// (4 bytes each, per cache); larger ones in the list of mapped allocations,
/// a linear search.
#ifndef HEXAGON_DSP_RESIDENCY_FD_TABLE_SIZE
#define HEXAGON_DSP_RESIDENCY_FD_TABLE_SIZE 4096
#endif

/// Log once per cache when a file descriptor does not fit the table.
#ifndef HEXAGON_DSP_RESIDENCY_WARN_FD_OVERFLOW
#define HEXAGON_DSP_RESIDENCY_WARN_FD_OVERFLOW 1
#endif

/// A residency cache, see above.
typedef struct hexagon_dsp_residency_s hexagon_dsp_residency_t;

/// How a command accesses an allocation it acquires.
typedef struct hexagon_dsp_residency_access_s {
  hexagon_dsp_cache_access_t kind;
  /// range the command accesses, within the allocation
  uint64_t offset;
  uint64_t length;
} hexagon_dsp_residency_access_t;

/// Which allocations the caller uses next, for choosing what to unmap.
typedef struct hexagon_dsp_residency_future_s {
  /// Distance to the next use of fd, in the caller's units (e.g. buffer uses),
  /// UINT64_MAX if never. Called with the cache's mutex held, on the thread
  /// that acquires, only when an allocation has to be unmapped.
  uint64_t (*next_use)(void *user_data, int fd);
  void *user_data;
} hexagon_dsp_residency_future_t;

/// Create an empty cache. Its budget is 0 until set.
/// @retval AEE_SUCCESS for success, AEE_ENOMEMORY if out of memory
int hexagon_dsp_residency_create(hexagon_dsp_residency_t **out_residency);

/// Unmap all allocations and free the cache, e.g. when an RPC session closes.
/// Allocations still in use stay mapped (and are reported as an error).
void hexagon_dsp_residency_destroy(hexagon_dsp_residency_t *residency);

/// Set the max. bytes kept mapped.
/// @retval AEE_SUCCESS for success, AEE_EBADPARM for a budget of 0
int hexagon_dsp_residency_set_budget(hexagon_dsp_residency_t *residency,
                                     uint64_t budget_bytes);

/**
 * @brief Map an allocation if not mapped yet and hold it mapped until
 *        hexagon_dsp_residency_release. Does the cache maintenance needed
 *        before the access.
 * @param[in] fd file descriptor of the registered allocation
 * @param[in] alloc_size size of the whole allocation
 * @param[in] future next uses, to choose which allocation to unmap if there is
 *            no room; NULL to unmap the least recently used one
 * @param[in] access how the command accesses the allocation
 * @param[out] out_vaddr DSP address of the start of the allocation
 * @retval AEE_SUCCESS for success
 * @retval AEE_EWOULDBLOCK if it does not fit, even after unmapping all unused
 *         allocations; releasing allocations in use may make it fit
 */
int hexagon_dsp_residency_acquire(hexagon_dsp_residency_t *residency, int fd,
                                  uint64_t alloc_size,
                                  const hexagon_dsp_residency_future_t *future,
                                  const hexagon_dsp_residency_access_t *access,
                                  uint8_t **out_vaddr);

/// Stop holding an allocation acquired with hexagon_dsp_residency_acquire. It
/// stays mapped until its address space is needed.
void hexagon_dsp_residency_release(hexagon_dsp_residency_t *residency, int fd);

/**
 * @brief Before an execution, invalidate [offset, offset + length) of an
 *        allocation the host may have written since the last one. Nothing to
 *        do if it is not mapped (or has no lines in the data cache).
 * @param[in] fd file descriptor of the allocation
 * @param[in] offset start of the range within the allocation
 * @param[in] length length of the range
 * @retval AEE_SUCCESS for success
 */
int hexagon_dsp_residency_host_sync(hexagon_dsp_residency_t *residency, int fd,
                                    uint64_t offset, uint64_t length);

/**
 * @brief After an execution, flush the whole data cache if any allocation
 *        may be dirty, as the host may read any of them.
 * @retval AEE_SUCCESS for success
 */
int hexagon_dsp_residency_end_execution(hexagon_dsp_residency_t *residency);

/**
 * @brief Unmap an allocation the host is about to free (and deregister).
 * @param[in] fd file descriptor of the allocation
 * @param[out] out_in_use set if the allocation is still in use and was not
 *             unmapped; the host must not deregister it then
 */
void hexagon_dsp_residency_forget(hexagon_dsp_residency_t *residency, int fd,
                                  bool *out_in_use);

#endif // #ifndef HEXAGON_DSP_RESIDENCY_H

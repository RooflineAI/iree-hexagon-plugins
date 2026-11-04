// Copyright 2026 RooflineAI GmbH
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef HEXAGON_DSP_RESIDENCY_H
#define HEXAGON_DSP_RESIDENCY_H

#include <stdbool.h>
#include <stdint.h>

/**
 * Residency cache: maps RPCmem allocations into the DSP address space when
 * commands use them and keeps them mapped until the space is needed.
 *
 * The host registers every allocation once (FASTRPC_MAP_FD_DELAYED), which
 * does not use any DSP address space. Only about 3.75 GiB can be mapped at
 * once, and a mapping always covers the whole allocation, so the cache maps
 * whole allocations (HAP_mmap2) on first use and unmaps unused ones, least
 * recently used first, to stay below a budget.
 * The eviction mechanism is not aware of fragmentation and will evict more,
 * until it fits
 *
 * Unmapping frees the DSP addresses for another allocation, so an allocation
 * is flushed and invalidated from the data cache before it is unmapped
 * Therefore, no cache invalidation on mapping is needed
 *
 * The address space belongs to the DSP process, so there is one cache per
 * process, in principle shared by all RPC sessions.
 */

/// Set the max. bytes kept mapped.
/// @retval AEE_SUCCESS for success, AEE_EBADPARM for a budget of 0
int hexagon_dsp_residency_set_budget(uint64_t budget_bytes);

/**
 * @brief Map an allocation if not mapped yet and hold it mapped until
 *        hexagon_dsp_residency_release.
 * @param[in] fd file descriptor of the registered allocation
 * @param[in] alloc_size size of the whole allocation
 * @param[out] out_vaddr DSP address of the start of the allocation
 * @retval AEE_SUCCESS for success
 * @retval AEE_EWOULDBLOCK if it does not fit, even after unmapping all unused
 *         allocations; releasing allocations in use may make it fit
 */
int hexagon_dsp_residency_acquire(int fd, uint64_t alloc_size,
                                  uint8_t **out_vaddr);

/// Stop holding an allocation acquired with hexagon_dsp_residency_acquire. It
/// stays mapped until its address space is needed.
void hexagon_dsp_residency_release(int fd);

/**
 * @brief Invalidate [offset, offset + length) of an allocation in the data
 *        cache, if it is mapped. An allocation that is not mapped has no cache
 *        lines, as it was flushed and invalidated when it was unmapped.
 * @param[in] fd file descriptor of the allocation
 * @param[in] offset start of the range within the allocation
 * @param[in] length length of the range
 * @param[in] flush flush the range before invalidating it
 * @retval AEE_SUCCESS for success
 */
int hexagon_dsp_residency_invalidate_if_mapped(int fd, uint64_t offset,
                                               uint64_t length, bool flush);

/**
 * @brief Unmap an allocation the host is about to free (and deregister).
 * @param[in] fd file descriptor of the allocation
 * @param[out] out_in_use set if the allocation is still in use and was not
 *             unmapped; the host must not deregister it then
 */
void hexagon_dsp_residency_forget(int fd, bool *out_in_use);

/// Unmap all allocations not in use, e.g. when an RPC session closes.
void hexagon_dsp_residency_clear(void);

#endif // #ifndef HEXAGON_DSP_RESIDENCY_H

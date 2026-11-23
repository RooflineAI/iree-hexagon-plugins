// Copyright 2026 RooflineAI GmbH
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef HEXAGON_DSP_CACHE_STATE_H
#define HEXAGON_DSP_CACHE_STATE_H

#include <stdbool.h>
#include <stdint.h>

/**
 * What the DSP data cache may hold of one mapped allocation, and which cache
 * maintenance an access needs. Only bookkeeping: the caller performs the
 * returned operations (see residency.c).
 *
 * Besides the DSP's loads and stores (through the caches), two points can
 * access DDR directly:
 * - the host, between executions, and only allocations it can map;
 * - DMA transfers that bypass the cache, during a dispatch.
 * Coherent DMA goes through the cache like the cores' loads and stores
 * ("cached" accesses).
 *
 * Rules:
 * - A cached read may leave clean lines, a cached write dirty ones.
 * - Before a bypassed read, the dirty lines of its range are flushed.
 * - Before a bypassed write, the lines of its range are flushed and
 *   invalidated: a dirty line written back later would overwrite the result.
 *     After the dispatch, they are invalidated again, as lines fetched while
 *     the DMA was running are stale. This guards against someone "accidentally"
 *     prefetching the lines from this buffer, probably rare in practice.
 * - At the end of an execution, the whole cache is flushed, as the host may
 *   read any allocation. The lines stay in the cache, clean.
 * - Before an execution, the lines of ranges the host may have written are
 *   invalidated.
 * - An allocation is flushed and invalidated before it is unmapped, so an
 *   allocation that is not mapped has no lines.
 * A plain invalidate is only used where no dirty line can be, so dirty bytes of
 * a neighbour sharing a cache line at the edge of a range are never discarded.
 *
 * The dirty lines are tracked as one range per allocation: IREE packs many
 * tensors into one allocation, and a bypassed read only needs the dirty part of
 * its range flushed. A range removed from its middle leaves it unchanged, which
 * only costs flushing more than needed.
 */

/// Flushes of this many bytes or more flush the whole data cache: cleaning a
/// range costs about 17 us per MB, the whole cache costs the same for any size.
/// Measured on one OnePlus 13 (Snapdragon 8 Elite).
#ifndef HEXAGON_DSP_WHOLE_CACHE_BYTES
#define HEXAGON_DSP_WHOLE_CACHE_BYTES (8ull << 20)
#endif

/// Invalidates (and flush-invalidates) of this many bytes or more flush and
/// invalidate the whole data cache. That also drops the hot lines of every
/// other allocation, e.g. when unmapping a large allocation whose lines have
/// mostly left the cache already, so the threshold is higher than for flushes.
/// FIXME: measure, e.g. on a paging workload (small residency budget): the
/// ranged operation costs about 17 us per MB, dropping the cache costs the
/// refills of the commands around it.
#ifndef HEXAGON_DSP_WHOLE_CACHE_INVALIDATE_BYTES
#define HEXAGON_DSP_WHOLE_CACHE_INVALIDATE_BYTES (64ull << 20)
#endif

/// How a command accesses a range of an allocation.
typedef enum hexagon_dsp_cache_access_e {
  /// loads (or coherent DMA reads) only
  HEXAGON_DSP_CACHE_ACCESS_CACHED_READ,
  /// stores (or coherent DMA writes), possibly loads
  HEXAGON_DSP_CACHE_ACCESS_CACHED_WRITE,
  /// DMA reads that bypass the cache only
  HEXAGON_DSP_CACHE_ACCESS_BYPASS_READ,
  /// DMA writes that bypass the cache, possibly bypassed DMA reads
  HEXAGON_DSP_CACHE_ACCESS_BYPASS_WRITE,
} hexagon_dsp_cache_access_t;

/// Kind of a data cache operation.
typedef enum hexagon_dsp_cache_op_kind_e {
  HEXAGON_DSP_CACHE_OP_NONE,
  HEXAGON_DSP_CACHE_OP_FLUSH,
  HEXAGON_DSP_CACHE_OP_INVALIDATE,
  HEXAGON_DSP_CACHE_OP_FLUSH_INVALIDATE,
} hexagon_dsp_cache_op_kind_t;

/// A data cache operation on [offset, offset + length) of an allocation.
typedef struct hexagon_dsp_cache_op_s {
  hexagon_dsp_cache_op_kind_t kind;
  uint64_t offset;
  uint64_t length;
} hexagon_dsp_cache_op_t;

/// What the data cache may hold of one mapped allocation.
typedef struct hexagon_dsp_cache_state_s {
  /// byte offsets within the allocation: dirty lines may exist only where they
  /// overlap [dirty_begin, dirty_end); none if equal
  uint64_t dirty_begin;
  uint64_t dirty_end;
  /// whether the cache may hold any line of the allocation, clean or dirty,
  /// also outside [dirty_begin, dirty_end); false only if it was never
  /// accessed through the cache since it was mapped
  bool has_lines;
} hexagon_dsp_cache_state_t;

/// State of a newly mapped allocation: no lines.
void hexagon_dsp_cache_state_init(hexagon_dsp_cache_state_t *state);

/**
 * @brief Record an access of a command and return the operation needed before
 *        it.
 * @param[in] access how the command accesses the range
 * @param[in] offset start of the range within the allocation
 * @param[in] length length of the range
 */
hexagon_dsp_cache_op_t
hexagon_dsp_cache_state_access(hexagon_dsp_cache_state_t *state,
                               hexagon_dsp_cache_access_t access,
                               uint64_t offset, uint64_t length);

/// Operation needed once a command that wrote the range by bypassing the cache
/// has completed.
hexagon_dsp_cache_op_t
hexagon_dsp_cache_state_after_bypass_write(hexagon_dsp_cache_state_t *state,
                                           uint64_t offset, uint64_t length);

/// Operation needed before an execution on a range the host may have written
/// since the last one.
hexagon_dsp_cache_op_t
hexagon_dsp_cache_state_host_sync(hexagon_dsp_cache_state_t *state,
                                  uint64_t offset, uint64_t length);

/// Record that the whole cache is flushed after an execution. Returns whether
/// the allocation may have been dirty, i.e. the flush is needed.
bool hexagon_dsp_cache_state_end_execution(hexagon_dsp_cache_state_t *state);

/// Operation needed before unmapping an allocation of size bytes.
hexagon_dsp_cache_op_t
hexagon_dsp_cache_state_unmap(const hexagon_dsp_cache_state_t *state,
                              uint64_t size);

/// Whether an operation is cheaper on the whole data cache than on its range.
/// For an invalidate, that means flushing and invalidating the whole cache.
bool hexagon_dsp_cache_op_uses_whole_cache(const hexagon_dsp_cache_op_t *op);

#endif // #ifndef HEXAGON_DSP_CACHE_STATE_H

// Copyright 2026 RooflineAI GmbH
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef HEXAGON_DSP_NEXT_USE_H
#define HEXAGON_DSP_NEXT_USE_H

#include <stdint.h>

/**
 * Next-use lookup for evicting buffers (Belady's rule): which allocation is
 * used next when, given the order in which a command buffer uses allocations.
 *
 * A command buffer uses allocations in a fixed order, one use per position
 * 0 .. period - 1, and executions of it repeat: after position period - 1 comes
 * position 0 of the next execution. Several positions can use the same
 * allocation (fd).
 */

/// distance returned by hexagon_dsp_next_use_distance for an unused fd
#define HEXAGON_DSP_NEXT_USE_NEVER UINT64_MAX

/// use of an allocation (fd) at a position of the command order
typedef struct hexagon_dsp_use_s {
  int32_t fd;
  uint32_t position;
} hexagon_dsp_use_t;

/// Sort uses by fd, then by position, for hexagon_dsp_next_use_distance.
void hexagon_dsp_uses_sort(hexagon_dsp_use_t *uses, uint32_t num_uses);

/**
 * @brief Distance from a position to the next use of an allocation after it,
 *        wrapping around to the next execution.
 * @param[in] sorted_uses uses sorted with hexagon_dsp_uses_sort, each position
 *            below period
 * @param[in] num_uses number of entries in sorted_uses
 * @param[in] period number of positions of one execution
 * @param[in] fd allocation to look up
 * @param[in] position current position, below period
 * @return number of positions until the next use of fd (period if its only
 *         use is at position itself), HEXAGON_DSP_NEXT_USE_NEVER if it is not
 *         used at all
 */
uint64_t hexagon_dsp_next_use_distance(const hexagon_dsp_use_t *sorted_uses,
                                       uint32_t num_uses, uint32_t period,
                                       int32_t fd, uint32_t position);

#endif // #ifndef HEXAGON_DSP_NEXT_USE_H

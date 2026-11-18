// Copyright 2026 RooflineAI GmbH
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "hexagon/dsp/next_use.h"

#include <stdlib.h>

static int hexa_use_compare(const void *lhs_ptr, const void *rhs_ptr) {
  const hexagon_dsp_use_t *lhs = (const hexagon_dsp_use_t *)lhs_ptr;
  const hexagon_dsp_use_t *rhs = (const hexagon_dsp_use_t *)rhs_ptr;
  if (lhs->fd != rhs->fd) {
    return lhs->fd < rhs->fd ? -1 : 1;
  }
  if (lhs->position != rhs->position) {
    return lhs->position < rhs->position ? -1 : 1;
  }
  return 0;
}

void hexagon_dsp_uses_sort(hexagon_dsp_use_t *uses, uint32_t num_uses) {
  qsort(uses, num_uses, sizeof(*uses), hexa_use_compare);
}

/// index of the first use not before (fd, position), num_uses if none
static uint32_t hexa_uses_lower_bound(const hexagon_dsp_use_t *sorted_uses,
                                      uint32_t num_uses, int32_t fd,
                                      uint64_t position) {
  uint32_t begin = 0;
  uint32_t end = num_uses;
  while (begin < end) {
    uint32_t mid = begin + (end - begin) / 2;
    const hexagon_dsp_use_t *use = &sorted_uses[mid];
    if (use->fd < fd || (use->fd == fd && use->position < position)) {
      begin = mid + 1;
    } else {
      end = mid;
    }
  }
  return begin;
}

uint64_t hexagon_dsp_next_use_distance(const hexagon_dsp_use_t *sorted_uses,
                                       uint32_t num_uses, uint32_t period,
                                       int32_t fd, uint32_t position) {
  // next use in this execution
  uint32_t idx =
      hexa_uses_lower_bound(sorted_uses, num_uses, fd, (uint64_t)position + 1);
  if (idx < num_uses && sorted_uses[idx].fd == fd) {
    return sorted_uses[idx].position - position;
  }
  // FIXME: wrap around possibly not desired
  //  first use in the next execution
  idx = hexa_uses_lower_bound(sorted_uses, num_uses, fd, 0);
  if (idx < num_uses && sorted_uses[idx].fd == fd) {
    return (uint64_t)period - position + sorted_uses[idx].position;
  }
  return HEXAGON_DSP_NEXT_USE_NEVER;
}

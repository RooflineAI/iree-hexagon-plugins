// Copyright 2026 RooflineAI GmbH
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "hexagon/dsp/cache_state.h"

/// End of [offset, offset + length), saturated instead of wrapping.
static uint64_t hexa_cache_range_end(uint64_t offset, uint64_t length) {
  return length > UINT64_MAX - offset ? UINT64_MAX : offset + length;
}

/// Operation of the given kind on [begin, end), none if the range is empty.
static hexagon_dsp_cache_op_t hexa_cache_op(hexagon_dsp_cache_op_kind_t kind,
                                            uint64_t begin, uint64_t end) {
  if (begin >= end) {
    // nothing to do
    return (hexagon_dsp_cache_op_t){.kind = HEXAGON_DSP_CACHE_OP_NONE};
  }
  return (hexagon_dsp_cache_op_t){
      .kind = kind, .offset = begin, .length = end - begin};
}

void hexagon_dsp_cache_state_init(hexagon_dsp_cache_state_t *state) {
  *state = (hexagon_dsp_cache_state_t){};
}

hexagon_dsp_cache_op_t
hexagon_dsp_cache_state_access(hexagon_dsp_cache_state_t *state,
                               hexagon_dsp_cache_access_t access,
                               uint64_t offset, uint64_t length) {
  uint64_t begin = offset;
  uint64_t end = hexa_cache_range_end(offset, length);
  switch (access) {
  case HEXAGON_DSP_CACHE_ACCESS_CACHED_READ:
    state->has_lines = true;
    return hexa_cache_op(HEXAGON_DSP_CACHE_OP_NONE, 0, 0);
  case HEXAGON_DSP_CACHE_ACCESS_CACHED_WRITE:
    state->has_lines = true;
    state->dirty = true;
    return hexa_cache_op(HEXAGON_DSP_CACHE_OP_NONE, 0, 0);
  }
  // fallthrough
  // unknown access: assume anything
  state->has_lines = true;
  state->dirty = true;
  return hexa_cache_op(HEXAGON_DSP_CACHE_OP_FLUSH_INVALIDATE, begin, end);
}

hexagon_dsp_cache_op_t
hexagon_dsp_cache_state_host_sync(hexagon_dsp_cache_state_t *state,
                                  uint64_t offset, uint64_t length) {
  if (!state->has_lines) {
    return hexa_cache_op(HEXAGON_DSP_CACHE_OP_NONE, 0, 0);
  }
  uint64_t end = hexa_cache_range_end(offset, length);
  // The whole cache was flushed after the last execution, so none of the range
  // is dirty, invalidate suffices.
  return hexa_cache_op(HEXAGON_DSP_CACHE_OP_INVALIDATE, offset, end);
}

bool hexagon_dsp_cache_state_end_execution(hexagon_dsp_cache_state_t *state) {
  bool dirty = state->dirty;
  state->dirty = false;
  return dirty;
}

hexagon_dsp_cache_op_t
hexagon_dsp_cache_state_unmap(const hexagon_dsp_cache_state_t *state,
                              uint64_t size) {
  // Dirty lines come with has_lines. Hardware prefetching may have fetched
  // lines next to the ranges accessed, so always take the whole allocation.
  return hexa_cache_op(state->has_lines ? HEXAGON_DSP_CACHE_OP_FLUSH_INVALIDATE
                                        : HEXAGON_DSP_CACHE_OP_NONE,
                       0, size);
}

bool hexagon_dsp_cache_op_uses_whole_cache(const hexagon_dsp_cache_op_t *op) {
  switch (op->kind) {
  case HEXAGON_DSP_CACHE_OP_NONE:
    return false;
  case HEXAGON_DSP_CACHE_OP_FLUSH:
    return op->length >= HEXAGON_DSP_WHOLE_CACHE_BYTES;
  case HEXAGON_DSP_CACHE_OP_INVALIDATE:
  case HEXAGON_DSP_CACHE_OP_FLUSH_INVALIDATE:
    return op->length >= HEXAGON_DSP_WHOLE_CACHE_INVALIDATE_BYTES;
  }
  return false;
}

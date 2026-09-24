// Copyright 2026 RooflineAI GmbH
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "test_support.h"

#include "hexagon/dsp/ukernel/hmx/api.h"

#include <stdint.h>
#include <string.h>

static _Float16 input_value(uint32_t row, uint32_t col) {
  int32_t value = (int32_t)((row * 7u + col * 3u) % 23u) - 11;
  return (_Float16)value / (_Float16)4.0f;
}

static int run_pack_case(hmx_test_resource_t *resource, const char *case_name,
                         uint32_t rows, uint32_t cols, uint32_t stride,
                         uint32_t row_tiles, uint32_t col_tiles) {
  hmx_test_arena_reset(resource);
  size_t source_count = (size_t)rows * stride;
  size_t packed_size = (size_t)row_tiles * col_tiles * HMX_TEST_TILE_BYTES;
  size_t checked_size = packed_size + HMX_TEST_CANARY_BYTES;
  _Float16 *source =
      hmx_test_alloc(resource, source_count * sizeof(*source), 2048u);
  _Float16 *actual = hmx_test_alloc(resource, checked_size, 2048u);
  _Float16 *expected = hmx_test_alloc(resource, checked_size, 2048u);
  if (!source || !actual || !expected) {
    hmx_test_report_message("%s: VTCM arena exhausted\n", case_name);
    return 1;
  }

  memset(source, 0x5a, source_count * sizeof(*source));
  memset(actual, 0xa5, checked_size);
  memset(expected, 0xa5, checked_size);
  for (uint32_t row = 0; row < rows; ++row) {
    for (uint32_t col = 0; col < cols; ++col) {
      source[row * stride + col] = input_value(row, col);
    }
  }
  hmx_test_pack_reference(expected, source, stride, rows, cols, row_tiles,
                          col_tiles);
  iree_hexagon_hmx_pack_f16((uint32_t)(uintptr_t)actual,
                            (uint32_t)(uintptr_t)source, stride, rows, cols,
                            row_tiles, col_tiles);
  return hmx_test_compare_bytes(case_name, actual, expected, checked_size);
}

static int run_transposed_pack_case(hmx_test_resource_t *resource,
                                    const char *case_name, uint32_t interleave,
                                    uint32_t other, uint32_t stride,
                                    uint32_t interleave_tiles,
                                    uint32_t other_tiles) {
  hmx_test_arena_reset(resource);
  size_t source_count = (size_t)other * stride;
  size_t packed_size =
      (size_t)interleave_tiles * other_tiles * HMX_TEST_TILE_BYTES;
  size_t checked_size = packed_size + HMX_TEST_CANARY_BYTES;
  _Float16 *source =
      hmx_test_alloc(resource, source_count * sizeof(*source), 2048u);
  _Float16 *actual = hmx_test_alloc(resource, checked_size, 2048u);
  _Float16 *expected = hmx_test_alloc(resource, checked_size, 2048u);
  if (!source || !actual || !expected) {
    hmx_test_report_message("%s: VTCM arena exhausted\n", case_name);
    return 1;
  }

  memset(source, 0x5a, source_count * sizeof(*source));
  memset(actual, 0xa5, checked_size);
  memset(expected, 0xa5, checked_size);
  for (uint32_t row = 0; row < other; ++row) {
    for (uint32_t col = 0; col < interleave; ++col) {
      source[row * stride + col] = input_value(col, row);
    }
  }
  hmx_test_pack_transposed_reference(expected, source, stride, interleave,
                                     other, interleave_tiles, other_tiles);
  iree_hexagon_hmx_pack_transposed_f16(
      (uint32_t)(uintptr_t)actual, (uint32_t)(uintptr_t)source, stride,
      interleave, other, interleave_tiles, other_tiles);
  return hmx_test_compare_bytes(case_name, actual, expected, checked_size);
}

int main(int argc, char **argv) {
  (void)argc;
  (void)argv;
  hmx_test_report_begin("hmx_pack_sim_test");

  hmx_test_resource_t resource;
  int rc = hmx_test_resource_acquire(&resource, HMX_TEST_VTCM_BYTES);
  if (rc != 0) {
    hmx_test_report_message("resource acquisition failed: rc=%d\n", rc);
    return hmx_test_report_end(1);
  }

  int failures = 0;
  failures +=
      run_pack_case(&resource, "pack_full_32x32", 32u, 32u, 32u, 1u, 1u);
  failures +=
      run_pack_case(&resource, "pack_ragged_35x37", 35u, 37u, 41u, 3u, 3u);
  failures += run_transposed_pack_case(&resource, "pack_transposed_full_32x32",
                                       32u, 32u, 32u, 1u, 1u);
  failures += run_transposed_pack_case(
      &resource, "pack_transposed_ragged_35x37", 35u, 37u, 39u, 3u, 3u);

  rc = hmx_test_resource_release(&resource);
  if (rc != 0) {
    hmx_test_report_message("resource release failed: rc=%d\n", rc);
    ++failures;
  }
  return hmx_test_report_end(failures);
}

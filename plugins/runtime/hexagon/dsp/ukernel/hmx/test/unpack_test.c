// Copyright 2026 RooflineAI GmbH
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "test_support.h"

#include "hexagon/dsp/ukernel/hmx/api.h"

#include <stdint.h>
#include <string.h>

static _Float16 tile_value(uint32_t row, uint32_t col) {
  int32_t value = (int32_t)((row * 5u + col * 9u) % 17u) - 8;
  return (_Float16)value / (_Float16)2.0f;
}

static void initialize_packed_source(_Float16 *packed, _Float16 *logical,
                                     uint32_t rows, uint32_t cols,
                                     uint32_t stride, uint32_t row_tiles,
                                     uint32_t col_tiles) {
  memset(logical, 0, (size_t)rows * stride * sizeof(*logical));
  for (uint32_t row = 0; row < rows; ++row) {
    for (uint32_t col = 0; col < cols; ++col) {
      logical[row * stride + col] = tile_value(row, col);
    }
  }
  hmx_test_pack_reference(packed, logical, stride, rows, cols, row_tiles,
                          col_tiles);
}

static int run_unpack_f16_case(hmx_test_resource_t *resource,
                               const char *case_name, uint32_t rows,
                               uint32_t cols, uint32_t stride,
                               uint32_t row_tiles, uint32_t col_tiles) {
  hmx_test_arena_reset(resource);
  size_t packed_size = (size_t)row_tiles * col_tiles * HMX_TEST_TILE_BYTES;
  size_t checked_packed_size = packed_size + HMX_TEST_CANARY_BYTES;
  size_t destination_count = (size_t)rows * stride + 32u;
  _Float16 *packed = hmx_test_alloc(resource, checked_packed_size, 2048u);
  _Float16 *packed_expected =
      hmx_test_alloc(resource, checked_packed_size, 2048u);
  _Float16 *logical =
      hmx_test_alloc(resource, (size_t)rows * stride * sizeof(*logical), 2048u);
  _Float16 *actual =
      hmx_test_alloc(resource, destination_count * sizeof(*actual), 2048u);
  _Float16 *expected =
      hmx_test_alloc(resource, destination_count * sizeof(*expected), 2048u);
  if (!packed || !packed_expected || !logical || !actual || !expected) {
    hmx_test_report_message("%s: VTCM arena exhausted\n", case_name);
    return 1;
  }

  memset(packed, 0xa5, checked_packed_size);
  initialize_packed_source(packed, logical, rows, cols, stride, row_tiles,
                           col_tiles);
  memcpy(packed_expected, packed, checked_packed_size);
  // A nonzero initializer verifies destination-passing accumulation. Padding
  // columns and the trailing canary must retain the initializer unchanged.
  for (size_t i = 0; i < destination_count; ++i) {
    actual[i] = (_Float16)4.0f;
    expected[i] = (_Float16)4.0f;
  }
  for (uint32_t row = 0; row < rows; ++row) {
    for (uint32_t col = 0; col < cols; ++col) {
      expected[row * stride + col] += logical[row * stride + col];
    }
  }

  iree_hexagon_hmx_unpack_acc_f16_to_f16((uint32_t)(uintptr_t)actual,
                                         (uint32_t)(uintptr_t)packed, stride,
                                         rows, cols, row_tiles, col_tiles);
  int failures =
      hmx_test_compare_f16(case_name, actual, expected, destination_count);
  failures += hmx_test_compare_bytes(case_name, packed, packed_expected,
                                     checked_packed_size);
  return failures;
}

static int run_unpack_f32_case(hmx_test_resource_t *resource,
                               const char *case_name, uint32_t rows,
                               uint32_t cols, uint32_t stride,
                               uint32_t row_tiles, uint32_t col_tiles) {
  hmx_test_arena_reset(resource);
  size_t packed_size = (size_t)row_tiles * col_tiles * HMX_TEST_TILE_BYTES;
  size_t checked_packed_size = packed_size + HMX_TEST_CANARY_BYTES;
  size_t destination_count = (size_t)rows * stride + 32u;
  _Float16 *packed = hmx_test_alloc(resource, checked_packed_size, 2048u);
  _Float16 *packed_expected =
      hmx_test_alloc(resource, checked_packed_size, 2048u);
  _Float16 *logical =
      hmx_test_alloc(resource, (size_t)rows * stride * sizeof(*logical), 2048u);
  float *actual =
      hmx_test_alloc(resource, destination_count * sizeof(*actual), 2048u);
  float *expected =
      hmx_test_alloc(resource, destination_count * sizeof(*expected), 2048u);
  if (!packed || !packed_expected || !logical || !actual || !expected) {
    hmx_test_report_message("%s: VTCM arena exhausted\n", case_name);
    return 1;
  }

  memset(packed, 0xa5, checked_packed_size);
  initialize_packed_source(packed, logical, rows, cols, stride, row_tiles,
                           col_tiles);
  memcpy(packed_expected, packed, checked_packed_size);
  for (size_t i = 0; i < destination_count; ++i) {
    actual[i] = 16.0f;
    expected[i] = 16.0f;
  }
  for (uint32_t row = 0; row < rows; ++row) {
    for (uint32_t col = 0; col < cols; ++col) {
      expected[row * stride + col] += (float)logical[row * stride + col];
    }
  }

  iree_hexagon_hmx_unpack_acc_f16_to_f32((uint32_t)(uintptr_t)actual,
                                         (uint32_t)(uintptr_t)packed, stride,
                                         rows, cols, row_tiles, col_tiles);
  int failures =
      hmx_test_compare_f32(case_name, actual, expected, destination_count);
  failures += hmx_test_compare_bytes(case_name, packed, packed_expected,
                                     checked_packed_size);
  return failures;
}

int main(int argc, char **argv) {
  (void)argc;
  (void)argv;
  hmx_test_report_begin("hmx_unpack_sim_test");

  hmx_test_resource_t resource;
  int rc = hmx_test_resource_acquire(&resource, HMX_TEST_VTCM_BYTES);
  if (rc != 0) {
    hmx_test_report_message("resource acquisition failed: rc=%d\n", rc);
    return hmx_test_report_end(1);
  }

  int failures = 0;
  failures += run_unpack_f16_case(&resource, "unpack_f16_add_full_32x32", 32u,
                                  32u, 32u, 1u, 1u);
  failures += run_unpack_f16_case(&resource, "unpack_f16_add_ragged_35x37", 35u,
                                  37u, 41u, 3u, 3u);
  failures += run_unpack_f32_case(&resource, "unpack_f32_full_32x32", 32u, 32u,
                                  32u, 1u, 1u);
  failures += run_unpack_f32_case(&resource, "unpack_f32_ragged_35x37", 35u,
                                  37u, 41u, 3u, 3u);

  rc = hmx_test_resource_release(&resource);
  if (rc != 0) {
    hmx_test_report_message("resource release failed: rc=%d\n", rc);
    ++failures;
  }
  return hmx_test_report_end(failures);
}

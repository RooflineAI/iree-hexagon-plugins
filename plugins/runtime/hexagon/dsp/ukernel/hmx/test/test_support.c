// Copyright 2026 RooflineAI GmbH
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "test_support.h"

#include "AEEStdErr.h"
#include "HAP_compute_res.h"
#include "HAP_power.h"

#include <stdarg.h>
#include <stdio.h>
#include <string.h>

static int hmx_test_set_power(void *context, int power_up) {
  HAP_power_request_t request = {0};
  request.type = HAP_power_set_HMX;
  request.hmx.power_up = power_up ? 1 : 0;
  return HAP_power_set(context, &request);
}

static int hmx_test_hvx_power_up(void) {
  static int power_state;
  HAP_power_request_t request = {0};
  request.type = HAP_power_set_HVX;
  request.hvx.power_up = 1;
  return HAP_power_set(&power_state, &request);
}

static int hmx_test_set_high_perf_dcvs(void) {
  static int power_state;
  HAP_power_request_t request = {0};
  request.type = HAP_power_set_DCVS_v3;
  request.dcvs_v3.set_dcvs_enable = 1;
  request.dcvs_v3.dcvs_enable = 1;
  request.dcvs_v3.dcvs_option = HAP_DCVS_V2_PERFORMANCE_MODE;
  request.dcvs_v3.set_latency = 1;
  request.dcvs_v3.latency = 100;
  request.dcvs_v3.set_core_params = 1;
  request.dcvs_v3.core_params.target_corner = HAP_DCVS_VCORNER_MAX;
  request.dcvs_v3.core_params.min_corner = HAP_DCVS_VCORNER_MAX;
  request.dcvs_v3.core_params.max_corner = HAP_DCVS_VCORNER_MAX;
  request.dcvs_v3.set_bus_params = 1;
  request.dcvs_v3.bus_params.target_corner = HAP_DCVS_VCORNER_MAX;
  request.dcvs_v3.bus_params.min_corner = HAP_DCVS_VCORNER_MAX;
  request.dcvs_v3.bus_params.max_corner = HAP_DCVS_VCORNER_MAX;
  request.dcvs_v3.set_sleep_disable = 1;
  request.dcvs_v3.sleep_disable = 1;
  return HAP_power_set(&power_state, &request);
}

int hmx_test_resource_acquire(hmx_test_resource_t *resource,
                              uint32_t vtcm_size) {
  if (!resource || vtcm_size == 0) {
    return AEE_EBADPARM;
  }

  memset(resource, 0, sizeof(*resource));
  resource->vtcm_size = vtcm_size;

  // This is the resource lifecycle proven by hmx_resource_acquire in the
  // L-roro/07-29-hmx-matmul-playground benchmark. Keep the ordering aligned
  // with that implementation so simulator tests exercise the same setup used
  // while the microkernels were developed.
  (void)hmx_test_set_high_perf_dcvs();

  compute_res_attr_t attributes;
  int rc = HAP_compute_res_attr_init(&attributes);
  if (rc == AEE_SUCCESS) {
    rc = HAP_compute_res_attr_set_vtcm_param(&attributes, vtcm_size,
                                             /*b_single_page=*/0);
  }
  if (rc == AEE_SUCCESS) {
    rc = HAP_compute_res_attr_set_hmx_param(&attributes, 1);
  }
  if (rc != AEE_SUCCESS) {
    return rc;
  }

  resource->context_id = HAP_compute_res_acquire(&attributes, 1000000);
  if (resource->context_id == 0) {
    return AEE_ERESOURCENOTFOUND;
  }
  resource->vtcm_base =
      (uint8_t *)HAP_compute_res_attr_get_vtcm_ptr(&attributes);

  int power_rc = hmx_test_set_power(&resource->hmx_power_ctx, 1);
  (void)hmx_test_hvx_power_up();
  rc = HAP_compute_res_hmx_lock(resource->context_id);
  if (!resource->vtcm_base || power_rc != AEE_SUCCESS || rc != AEE_SUCCESS) {
    (void)hmx_test_set_power(&resource->hmx_power_ctx, 0);
    (void)HAP_compute_res_release(resource->context_id);
    resource->context_id = 0;
    resource->vtcm_base = NULL;
    return rc != AEE_SUCCESS
               ? rc
               : (power_rc != AEE_SUCCESS ? power_rc : AEE_ENOMEMORY);
  }

  if (((uintptr_t)resource->vtcm_base % 2048u) != 0u) {
    (void)hmx_test_resource_release(resource);
    return AEE_EBADMEMALIGN;
  }
  return AEE_SUCCESS;
}

int hmx_test_resource_release(hmx_test_resource_t *resource) {
  if (!resource || resource->context_id == 0) {
    return AEE_EBADPARM;
  }

  int rc = HAP_compute_res_hmx_unlock(resource->context_id);
  (void)hmx_test_set_power(&resource->hmx_power_ctx, 0);
  int release_rc = HAP_compute_res_release(resource->context_id);
  if (rc == AEE_SUCCESS) {
    rc = release_rc;
  }

  memset(resource, 0, sizeof(*resource));
  return rc;
}

void hmx_test_arena_reset(hmx_test_resource_t *resource) {
  resource->arena_offset = 0;
}

void *hmx_test_alloc(hmx_test_resource_t *resource, size_t size,
                     size_t alignment) {
  if (!resource || !resource->vtcm_base || alignment == 0 ||
      (alignment & (alignment - 1u)) != 0u) {
    return NULL;
  }
  size_t offset = (resource->arena_offset + alignment - 1u) & ~(alignment - 1u);
  if (offset > resource->vtcm_size || size > resource->vtcm_size - offset) {
    return NULL;
  }
  resource->arena_offset = (uint32_t)(offset + size);
  return resource->vtcm_base + offset;
}

void hmx_test_report_begin(const char *suite_name) {
  hmx_test_report_message("RUN %s\n", suite_name);
}

void hmx_test_report_message(const char *format, ...) {
  va_list args;
  va_start(args, format);
  vprintf(format, args);
  va_end(args);
}

int hmx_test_report_end(int failures) {
  hmx_test_report_message("%s (%d failure%s)\n", failures ? "FAIL" : "PASS",
                          failures, failures == 1 ? "" : "s");
  return failures ? 1 : 0;
}

int hmx_test_compare_bytes(const char *case_name, const void *actual,
                           const void *expected, size_t size) {
  const uint8_t *actual_bytes = (const uint8_t *)actual;
  const uint8_t *expected_bytes = (const uint8_t *)expected;
  for (size_t i = 0; i < size; ++i) {
    if (actual_bytes[i] != expected_bytes[i]) {
      hmx_test_report_message("%s: byte %u expected=0x%02x actual=0x%02x\n",
                              case_name, (unsigned)i, expected_bytes[i],
                              actual_bytes[i]);
      return 1;
    }
  }
  return 0;
}

int hmx_test_compare_f16(const char *case_name, const _Float16 *actual,
                         const _Float16 *expected, size_t count) {
  return hmx_test_compare_bytes(case_name, actual, expected,
                                count * sizeof(_Float16));
}

int hmx_test_compare_f32(const char *case_name, const float *actual,
                         const float *expected, size_t count) {
  for (size_t i = 0; i < count; ++i) {
    if (actual[i] != expected[i]) {
      hmx_test_report_message("%s: element %u expected=%f actual=%f\n",
                              case_name, (unsigned)i, (double)expected[i],
                              (double)actual[i]);
      return 1;
    }
  }
  return 0;
}

void hmx_test_pack_reference(_Float16 *dest, const _Float16 *source,
                             uint32_t source_stride, uint32_t actual_rows,
                             uint32_t actual_cols, uint32_t row_tiles,
                             uint32_t col_tiles) {
  memset(dest, 0, row_tiles * col_tiles * HMX_TEST_TILE_BYTES);
  for (uint32_t row = 0; row < actual_rows; ++row) {
    for (uint32_t col = 0; col < actual_cols; ++col) {
      uint32_t row_tile = row / HMX_TEST_TILE;
      uint32_t col_tile = col / HMX_TEST_TILE;
      if (row_tile >= row_tiles || col_tile >= col_tiles) {
        continue;
      }
      uint32_t tile_index = row_tile * col_tiles + col_tile;
      uint32_t tile_row = row % HMX_TEST_TILE;
      uint32_t tile_col = col % HMX_TEST_TILE;
      uint32_t packed_index = tile_index * HMX_TEST_TILE_ELEMENTS +
                              (tile_row / 2u) * 64u + 2u * tile_col +
                              (tile_row & 1u);
      dest[packed_index] = source[row * source_stride + col];
    }
  }
}

void hmx_test_pack_transposed_reference(_Float16 *dest, const _Float16 *source,
                                        uint32_t source_stride,
                                        uint32_t actual_interleave,
                                        uint32_t actual_other,
                                        uint32_t interleave_tiles,
                                        uint32_t other_tiles) {
  memset(dest, 0, interleave_tiles * other_tiles * HMX_TEST_TILE_BYTES);
  for (uint32_t interleave = 0; interleave < actual_interleave; ++interleave) {
    for (uint32_t other = 0; other < actual_other; ++other) {
      uint32_t interleave_tile = interleave / HMX_TEST_TILE;
      uint32_t other_tile = other / HMX_TEST_TILE;
      if (interleave_tile >= interleave_tiles || other_tile >= other_tiles) {
        continue;
      }
      uint32_t tile_index = interleave_tile * other_tiles + other_tile;
      uint32_t tile_row = interleave % HMX_TEST_TILE;
      uint32_t tile_col = other % HMX_TEST_TILE;
      uint32_t packed_index = tile_index * HMX_TEST_TILE_ELEMENTS +
                              (tile_row / 2u) * 64u + 2u * tile_col +
                              (tile_row & 1u);
      dest[packed_index] = source[other * source_stride + interleave];
    }
  }
}

void hmx_test_matmul_reference(_Float16 *dest, const _Float16 *lhs,
                               const _Float16 *rhs,
                               uint32_t accumulation_count) {
  for (uint32_t row = 0; row < HMX_TEST_TILE; ++row) {
    for (uint32_t col = 0; col < HMX_TEST_TILE; ++col) {
      float value = 0.0f;
      for (uint32_t depth = 0; depth < HMX_TEST_TILE; ++depth) {
        value += (float)lhs[row * HMX_TEST_TILE + depth] *
                 (float)rhs[depth * HMX_TEST_TILE + col];
      }
      dest[row * HMX_TEST_TILE + col] =
          (_Float16)(value * (float)accumulation_count);
    }
  }
}

// Copyright 2026 RooflineAI GmbH
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "test_support.h"

#include "hexagon/dsp/ukernel/hmx/api.h"

#include <stdint.h>
#include <string.h>

static _Float16 activation_value(uint32_t row, uint32_t col) {
  int32_t value = (int32_t)((row * 3u + col * 5u) % 9u) - 4;
  return (_Float16)value / (_Float16)2.0f;
}

// Use exactly representable f16 values and a sparse, non-identity weight. The
// off-diagonal term detects operand-order, tile-layout, and lane-mixing bugs
// that an identity-only matmul could hide.
static void initialize_matrices(_Float16 *activation, _Float16 *weight) {
  for (uint32_t row = 0; row < HMX_TEST_TILE; ++row) {
    for (uint32_t col = 0; col < HMX_TEST_TILE; ++col) {
      activation[row * HMX_TEST_TILE + col] = activation_value(row, col);
      weight[row * HMX_TEST_TILE + col] =
          row == col ? (_Float16)1.0f
                     : (col == (row + 1u) % HMX_TEST_TILE ? (_Float16)0.5f
                                                          : (_Float16)0.0f);
    }
  }
}

// The setup microkernel must initialize exactly the 256-byte bias block. The
// surrounding 0xa5 bytes act as a canary for an oversized write.
static void initialize_expected_config(uint8_t *expected, size_t size) {
  memset(expected, 0xa5, size);
  memset(expected, 0, 256u);
  for (uint32_t col = 0; col < HMX_TEST_TILE; ++col) {
    expected[col * 4u + 1u] = 0x3c;
  }
}

// End-to-end baseline: a clear accumulator followed by one MMA and one read
// must exactly match the independent scalar 32x32x32 matmul. Comparing the
// trailing canary also checks that acc_read writes exactly one packed tile.
static int run_single_mma(const _Float16 *packed_activation,
                          const _Float16 *packed_weight, _Float16 *output,
                          const _Float16 *expected) {
  memset(output, 0xa5, HMX_TEST_TILE_BYTES + HMX_TEST_CANARY_BYTES);
  iree_hexagon_hmx_acc_clear_f16();
  iree_hexagon_hmx_mma_f16((uint32_t)(uintptr_t)packed_activation,
                           (uint32_t)(uintptr_t)packed_weight);
  iree_hexagon_hmx_acc_read_f16((uint32_t)(uintptr_t)output);
  return hmx_test_compare_bytes("mma_single", output, expected,
                                HMX_TEST_TILE_BYTES + HMX_TEST_CANARY_BYTES);
}

// Accumulator persistence: two MMAs without an intervening clear must produce
// twice the scalar product, rather than overwriting the first MMA result.
static int run_accumulating_mma(const _Float16 *packed_activation,
                                const _Float16 *packed_weight, _Float16 *output,
                                const _Float16 *expected) {
  memset(output, 0xa5, HMX_TEST_TILE_BYTES + HMX_TEST_CANARY_BYTES);
  iree_hexagon_hmx_acc_clear_f16();
  iree_hexagon_hmx_mma_f16((uint32_t)(uintptr_t)packed_activation,
                           (uint32_t)(uintptr_t)packed_weight);
  iree_hexagon_hmx_mma_f16((uint32_t)(uintptr_t)packed_activation,
                           (uint32_t)(uintptr_t)packed_weight);
  iree_hexagon_hmx_acc_read_f16((uint32_t)(uintptr_t)output);
  return hmx_test_compare_bytes("mma_accumulates", output, expected,
                                HMX_TEST_TILE_BYTES + HMX_TEST_CANARY_BYTES);
}

// Accumulator isolation: clearing after a nonzero MMA must remove all prior
// state, so a subsequent read returns an all-zero packed tile.
static int run_clear_case(const _Float16 *packed_activation,
                          const _Float16 *packed_weight, _Float16 *output,
                          const _Float16 *expected) {
  memset(output, 0xa5, HMX_TEST_TILE_BYTES + HMX_TEST_CANARY_BYTES);
  iree_hexagon_hmx_acc_clear_f16();
  iree_hexagon_hmx_mma_f16((uint32_t)(uintptr_t)packed_activation,
                           (uint32_t)(uintptr_t)packed_weight);
  iree_hexagon_hmx_acc_clear_f16();
  iree_hexagon_hmx_acc_read_f16((uint32_t)(uintptr_t)output);
  return hmx_test_compare_bytes("acc_clear_removes_prior_state", output,
                                expected,
                                HMX_TEST_TILE_BYTES + HMX_TEST_CANARY_BYTES);
}

int main(int argc, char **argv) {
  (void)argc;
  (void)argv;
  hmx_test_report_begin("hmx_accumulator_sim_test");

  hmx_test_resource_t resource;
  int rc = hmx_test_resource_acquire(&resource, HMX_TEST_VTCM_BYTES);
  if (rc != 0) {
    hmx_test_report_message("resource acquisition failed: rc=%d\n", rc);
    return hmx_test_report_end(1);
  }

  uint8_t *config = hmx_test_alloc(&resource, 2048u, 2048u);
  uint8_t *expected_config = hmx_test_alloc(&resource, 2048u, 2048u);
  _Float16 *activation = hmx_test_alloc(&resource, HMX_TEST_TILE_BYTES, 2048u);
  _Float16 *weight = hmx_test_alloc(&resource, HMX_TEST_TILE_BYTES, 2048u);
  _Float16 *packed_activation =
      hmx_test_alloc(&resource, HMX_TEST_TILE_BYTES, 2048u);
  _Float16 *packed_weight =
      hmx_test_alloc(&resource, HMX_TEST_TILE_BYTES, 2048u);
  _Float16 *output = hmx_test_alloc(
      &resource, HMX_TEST_TILE_BYTES + HMX_TEST_CANARY_BYTES, 2048u);
  _Float16 *logical_expected =
      hmx_test_alloc(&resource, HMX_TEST_TILE_BYTES, 2048u);
  _Float16 *expected = hmx_test_alloc(
      &resource, HMX_TEST_TILE_BYTES + HMX_TEST_CANARY_BYTES, 2048u);

  int failures = 0;
  if (!config || !expected_config || !activation || !weight ||
      !packed_activation || !packed_weight || !output || !logical_expected ||
      !expected) {
    hmx_test_report_message("VTCM exhausted\n");
    failures = 1;
  } else {
    memset(config, 0xa5, 2048u);
    initialize_expected_config(expected_config, 2048u);
    initialize_matrices(activation, weight);
    hmx_test_pack_reference(packed_activation, activation, HMX_TEST_TILE,
                            HMX_TEST_TILE, HMX_TEST_TILE, 1u, 1u);
    hmx_test_pack_reference(packed_weight, weight, HMX_TEST_TILE, HMX_TEST_TILE,
                            HMX_TEST_TILE, 1u, 1u);

    // Bias setup is tested independently from MMA state. Checking all 2048
    // bytes verifies both its contents and the untouched config canary.
    iree_hexagon_hmx_acc_setup_read_f16((uint32_t)(uintptr_t)config);
    failures += hmx_test_compare_bytes("acc_setup_read_f16_config", config,
                                       expected_config, 2048u);

    memset(expected, 0xa5, HMX_TEST_TILE_BYTES + HMX_TEST_CANARY_BYTES);
    hmx_test_matmul_reference(logical_expected, activation, weight, 1u);
    hmx_test_pack_reference(expected, logical_expected, HMX_TEST_TILE,
                            HMX_TEST_TILE, HMX_TEST_TILE, 1u, 1u);
    failures +=
        run_single_mma(packed_activation, packed_weight, output, expected);

    memset(expected, 0xa5, HMX_TEST_TILE_BYTES + HMX_TEST_CANARY_BYTES);
    hmx_test_matmul_reference(logical_expected, activation, weight, 2u);
    hmx_test_pack_reference(expected, logical_expected, HMX_TEST_TILE,
                            HMX_TEST_TILE, HMX_TEST_TILE, 1u, 1u);
    failures += run_accumulating_mma(packed_activation, packed_weight, output,
                                     expected);

    memset(logical_expected, 0, HMX_TEST_TILE_BYTES);
    memset(expected, 0xa5, HMX_TEST_TILE_BYTES + HMX_TEST_CANARY_BYTES);
    hmx_test_pack_reference(expected, logical_expected, HMX_TEST_TILE,
                            HMX_TEST_TILE, HMX_TEST_TILE, 1u, 1u);
    failures +=
        run_clear_case(packed_activation, packed_weight, output, expected);
  }

  rc = hmx_test_resource_release(&resource);
  if (rc != 0) {
    hmx_test_report_message("resource release failed: rc=%d\n", rc);
    ++failures;
  }
  return hmx_test_report_end(failures);
}

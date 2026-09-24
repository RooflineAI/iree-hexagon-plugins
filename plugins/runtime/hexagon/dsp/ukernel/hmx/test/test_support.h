// Copyright 2026 RooflineAI GmbH
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef ROOF_HEXAGON_RUNTIME_DSP_UKERNEL_HMX_TEST_TEST_SUPPORT_H_
#define ROOF_HEXAGON_RUNTIME_DSP_UKERNEL_HMX_TEST_TEST_SUPPORT_H_

#include <stddef.h>
#include <stdint.h>

#define HMX_TEST_TILE 32u
#define HMX_TEST_TILE_ELEMENTS (HMX_TEST_TILE * HMX_TEST_TILE)
#define HMX_TEST_TILE_BYTES (HMX_TEST_TILE_ELEMENTS * sizeof(_Float16))
#define HMX_TEST_VTCM_BYTES (128u * 1024u)
#define HMX_TEST_CANARY_BYTES 64u

// Owns the compute-resource context and the test's bump-allocation state.
// Call hmx_test_resource_release after a successful acquisition.
typedef struct {
  uint32_t context_id;
  uint8_t *vtcm_base;
  uint32_t vtcm_size;
  uint32_t arena_offset;
  int hmx_power_ctx;
} hmx_test_resource_t;

/**
 * Acquires one compute-resource context containing VTCM and HMX, applies the
 * playground's power sequence, and locks HMX on the calling thread.
 *
 * On success, `resource->vtcm_base` owns `vtcm_size` bytes aligned for HMX
 * operands. Returns AEE_SUCCESS or an SDK error code.
 */
int hmx_test_resource_acquire(hmx_test_resource_t *resource,
                              uint32_t vtcm_size);

/** Unlocks HMX, powers it down, and releases an acquired context. */
int hmx_test_resource_release(hmx_test_resource_t *resource);

/**
 * Rewinds the test-only VTCM bump allocator. Existing allocations may be
 * overwritten by subsequent hmx_test_alloc calls.
 */
void hmx_test_arena_reset(hmx_test_resource_t *resource);

/**
 * Suballocates `size` bytes from the acquired VTCM.
 * `alignment` must be a nonzero power of two. Returns NULL on invalid input or
 * if the arena has insufficient space; individual allocations cannot be freed.
 */
void *hmx_test_alloc(hmx_test_resource_t *resource, size_t size,
                     size_t alignment);

/** Prints the start marker for a simulator test suite. */
void hmx_test_report_begin(const char *suite_name);

/** Prints one formatted diagnostic through the simulator's standard output. */
void hmx_test_report_message(const char *format, ...);

/** Prints the suite result and returns a process exit code for `failures`. */
int hmx_test_report_end(int failures);

/**
 * Compares `size` bytes exactly and reports the first differing byte.
 * Returns 0 for equality and 1 for a mismatch.
 */
int hmx_test_compare_bytes(const char *case_name, const void *actual,
                           const void *expected, size_t size);

/** Bit-exact f16 comparison; reports the first differing byte. */
int hmx_test_compare_f16(const char *case_name, const _Float16 *actual,
                         const _Float16 *expected, size_t count);

/** Exact f32 comparison; reports the first differing element. */
int hmx_test_compare_f32(const char *case_name, const float *actual,
                         const float *expected, size_t count);

/**
 * Scalar reference for the standard packer. Produces a row-major grid of
 * 32x32 HMX tiles and zero-fills ragged regions and extra padding tiles.
 */
void hmx_test_pack_reference(_Float16 *dest, const _Float16 *source,
                             uint32_t source_stride, uint32_t actual_rows,
                             uint32_t actual_cols, uint32_t row_tiles,
                             uint32_t col_tiles);

/**
 * Scalar reference for packing a source stored as [other][interleave].
 * Produces the same packed layout as the logical, non-transposed matrix.
 */
void hmx_test_pack_transposed_reference(_Float16 *dest, const _Float16 *source,
                                        uint32_t source_stride,
                                        uint32_t actual_interleave,
                                        uint32_t actual_other,
                                        uint32_t interleave_tiles,
                                        uint32_t other_tiles);

/**
 * Computes a scalar 32x32x32 f16 matmul reference. `accumulation_count`
 * models issuing the same HMX MMA repeatedly before reading the accumulator.
 */
void hmx_test_matmul_reference(_Float16 *dest, const _Float16 *lhs,
                               const _Float16 *rhs,
                               uint32_t accumulation_count);

#endif // ROOF_HEXAGON_RUNTIME_DSP_UKERNEL_HMX_TEST_TEST_SUPPORT_H_

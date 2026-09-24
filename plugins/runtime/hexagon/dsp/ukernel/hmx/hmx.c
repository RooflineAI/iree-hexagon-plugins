// Copyright 2026 RooflineAI GmbH
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "hexagon/dsp/ukernel/hmx/api.h"

#include <hexagon_protos.h>
#include <hexagon_types.h>
#include <hmx_hexagon_protos.h>
#include <stdint.h>
#include <string.h>

// Instruction operands for a full 32x32x32 FP16 HMX tile. These match the
// immediates emitted by hexKL's micro HMX primitive
//   activation Rt: spatial mask plus full input-channel range
//   weight Rt:     dW distance to the last 128 B weight vector
//   convert Rs:    bias set 0, no feedback, clear+swap accumulator
#define HMX_ACT_RT 0x7FFFu
#define HMX_WEI_RT 1920u
#define HMX_CVT_RS 0u
#define HMX_BIAS_BYTES 256u
#define HMX_TILE 32u
#define HMX_TILE_BYTES 2048u
#define HMX_TILE_VECTORS 16u

typedef uint8_t hmx_half_vector_t __attribute__((vector_size(64), aligned(64)));
typedef union {
  HVX_Vector vector;
  hmx_half_vector_t halves[2];
} hmx_vector_halves_t;

void iree_hexagon_hmx_acc_setup_read_f16(uint32_t config) {
  uint8_t *bias = (uint8_t *)config;
  memset(bias, 0, HMX_BIAS_BYTES);
  for (uint32_t c = 0; c < HMX_TILE; ++c) {
    bias[c * 4u + 1u] = 0x3Cu;
  }

  // Program bias set 0 as a raw pass-through for FP16 accumulator read-out:
  // scale=1.0 and all additive terms zero. The later convert/read instruction
  // selects this state with Rs=0.
  Q6_bias_mxmem2_A(bias);
}

void iree_hexagon_hmx_acc_clear_f16(void) { Q6_mxclracc_hf(); }

// Number of valid rows/cols (0..32) of tile `tile` given the actual logical
// extent `actual` of that axis. Tiles beyond `actual` are pure padding.
static inline uint32_t iree_hexagon_hmx_tile_extent(uint32_t actual,
                                                    uint32_t tile) {
  uint32_t base = tile * HMX_TILE;
  if (base >= actual) {
    return 0u;
  }
  uint32_t rem = actual - base;
  return rem < HMX_TILE ? rem : HMX_TILE;
}

// Rearrange one row-major f16 tile at (base_row, base_col) into the HMX
// tile-major layout. `valid_rows`/`valid_cols` (<=32) bound the region present
// in the source; the remainder of the 32x32 tile is zero-filled so the HMX
// unit (which has no masking) sees zeros in the padding.
static void iree_hexagon_hmx_pack_rm_tile_to_hmx_layout(
    uint8_t *dest, const _Float16 *source, uint32_t source_stride,
    uint32_t base_row, uint32_t base_col, uint32_t valid_rows,
    uint32_t valid_cols) {
  HVX_Vector *packed = (HVX_Vector *)dest;
  // Fast path: a full tile whose source rows are vector-aligned (stride is a
  // multiple of the 32-element vector). Ragged strides (e.g. K not a multiple
  // of 32) fall back to the scalar-gather path below.
  if (valid_rows >= HMX_TILE && valid_cols >= HMX_TILE &&
      (source_stride % HMX_TILE) == 0u) {
    for (uint32_t row_pair = 0; row_pair < HMX_TILE / 2u; ++row_pair) {
      const _Float16 *row0 =
          source + (base_row + 2u * row_pair) * source_stride + base_col;
      const _Float16 *row1 = row0 + source_stride;
      hmx_vector_halves_t combined;
      combined.halves[0] = *(const hmx_half_vector_t *)row0;
      combined.halves[1] = *(const hmx_half_vector_t *)row1;
      packed[row_pair] = Q6_Vh_vshuff_Vh(combined.vector);
    }
    return;
  }
  // Boundary/ragged tile: gather the valid sub-region into a zeroed 32x32
  // scratch (scalar, alignment-agnostic), then shuffle the padded scratch.
  _Float16 scratch[HMX_TILE * HMX_TILE] __attribute__((aligned(128)));
  memset(scratch, 0, sizeof(scratch));
  for (uint32_t r = 0; r < valid_rows; ++r) {
    const _Float16 *src_row =
        source + (base_row + r) * source_stride + base_col;
    for (uint32_t c = 0; c < valid_cols; ++c) {
      scratch[r * HMX_TILE + c] = src_row[c];
    }
  }
  for (uint32_t row_pair = 0; row_pair < HMX_TILE / 2u; ++row_pair) {
    hmx_vector_halves_t combined;
    combined.halves[0] =
        *(const hmx_half_vector_t *)(scratch + (2u * row_pair) * HMX_TILE);
    combined.halves[1] =
        *(const hmx_half_vector_t *)(scratch + (2u * row_pair + 1u) * HMX_TILE);
    packed[row_pair] = Q6_Vh_vshuff_Vh(combined.vector);
  }
}

// Pack a (possibly ragged) row-major matrix of `actual_rows` x `actual_cols`
// into a `row_tiles` x `col_tiles` grid of HMX tiles. The grid is sized to the
// padded upper bound; tiles beyond the valid region are zero-filled so the
// staged source can stay exactly the logical size.
void iree_hexagon_hmx_pack_f16(uint32_t dest, uint32_t source,
                               uint32_t source_stride, uint32_t actual_rows,
                               uint32_t actual_cols, uint32_t row_tiles,
                               uint32_t col_tiles) {
  uint8_t *packed_tiles = (uint8_t *)dest;
  const _Float16 *matrix = (const _Float16 *)source;
  for (uint32_t row_tile = 0; row_tile < row_tiles; ++row_tile) {
    uint32_t valid_rows = iree_hexagon_hmx_tile_extent(actual_rows, row_tile);
    for (uint32_t col_tile = 0; col_tile < col_tiles; ++col_tile) {
      uint32_t valid_cols = iree_hexagon_hmx_tile_extent(actual_cols, col_tile);
      uint8_t *tile =
          packed_tiles + (row_tile * col_tiles + col_tile) * HMX_TILE_BYTES;
      if (valid_rows == 0u || valid_cols == 0u) {
        memset(tile, 0, HMX_TILE_BYTES);
        continue;
      }
      iree_hexagon_hmx_pack_rm_tile_to_hmx_layout(
          tile, matrix, source_stride, row_tile * HMX_TILE, col_tile * HMX_TILE,
          valid_rows, valid_cols);
    }
  }
}

// One stage of the 32x32 transpose ladder. For every vector pair (i, i+d) whose
// low index satisfies (i / d) % 2 == 0, Q6_W_vshuff_VVR(v[i+d], v[i], rt)
// shuffles the pair at |rt|/2-halfword block granularity; the low result
// replaces v[i] and the high result replaces v[i+d]. Each such op is a pure
// permutation of the tile's 10-bit linear index (it swaps one vector-select bit
// with one within-vector offset bit), so a fixed sequence of these stages
// realizes the transpose. Ported from the playground transposed HMX kernel
// (hmx_matmul_hexkl_dma_hoisted_transposed.c), bit-verified on the simulator.
static inline void
iree_hexagon_hmx_vshuff_cross_stage(HVX_Vector v[HMX_TILE_VECTORS], uint32_t d,
                                    int32_t rt) {
  HVX_Vector nv[HMX_TILE_VECTORS];
  for (uint32_t i = 0; i < HMX_TILE_VECTORS; ++i) {
    nv[i] = v[i];
  }
  for (uint32_t i = 0; i < HMX_TILE_VECTORS; ++i) {
    if ((i / d) % 2u == 0u) {
      HVX_VectorPair w = Q6_W_vshuff_VVR(v[i + d], v[i], rt);
      nv[i] = Q6_V_lo_W(w);
      nv[i + d] = Q6_V_hi_W(w);
    }
  }
  for (uint32_t i = 0; i < HMX_TILE_VECTORS; ++i) {
    v[i] = nv[i];
  }
}

// Fast path: pack one full HMX tile from a TRANSPOSED row-major source.
// `source` is stored [other][interleave] row-major with stride `source_stride`
// (elements) between other-rows, so the value logically at tile[i][o] (i =
// interleave axis, e.g. k for the weight; o = other axis, e.g. n) lives at
// source[(base_o + o) * source_stride + base_i + i]. The result is
// byte-identical to packing the non-transposed tile[i][o] with the row-pair
// vshuffh packer, i.e. dst[(i/2)*64 + 2*o + (i&1)] = tile[i][o]. The tile
// window is loaded two other-rows per vector (the 32 lanes index the interleave
// axis), then a 5-stage vshuff ladder performs the 32x32 transpose AND the HMX
// row-pair interleave in one pass. Requires a full, vector-aligned tile (see
// the gate in the driver).
static void iree_hexagon_hmx_pack_transposed_full_tile(uint8_t *dest,
                                                       const _Float16 *source,
                                                       uint32_t source_stride,
                                                       uint32_t base_interleave,
                                                       uint32_t base_other) {
  HVX_Vector v[HMX_TILE_VECTORS];
  for (uint32_t p = 0; p < HMX_TILE_VECTORS; ++p) {
    const _Float16 *o0 = source +
                         (size_t)(base_other + 2u * p) * source_stride +
                         base_interleave;
    const _Float16 *o1 = o0 + source_stride;
    hmx_vector_halves_t combined;
    combined.halves[0] = *(const hmx_half_vector_t *)o0;
    combined.halves[1] = *(const hmx_half_vector_t *)o1;
    v[p] = combined.vector;
  }

  iree_hexagon_hmx_vshuff_cross_stage(v, 1u, -4);
  iree_hexagon_hmx_vshuff_cross_stage(v, 8u, -8);
  iree_hexagon_hmx_vshuff_cross_stage(v, 4u, -8);
  iree_hexagon_hmx_vshuff_cross_stage(v, 2u, -8);
  iree_hexagon_hmx_vshuff_cross_stage(v, 1u, -4);

  HVX_Vector *packed = (HVX_Vector *)dest;
  for (uint32_t q = 0; q < HMX_TILE_VECTORS; ++q) {
    packed[q] = v[q];
  }
}

// Pack a (possibly ragged) TRANSPOSED row-major matrix into an
// `interleave_tiles` x `other_tiles` grid of HMX tiles. `source` is stored
// [actual_other][actual_interleave] (e.g. a weight B^T laid out [n][k]); the
// produced grid is byte-identical to packing the non-transposed
// [actual_interleave][actual_other] matrix, so the mm loop reads it exactly
// like a standard operand pack. Mirrors iree_hexagon_hmx_pack_f16: full aligned
// tiles take the vshuff-ladder fast path, boundary tiles gather the valid
// transposed sub-region into a zeroed 32x32 scratch (already interleave-major)
// and reuse the standard row-pair vshuffh, and fully-padding tiles are zeroed.
void iree_hexagon_hmx_pack_transposed_f16(uint32_t dest, uint32_t source,
                                          uint32_t source_stride,
                                          uint32_t actual_interleave,
                                          uint32_t actual_other,
                                          uint32_t interleave_tiles,
                                          uint32_t other_tiles) {
  uint8_t *packed_tiles = (uint8_t *)dest;
  const _Float16 *matrix_t = (const _Float16 *)source;
  for (uint32_t i_tile = 0; i_tile < interleave_tiles; ++i_tile) {
    uint32_t valid_i = iree_hexagon_hmx_tile_extent(actual_interleave, i_tile);
    for (uint32_t o_tile = 0; o_tile < other_tiles; ++o_tile) {
      uint32_t valid_o = iree_hexagon_hmx_tile_extent(actual_other, o_tile);
      uint8_t *tile =
          packed_tiles + (i_tile * other_tiles + o_tile) * HMX_TILE_BYTES;
      if (valid_i == 0u || valid_o == 0u) {
        memset(tile, 0, HMX_TILE_BYTES);
        continue;
      }
      uint32_t base_i = i_tile * HMX_TILE;
      uint32_t base_o = o_tile * HMX_TILE;
      // Fast path: a full tile whose transposed source rows are vector-aligned.
      if (valid_i >= HMX_TILE && valid_o >= HMX_TILE &&
          (source_stride % HMX_TILE) == 0u) {
        iree_hexagon_hmx_pack_transposed_full_tile(
            tile, matrix_t, source_stride, base_i, base_o);
        continue;
      }
      // Boundary/ragged tile: gather the valid transposed sub-region into a
      // zeroed interleave-major scratch (scratch[i][o] =
      // source[base_o+o][base_i+i]), which is exactly the standard operand tile
      // layout, then shuffle it.
      _Float16 scratch[HMX_TILE * HMX_TILE] __attribute__((aligned(128)));
      memset(scratch, 0, sizeof(scratch));
      for (uint32_t o = 0; o < valid_o; ++o) {
        const _Float16 *src_row =
            matrix_t + (size_t)(base_o + o) * source_stride + base_i;
        for (uint32_t i = 0; i < valid_i; ++i) {
          scratch[i * HMX_TILE + o] = src_row[i];
        }
      }
      HVX_Vector *packed = (HVX_Vector *)tile;
      for (uint32_t row_pair = 0; row_pair < HMX_TILE / 2u; ++row_pair) {
        hmx_vector_halves_t combined;
        combined.halves[0] =
            *(const hmx_half_vector_t *)(scratch + (2u * row_pair) * HMX_TILE);
        combined.halves[1] =
            *(const hmx_half_vector_t *)(scratch +
                                         (2u * row_pair + 1u) * HMX_TILE);
        packed[row_pair] = Q6_Vh_vshuff_Vh(combined.vector);
      }
    }
  }
}

// Convert one HMX accumulator-read tile to f32 and add its valid `valid_rows` x
// `valid_cols` sub-region into the (possibly strided) row-major destination.
static void iree_hexagon_hmx_unpack_acc_tile(float *dest, const HVX_Vector *src,
                                             uint32_t dest_stride,
                                             uint32_t valid_rows,
                                             uint32_t valid_cols) {
  // Fast path: a full tile whose destination rows are vector-aligned.
  if (valid_rows >= HMX_TILE && valid_cols >= HMX_TILE &&
      (dest_stride % HMX_TILE) == 0u) {
    for (uint32_t row_pair = 0; row_pair < HMX_TILE / 2u; ++row_pair) {
      HVX_VectorPair rows_f32 = Q6_Wsf_vcvt_Vhf(src[row_pair]);
      float *row0 = dest + (2u * row_pair) * dest_stride;
      float *row1 = dest + (2u * row_pair + 1u) * dest_stride;
      *(HVX_Vector *)row0 =
          Q6_Vsf_vadd_VsfVsf(*(HVX_Vector *)row0, Q6_V_lo_W(rows_f32));
      *(HVX_Vector *)row1 =
          Q6_Vsf_vadd_VsfVsf(*(HVX_Vector *)row1, Q6_V_hi_W(rows_f32));
    }
    return;
  }
  // Boundary/ragged tile: convert to a scratch, then add only the valid region.
  float scratch[HMX_TILE * HMX_TILE] __attribute__((aligned(128)));
  for (uint32_t row_pair = 0; row_pair < HMX_TILE / 2u; ++row_pair) {
    HVX_VectorPair rows_f32 = Q6_Wsf_vcvt_Vhf(src[row_pair]);
    *(HVX_Vector *)(scratch + (2u * row_pair) * HMX_TILE) = Q6_V_lo_W(rows_f32);
    *(HVX_Vector *)(scratch + (2u * row_pair + 1u) * HMX_TILE) =
        Q6_V_hi_W(rows_f32);
  }
  for (uint32_t r = 0; r < valid_rows; ++r) {
    float *drow = dest + r * dest_stride;
    const float *srow = scratch + r * HMX_TILE;
    for (uint32_t c = 0; c < valid_cols; ++c) {
      drow[c] += srow[c];
    }
  }
}

// Unpack a `row_tiles` x `col_tiles` grid of HMX accumulator-read tiles into a
// (possibly ragged) row-major f32 destination of `actual_rows` x `actual_cols`.
// Only the valid region of each boundary tile is written, so the destination
// stays exactly the logical size (padding tiles are skipped).
void iree_hexagon_hmx_unpack_acc_f16_to_f32(
    uint32_t dest, uint32_t source, uint32_t dest_stride, uint32_t actual_rows,
    uint32_t actual_cols, uint32_t row_tiles, uint32_t col_tiles) {
  // Preserve linalg.matmul DPS semantics: the compiler keeps the f32 init tile
  // in `dest`, while HMX computes the product from a cleared hardware
  // accumulator. Add the converted read-out into that tile.
  float *dest_base = (float *)dest;
  const uint8_t *src_tiles = (const uint8_t *)source;
  for (uint32_t row_tile = 0; row_tile < row_tiles; ++row_tile) {
    uint32_t valid_rows = iree_hexagon_hmx_tile_extent(actual_rows, row_tile);
    if (valid_rows == 0u) {
      continue;
    }
    for (uint32_t col_tile = 0; col_tile < col_tiles; ++col_tile) {
      uint32_t valid_cols = iree_hexagon_hmx_tile_extent(actual_cols, col_tile);
      if (valid_cols == 0u) {
        continue;
      }
      const HVX_Vector *src =
          (const HVX_Vector *)(src_tiles + (row_tile * col_tiles + col_tile) *
                                               HMX_TILE_BYTES);
      float *dtile =
          dest_base + (row_tile * HMX_TILE) * dest_stride + col_tile * HMX_TILE;
      iree_hexagon_hmx_unpack_acc_tile(dtile, src, dest_stride, valid_rows,
                                       valid_cols);
    }
  }
}

// De-interleave one HMX accumulator-read tile (row-pair interleaved f16, the
// same layout the packer produces) into its valid `valid_rows` x `valid_cols`
// sub-region of the (possibly strided) row-major f16 destination. Unlike the
// f32 variant this does NOT widen: `Q6_Vh_vdeal_Vh` is the inverse of the
// packer's `Q6_Vh_vshuff_Vh`, splitting the interleaved vector into its two
// 32-wide f16 rows (the two 64-byte halves). The valid region is added to the
// destination to preserve linalg.matmul destination-passing semantics.
static void iree_hexagon_hmx_unpack_acc_tile_f16(_Float16 *dest,
                                                 const HVX_Vector *src,
                                                 uint32_t dest_stride,
                                                 uint32_t valid_rows,
                                                 uint32_t valid_cols) {
  // Fast path: a full tile whose destination rows are 64-byte (half-vector)
  // aligned (a 32-wide f16 row is 64 bytes).
  if (valid_rows >= HMX_TILE && valid_cols >= HMX_TILE &&
      (dest_stride % HMX_TILE) == 0u) {
    for (uint32_t row_pair = 0; row_pair < HMX_TILE / 2u; ++row_pair) {
      hmx_vector_halves_t rows;
      rows.vector = Q6_Vh_vdeal_Vh(src[row_pair]);
      _Float16 *row0 = dest + (2u * row_pair) * dest_stride;
      _Float16 *row1 = dest + (2u * row_pair + 1u) * dest_stride;
      hmx_vector_halves_t dest_rows;
      dest_rows.halves[0] = *(const hmx_half_vector_t *)row0;
      dest_rows.halves[1] = *(const hmx_half_vector_t *)row1;
      rows.vector = Q6_Vhf_vadd_VhfVhf(dest_rows.vector, rows.vector);
      *(hmx_half_vector_t *)row0 = rows.halves[0];
      *(hmx_half_vector_t *)row1 = rows.halves[1];
    }
    return;
  }
  // Boundary/ragged tile: de-interleave into a scratch, then copy only the
  // valid region.
  _Float16 scratch[HMX_TILE * HMX_TILE] __attribute__((aligned(128)));
  for (uint32_t row_pair = 0; row_pair < HMX_TILE / 2u; ++row_pair) {
    hmx_vector_halves_t rows;
    rows.vector = Q6_Vh_vdeal_Vh(src[row_pair]);
    *(hmx_half_vector_t *)(scratch + (2u * row_pair) * HMX_TILE) =
        rows.halves[0];
    *(hmx_half_vector_t *)(scratch + (2u * row_pair + 1u) * HMX_TILE) =
        rows.halves[1];
  }
  for (uint32_t r = 0; r < valid_rows; ++r) {
    _Float16 *drow = dest + r * dest_stride;
    const _Float16 *srow = scratch + r * HMX_TILE;
    for (uint32_t c = 0; c < valid_cols; ++c) {
      drow[c] += srow[c];
    }
  }
}

// Unpack a `row_tiles` x `col_tiles` grid of HMX accumulator-read tiles into a
// (possibly ragged) row-major f16 destination of `actual_rows` x `actual_cols`,
// without widening to f32. Mirrors iree_hexagon_hmx_unpack_acc_f16_to_f32 but
// for an f16 output matmul; only the valid region of each boundary tile is
// added to the destination.
void iree_hexagon_hmx_unpack_acc_f16_to_f16(
    uint32_t dest, uint32_t source, uint32_t dest_stride, uint32_t actual_rows,
    uint32_t actual_cols, uint32_t row_tiles, uint32_t col_tiles) {
  _Float16 *dest_base = (_Float16 *)dest;
  const uint8_t *src_tiles = (const uint8_t *)source;
  for (uint32_t row_tile = 0; row_tile < row_tiles; ++row_tile) {
    uint32_t valid_rows = iree_hexagon_hmx_tile_extent(actual_rows, row_tile);
    if (valid_rows == 0u) {
      continue;
    }
    for (uint32_t col_tile = 0; col_tile < col_tiles; ++col_tile) {
      uint32_t valid_cols = iree_hexagon_hmx_tile_extent(actual_cols, col_tile);
      if (valid_cols == 0u) {
        continue;
      }
      const HVX_Vector *src =
          (const HVX_Vector *)(src_tiles + (row_tile * col_tiles + col_tile) *
                                               HMX_TILE_BYTES);
      _Float16 *dtile =
          dest_base + (row_tile * HMX_TILE) * dest_stride + col_tile * HMX_TILE;
      iree_hexagon_hmx_unpack_acc_tile_f16(dtile, src, dest_stride, valid_rows,
                                           valid_cols);
    }
  }
}

void iree_hexagon_hmx_mma_f16(uint32_t activation, uint32_t weight) {
  // The activation load consumes the packed activation tile. Rt=0x7FFF selects
  // the full 32-spatial by 32-input-channel region used by one HMX FP16 tile.
  Q6_activation_hf_mxmem_RR(activation, HMX_ACT_RT);

  // The weight load must immediately follow the activation load to issue one
  // multiply-accumulate. Rt=1920 is the full-tile dW encoding: the byte
  // distance to the last 128 B vector in the packed 32x32 FP16 weight tile.
  Q6_weight_hf_mxmem_RR(weight, HMX_WEI_RT);
}

void iree_hexagon_hmx_acc_read_f16(uint32_t output) {
  // Rs=0 selects bias set 0 with no feedback and requests the standard
  // clear+swap behavior after converting the HMX accumulator to FP16 memory.
  Q6_mxmem_AR_after_hf((void *)output, HMX_CVT_RS);
}

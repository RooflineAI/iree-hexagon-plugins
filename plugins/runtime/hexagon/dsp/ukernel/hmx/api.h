// Copyright 2026 RooflineAI GmbH
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef ROOF_HEXAGON_RUNTIME_DSP_UKERNEL_HMX_API_H_
#define ROOF_HEXAGON_RUNTIME_DSP_UKERNEL_HMX_API_H_

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// All pointer arguments use the Hexagon DSP's 32-bit address ABI. Buffers must
// reside in VTCM. Row-major f16 buffers are 64-byte aligned, row-major f32
// buffers are 128-byte aligned, and packed tiles, accumulator read-out tiles,
// and config are 2048-byte aligned.

// Initializes the 256-byte accumulator-read bias block. No intentionally large
// stack allocation is used; compiler-generated spills are not included.
void iree_hexagon_hmx_acc_setup_read_f16(uint32_t config);

// Clears the hardware accumulator. No intentionally large stack allocation is
// used; compiler-generated spills are not included.
void iree_hexagon_hmx_acc_clear_f16(void);

// Packs row-major f16 data into 32x32 HMX tiles. Full tiles use only small
// locals; a ragged tile uses approximately 2 KiB of aligned stack scratch.
void iree_hexagon_hmx_pack_f16(uint32_t dest, uint32_t source,
                               uint32_t source_stride, uint32_t actual_rows,
                               uint32_t actual_cols, uint32_t row_tiles,
                               uint32_t col_tiles);

// Packs a transposed row-major f16 matrix into HMX tiles. A ragged tile uses
// approximately 2 KiB of aligned stack scratch. The full-tile transpose may
// require roughly 4 KiB for its vector arrays before compiler spills.
void iree_hexagon_hmx_pack_transposed_f16(uint32_t dest, uint32_t source,
                                          uint32_t source_stride,
                                          uint32_t actual_interleave,
                                          uint32_t actual_other,
                                          uint32_t interleave_tiles,
                                          uint32_t other_tiles);

// Converts HMX f16 read-out tiles to f32 and adds them to the destination. A
// boundary tile uses approximately 4 KiB of aligned stack scratch.
void iree_hexagon_hmx_unpack_acc_f16_to_f32(
    uint32_t dest, uint32_t source, uint32_t dest_stride, uint32_t actual_rows,
    uint32_t actual_cols, uint32_t row_tiles, uint32_t col_tiles);

// De-interleaves HMX f16 read-out tiles and adds them to an f16 destination. A
// boundary tile uses approximately 2 KiB of aligned stack scratch.
void iree_hexagon_hmx_unpack_acc_f16_to_f16(
    uint32_t dest, uint32_t source, uint32_t dest_stride, uint32_t actual_rows,
    uint32_t actual_cols, uint32_t row_tiles, uint32_t col_tiles);

// Issues one 32x32x32 f16 HMX multiply-accumulate. No intentionally large stack
// allocation is used; compiler-generated spills are not included.
void iree_hexagon_hmx_mma_f16(uint32_t activation, uint32_t weight);

// Reads the hardware accumulator into one packed f16 tile. No intentionally
// large stack allocation is used; compiler-generated spills are not included.
void iree_hexagon_hmx_acc_read_f16(uint32_t output);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // ROOF_HEXAGON_RUNTIME_DSP_UKERNEL_HMX_API_H_

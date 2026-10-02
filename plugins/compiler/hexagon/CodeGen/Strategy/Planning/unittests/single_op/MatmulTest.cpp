// Copyright 2026 RooflineAI GmbH
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "TestUtils.h"

namespace mlir::iree_compiler::hexagon::codegen::planning {
namespace {

/// linalg.matmul, f32.
REGISTER_ESTIMATION_TEST_SUITE(
    Matmul,
    R"mlir(
      func.func @matmul(%lhs: tensor<256x256xf32>, %rhs: tensor<256x256xf32>,
                         %out: tensor<256x256xf32>) -> tensor<256x256xf32> {
        %0 = linalg.matmul ins(%lhs, %rhs : tensor<256x256xf32>, tensor<256x256xf32>)
                            outs(%out : tensor<256x256xf32>) -> tensor<256x256xf32>
        return %0 : tensor<256x256xf32>
      }
    )mlir",
    DispatchConfig{"IreeChosenTiling",
                   /*tileSizes=*/{32, 32, 8}, // [M, N, K]
                   /*expectedVector=*/34,     // 32 accumulators + 1 rhs + 1 lhs
                   /*expectedFailure=*/"",
                   "measured HvxVR=35; The difference is due to an additional "
                   "register needed for unaligned loads"},
    DispatchConfig{
        "16x16x16",
        /*tileSizes=*/{16, 16, 16}, // [M, N, K]
        /*expectedVector=*/18,      // 16 accumulators + 1 lhs + 1rhs
        /*expectedFailure=*/"",
        "measured HvxVR=37; measurement: the machine scheduler fills the "
        "remaining registers with hoisted splats to hide latencs"},

    DispatchConfig{
        "HandPickedMisalignedTileEpilogueDominates",
        /*tileSizes=*/{5, 6, 7}, // [M, N, K]
                                 // 5 accumulator + 1 lhs + 1 rhs
        /*expectedVector=*/7,
        /*expectedFailure=*/"",
        "measured HvxVR=20"
        "The Real register usage is dominated by vectors needed for rotate "
        "and mask select operations, since the tiling does not divide the "
        "vector length"},
    DispatchConfig{
        "ExactRegisterMultiple",
        /*tileSizes=*/{8, 8, 8}, // [M, N, K]
                                 // 8 accumulator + 1 lhs + 1 rhs
        /*expectedVector=*/10,
        /*expectedFailure=*/"",
        "Measured HvxVR=0, IntRegs=166. Operation is scalarized, since "
        "the rows only fill a quater of a reg"},
    DispatchConfig{
        "ReductionHeavyTileDominatesInputs",
        /*tileSizes=*/{4, 4, 64}, // [M, N, K]
                                  // 4 accumulator + 2 lhs + 1 rhs
        /*expectedVector=*/7,
        /*expectedFailure=*/"",
        "Measured HvxVR=0, IntRegs=471 - Scalarized, as the test above"});

/// linalg.matmul, f16 into f32.
REGISTER_ESTIMATION_TEST_SUITE(
    Matmul16To32,
    R"mlir(
      func.func @matmul(%lhs: tensor<256x256xf16>, %rhs: tensor<256x256xf16>,
                         %out: tensor<256x256xf32>) -> tensor<256x256xf32> {
        %0 = linalg.matmul ins(%lhs, %rhs : tensor<256x256xf16>, tensor<256x256xf16>)
                            outs(%out : tensor<256x256xf32>) -> tensor<256x256xf32>
        return %0 : tensor<256x256xf32>
      }
    )mlir",
    DispatchConfig{
        "IreeChosenf32Tiling",
        /*tileSizes=*/{32, 32, 8}, // [M, N, K]
        /*expectedVector=*/34,     // same reasoning as pure f32 case
        /*expectedFailure=*/"",
    },
    DispatchConfig{"ExactRegisterMultiple",
                   /*tileSizes=*/{16, 16, 16}, // [M, N, K]
                   /*expectedVector=*/18, // same reasoning as pure f32 case
                   /*expectedFailure=*/"", "Measured HvxVR=37, TODO 2"},
    DispatchConfig{"ReductionHeavyTileDominatesInputs",
                   /*tileSizes=*/{4, 4, 64}, // [M, N, K]
                   /*expectedVector=*/6,     // same reasoning as pure f32 case
                   /*expectedFailure=*/"",
                   "Measured HvxVR=10, TODO 3 evaluate"});

/// linalg.matmul with a transposed lhs
REGISTER_ESTIMATION_TEST_SUITE(
    MatmulTransposedLhs, R"mlir(
    #lhs_t = affine_map<(m, n, k) -> (k, m)>
    #rhs_n = affine_map<(m, n, k) -> (k, n)>
    #out_n = affine_map<(m, n, k) -> (m, n)>
    func.func @matmul_transpose_a(%lhs: tensor<256x256xf32>,
                                  %rhs: tensor<256x256xf32>,
                                  %out: tensor<256x256xf32>)
                                  -> tensor<256x256xf32> {
      %0 = linalg.matmul indexing_maps = [#lhs_t, #rhs_n, #out_n]
                          ins(%lhs, %rhs : tensor<256x256xf32>, tensor<256x256xf32>)
                          outs(%out : tensor<256x256xf32>) -> tensor<256x256xf32>
      return %0 : tensor<256x256xf32>
    }
  )mlir",
    DispatchConfig{"TransposedLhsExactRegisterMultiple",
                   /*tileSizes=*/{8, 8, 8}, // [M, N, K]
                                            // 8 accumulator + 1 lhs + 1 rhs
                   /*expectedVector=*/10,
                   /*expectedFailure=*/"",
                   "Measured HvxVR=0, same as un-transposed, no vectorization"},
    DispatchConfig{"TransposedLhsMisalignedTile",
                   /*tileSizes=*/{5, 6, 7}, // [M, N, K]
                                            // 5 accumulator + 1 lhs + 1 rhs
                   /*expectedVector=*/7,
                   /*expectedFailure=*/"", "Compiler error??"},
    DispatchConfig{"TransposedLHs32",
                   /*tileSizes=*/{32, 32, 8}, // [M, N, K]
                                              // same as un-transposed
                   /*expectedVector=*/34,
                   /*expectedFailure=*/"",
                   "Measured: 35, same as non-transposed case"});

/// linalg.matmul with a transposed rhs
REGISTER_ESTIMATION_TEST_SUITE(
    MatmulTransposedRhs, R"mlir(
    #lhs_n = affine_map<(m, n, k) -> (m, k)>
    #rhs_t = affine_map<(m, n, k) -> (n, k)>
    #out_n = affine_map<(m, n, k) -> (m, n)>
    func.func @matmul_transpose_b(%lhs: tensor<256x256xf32>,
                                  %rhs: tensor<256x256xf32>,
                                  %out: tensor<256x256xf32>)
                                  -> tensor<256x256xf32> {
      %0 = linalg.matmul indexing_maps = [#lhs_n, #rhs_t, #out_n]
                          ins(%lhs, %rhs : tensor<256x256xf32>, tensor<256x256xf32>)
                          outs(%out : tensor<256x256xf32>) -> tensor<256x256xf32>
      return %0 : tensor<256x256xf32>
    }
  )mlir",
    DispatchConfig{"TransposedRhsReductionHeavyTile",
                   /*tileSizes=*/{4, 4, 64}, // [M, N, K]
                                             // 4 accumulator + 2 lhs + 8 rhs
                   // + 8 for the rhs tile's transposed copy, written while the
                   // gathered source is read
                   /*expectedVector=*/22,
                   /*expectedFailure=*/"",
                   "Measured: 9."
                   "resident rhs (8) + 1 accumulator"
                   "Estimation is off: lhs is loaded scalar and broadcast:"
                   "it does not have a vector reg that stays life. The "
                   "transposed copy's rows are 4 f32 wide, which scalarizes, "
                   "so no HVX register ever holds it"},
    DispatchConfig{
        "TransposedRhs16",
        /*tileSizes=*/{16, 16, 8}, // [M, N, K]
                                   // 16 accumulator + 1 lhs + 4 rhs
        // + 4 for the rhs tile's transposed copy, written while the gathered
        // source is read
        /*expectedVector=*/25,
        /*expectedFailure=*/"",
        "measured HvxVR=37. Additional shuffle overhead when re-packing "
        "the 16 half-filled registers in the end"});

/// linalg.matmul with both operands transposed
REGISTER_ESTIMATION_TEST_SUITE(
    MatmulTransposedBoth, R"mlir(
    #lhs_t = affine_map<(m, n, k) -> (k, m)>
    #rhs_t = affine_map<(m, n, k) -> (n, k)>
    #out_n = affine_map<(m, n, k) -> (m, n)>
    func.func @matmul_transpose_ab(%lhs: tensor<256x256xf32>,
                                   %rhs: tensor<256x256xf32>,
                                   %out: tensor<256x256xf32>)
                                   -> tensor<256x256xf32> {
      %0 = linalg.matmul indexing_maps = [#lhs_t, #rhs_t, #out_n]
                          ins(%lhs, %rhs : tensor<256x256xf32>, tensor<256x256xf32>)
                          outs(%out : tensor<256x256xf32>) -> tensor<256x256xf32>
      return %0 : tensor<256x256xf32>
    }
  )mlir",
    DispatchConfig{
        "TransposedBothReductionHeavyTile",
        /*tileSizes=*/{4, 4, 64}, // [M, N, K]
                                  // 4 accumulator + 1 lhs + 8 rhs
        // + 8 for the rhs tile's transposed copy, written while the gathered
        // source is read
        /*expectedVector=*/21,
        /*expectedFailure=*/"",
        "measured HvxVR=9, same as TransposedRhsReductionHeavyTile: the "
        "4-wide transposed copy scalarizes"},
    DispatchConfig{"TransposedBothExactRegisterMultiple",
                   /*tileSizes=*/{16, 16, 8}, // [M, N, K]
                                              // 16 accumulator + 1 lhs + 4 rhs
                   // + 4 for the rhs tile's transposed copy, written while the
                   // gathered source is read
                   /*expectedVector=*/25,
                   /*expectedFailure=*/"",
                   "Measured:37>"
                   "same as above, when only rhs is transposed."});

/// linalg.matmul with both operands transposed
/// the result is f32 instead of the inputs f16

REGISTER_ESTIMATION_TEST_SUITE(
    MatmulTransposedBothMixedPrecision,
    R"mlir(
    #lhs_t = affine_map<(m, n, k) -> (k, m)>
    #rhs_t = affine_map<(m, n, k) -> (n, k)>
    #out_n = affine_map<(m, n, k) -> (m, n)>
    func.func @matmul_transpose_ab_mixed(%lhs: tensor<256x256xf16>,
                                         %rhs: tensor<256x256xf16>,
                                         %out: tensor<256x256xf32>)
                                         -> tensor<256x256xf32> {
      %0 = linalg.matmul indexing_maps = [#lhs_t, #rhs_t, #out_n]
                          ins(%lhs, %rhs : tensor<256x256xf16>, tensor<256x256xf16>)
                          outs(%out : tensor<256x256xf32>) -> tensor<256x256xf32>
      return %0 : tensor<256x256xf32>
    }
  )mlir",
    DispatchConfig{
        "TransposedBothMixedPrecisionPartialK",
        /*tileSizes=*/{4, 4, 32}, // [M, N, K], loop extents 256, 256, 256
        // + 2 for the f16 rhs tile's transposed copy, written while the
        // gathered source is read
        /*expectedVector=*/9,
        /*expectedFailure=*/"",
        "Measured: 10 estimation is off due to datatype change, where in "
        "this "
        "case the heuristic overcompensates."});

} // namespace
} // namespace mlir::iree_compiler::hexagon::codegen::planning

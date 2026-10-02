// Copyright 2026 RooflineAI GmbH
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "TestUtils.h"

// Anchored to hexagon-tiling-tests/examples/permute: a 1x56x56x64 NHWC
// tensor permuted to 1x64x56x56 NCHW. IREE collapses the leading unit/
// spatial dims into a single 2D dispatch (examples/permute/config.mlir): a
// linalg.generic over (d0=3136, d1=64) whose output map is the permuted
// `(d1, d0)` instead of the identity - the first anchored op where an
// operand's indexing map is not the identity or a partial-dim dependency,
// but a full permutation.

namespace mlir::iree_compiler::hexagon::codegen::planning {
namespace {

/// 2 loops (d0, d1), both parallel. The input's map is the identity
/// `(d0, d1)`; the output's map is the permuted `(d1, d0)` - it still
/// depends on both dims (`AffineExpr::isFunctionOfDim` doesn't care about
/// order), so per the documented formula its register count is unaffected
/// by the permutation.
REGISTER_ESTIMATION_TEST_SUITE(
    Permute,
    R"mlir(
      func.func @permute(%in: tensor<3136x64xf32>, %out: tensor<64x3136xf32>) -> tensor<64x3136xf32> {
        %0 = linalg.generic {
          indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                            affine_map<(d0, d1) -> (d1, d0)>],
          iterator_types = ["parallel", "parallel"]
        } ins(%in : tensor<3136x64xf32>) outs(%out : tensor<64x3136xf32>) {
        ^bb0(%in0: f32, %init: f32):
          linalg.yield %in0 : f32
        } -> tensor<64x3136xf32>
        return %0 : tensor<64x3136xf32>
      }
    )mlir",
    DispatchConfig{
        "IreeChosenTiling",
        /*tileSizes=*/{1, 32}, // [d0, d1]
        /*expectedVector=*/-1,
        /*expectedFailure=*/"",
        "measured HvxVR=1. This uses 105 IntRegs (Measured)"
        "the cost are hidden in terms of HVX registers; the heuristic "
        "is designed to over-estimate this cost"},
    DispatchConfig{"PermutedOperandRegisterCountUnaffectedByOrder",
                   /*tileSizes=*/{8, 64}, // [d0, d1]
                   /*expectedVector=*/80,
                   /*expectedFailure=*/"",
                   "Measured: 17 hvx registers, 523 int registers."
                   " Again, the cost moves to the Int Registers"},
    DispatchConfig{
        "SameShapeAsTranspose",
        /*tileSizes=*/{16, 32}, // [d0, d1]
        /*expectedVector=*/-1,
        /*expectedFailure=*/"",
        "Measured: 89 hvx registers, 22 int registers in this case, the "
        "re-ordering actually uses hvx registers instead of scalar ones"});

} // namespace
} // namespace mlir::iree_compiler::hexagon::codegen::planning

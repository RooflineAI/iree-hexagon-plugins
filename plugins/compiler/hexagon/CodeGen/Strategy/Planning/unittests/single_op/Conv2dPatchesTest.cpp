// Copyright 2026 RooflineAI GmbH
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "TestUtils.h"

namespace mlir::iree_compiler::hexagon::codegen::planning {
namespace {

/// 5 loops (d0=OH, d1=OW, d2=KH, d3=KW, d4=C), all parallel. The single
/// input's indexing map is `(d0+d2, d1+d3, d4)`: a sliding-window gather
/// where the output row dim OH and the kernel row dim KH both add into the
/// same input axis (and likewise OW/KW into the other) - so per
/// `AffineExpr::isFunctionOfDim`, the input depends on *all 5* loop dims,
/// not just the ones its own shape's rank would suggest. The output's
/// identity map also depends on all 5. The op body is a pure `linalg.yield`
/// (no arithmetic) - it only rearranges data.
REGISTER_ESTIMATION_TEST_SUITE(
    Conv2dPatches,
    R"mlir(
      func.func @conv2d_patches(%in: tensor<34x34x32xf32>,
                                 %out: tensor<32x32x3x3x32xf32>) -> tensor<32x32x3x3x32xf32> {
        %0 = linalg.generic {
          indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0 + d2, d1 + d3, d4)>,
                            affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>],
          iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]
        } ins(%in : tensor<34x34x32xf32>) outs(%out : tensor<32x32x3x3x32xf32>) {
        ^bb0(%in0: f32, %init: f32):
          linalg.yield %in0 : f32
        } -> tensor<32x32x3x3x32xf32>
        return %0 : tensor<32x32x3x3x32xf32>
      }
    )mlir",
    DispatchConfig{
        "IreeChosenTiling",
        /*tileSizes=*/{1, 1, 1, 1, 32}, // [OH, OW, KH, KW, C]
        /*expectedVector=*/-1,
        /*expectedFailure=*/"",
        "measured HvxVR=0 for "
        "`linalg.yield` with no arithmetic, no need to ever load it into a "
        "register."},
    DispatchConfig{"OverlappingWindowDoubleCountsSlidingDim",
                   /*tileSizes=*/{4, 1, 3, 1, 32}, // [OH, OW, KH, KW, C]
                   /*expectedVector=*/-1,
                   /*expectedFailure=*/"", "measured: 11"});

} // namespace
} // namespace mlir::iree_compiler::hexagon::codegen::planning

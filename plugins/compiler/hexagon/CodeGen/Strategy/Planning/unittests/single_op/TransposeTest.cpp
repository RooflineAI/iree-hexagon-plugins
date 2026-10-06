// Copyright 2026 RooflineAI GmbH
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "TestUtils.h"

namespace mlir::iree_compiler::hexagon::codegen::planning {
namespace {

/// 2 loops (d0, d1), both parallel. Structurally identical to
/// PermuteTest's op: identity input map, permuted `(d1, d0)` output map.
REGISTER_ESTIMATION_TEST_SUITE(
    Transpose,
    R"mlir(
      func.func @transpose(%in: tensor<1024x1024xf32>, %out: tensor<1024x1024xf32>) -> tensor<1024x1024xf32> {
        %0 = linalg.generic {
          indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                            affine_map<(d0, d1) -> (d1, d0)>],
          iterator_types = ["parallel", "parallel"]
        } ins(%in : tensor<1024x1024xf32>) outs(%out : tensor<1024x1024xf32>) {
        ^bb0(%in0: f32, %init: f32):
          linalg.yield %in0 : f32
        } -> tensor<1024x1024xf32>
        return %0 : tensor<1024x1024xf32>
      }
    )mlir",
    DispatchConfig{
        "IreeChosenTiling",
        /*tileSizes=*/{1, 32}, // [d0, d1]
        /*expectedVector=*/-1,
        /*expectedFailure=*/"",
        "measured HvxVR=1, with the cost in the scalar file instead, "
        "as in PermuteTest's IreeChosenTiling"},
    DispatchConfig{"ScaledUpTileMatchesPermuteAnalogue",
                   /*tileSizes=*/{16, 32}, // [d0, d1]
                   /*expectedVector=*/-1,
                   /*expectedFailure=*/"", "measured: 89. shuffling overhead"});

} // namespace
} // namespace mlir::iree_compiler::hexagon::codegen::planning

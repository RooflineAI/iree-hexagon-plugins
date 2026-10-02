// Copyright 2026 RooflineAI GmbH
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "TestUtils.h"

// `math.sin` / `math.cos` over the same contiguous 150528-element 1D iteration
// space as the baseline.

namespace mlir::iree_compiler::hexagon::codegen::planning {
namespace {

REGISTER_ESTIMATION_TEST_SUITE(
    Sin,
    R"mlir(
      func.func @sin(%x: tensor<150528xf32>, %out: tensor<150528xf32>) -> tensor<150528xf32> {
        %0 = linalg.generic {
          indexing_maps = [affine_map<(d0) -> (d0)>,
                            affine_map<(d0) -> (d0)>],
          iterator_types = ["parallel"]
        } ins(%x : tensor<150528xf32>) outs(%out : tensor<150528xf32>) {
        ^bb0(%in: f32, %init: f32):
          %s = math.sin %in : f32
          linalg.yield %s : f32
        } -> tensor<150528xf32>
        return %0 : tensor<150528xf32>
      }
    )mlir",
    DispatchConfig{"OneTile", /*tileSizes=*/{32}, /*expectedVector=*/26,
                   /*expectedFailure=*/"", "measured HvxVR=26"},
    DispatchConfig{"TwoTiles", /*tileSizes=*/{64}, /*expectedVector=*/33,
                   /*expectedFailure=*/"", "measured HvxVR=33"},
    DispatchConfig{"FourTiles", /*tileSizes=*/{128}, /*expectedVector=*/47,
                   /*expectedFailure=*/"", "measured HvxVR=47"});

REGISTER_ESTIMATION_TEST_SUITE(
    Cos,
    R"mlir(
      func.func @cos(%x: tensor<150528xf32>, %out: tensor<150528xf32>) -> tensor<150528xf32> {
        %0 = linalg.generic {
          indexing_maps = [affine_map<(d0) -> (d0)>,
                            affine_map<(d0) -> (d0)>],
          iterator_types = ["parallel"]
        } ins(%x : tensor<150528xf32>) outs(%out : tensor<150528xf32>) {
        ^bb0(%in: f32, %init: f32):
          %c = math.cos %in : f32
          linalg.yield %c : f32
        } -> tensor<150528xf32>
        return %0 : tensor<150528xf32>
      }
    )mlir",
    DispatchConfig{"OneTile", /*tileSizes=*/{32}, /*expectedVector=*/28,
                   /*expectedFailure=*/"", "measured HvxVR=28"},
    DispatchConfig{"TwoTiles", /*tileSizes=*/{64}, /*expectedVector=*/35,
                   /*expectedFailure=*/"", "measured HvxVR=36"},
    DispatchConfig{"FourTiles", /*tileSizes=*/{128}, /*expectedVector=*/49,
                   /*expectedFailure=*/"", "measured HvxVR=50"});

} // namespace
} // namespace mlir::iree_compiler::hexagon::codegen::planning

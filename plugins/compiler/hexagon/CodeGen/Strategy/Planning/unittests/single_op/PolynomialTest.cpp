// Copyright 2026 RooflineAI GmbH
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "TestUtils.h"

// Anchored to hexagon-tiling-tests/examples/polynomial: a degree-4 Horner
// evaluation over the same contiguous 150528-element 1D iteration space as
// the `add` baseline, so the only difference from that example is the body.
//
// This is the first op in the corpus whose body holds values that are neither
// an operand tile nor a per-element temporary: five coefficient constants,
// uniform across the iteration space. Each is splatted once into a vector
// register and hoisted out of the loop, so its cost stays at one register
// however wide the tile is. That is what the two configs below pin - see
// examples/polynomial/NOTES.md for the measured numbers.

namespace mlir::iree_compiler::hexagon::codegen::planning {
namespace {

/// 1 loop (d0), parallel, identity map on both operands. The body evaluates
/// ((((c4*x + c3)*x + c2)*x + c1)*x + c0) by Horner's rule: at any point only
/// the input slice and one intermediate are live, while all five coefficients
/// stay resident.
REGISTER_ESTIMATION_TEST_SUITE(
    Polynomial,
    R"mlir(
      func.func @polynomial(%x: tensor<150528xf32>, %out: tensor<150528xf32>) -> tensor<150528xf32> {
        %0 = linalg.generic {
          indexing_maps = [affine_map<(d0) -> (d0)>,
                            affine_map<(d0) -> (d0)>],
          iterator_types = ["parallel"]
        } ins(%x : tensor<150528xf32>) outs(%out : tensor<150528xf32>) {
        ^bb0(%in: f32, %init: f32):
          %c4 = arith.constant 4.000000e+00 : f32
          %c3 = arith.constant 3.000000e+00 : f32
          %c2 = arith.constant 2.000000e+00 : f32
          %c1 = arith.constant 1.000000e+00 : f32
          %c0 = arith.constant 5.000000e-01 : f32
          %m1 = arith.mulf %c4, %in : f32
          %a1 = arith.addf %m1, %c3 : f32
          %m2 = arith.mulf %a1, %in : f32
          %a2 = arith.addf %m2, %c2 : f32
          %m3 = arith.mulf %a2, %in : f32
          %a3 = arith.addf %m3, %c1 : f32
          %m4 = arith.mulf %a3, %in : f32
          %a4 = arith.addf %m4, %c0 : f32
          linalg.yield %a4 : f32
        } -> tensor<150528xf32>
        return %0 : tensor<150528xf32>
      }
    )mlir",
    DispatchConfig{"IreeChosenTiling",
                   /*tileSizes=*/{32},
                   /*expectedVector=*/7,
                   /*expectedFailure=*/"",
                   "measured HvxVR=7, 5 hoisted coefficient splats"
                   " plus 2 varying registers (input + one intermediate).",
                   // 7 full registers * 128 B.
                   /*expectedUsefulBytes=*/896},
    DispatchConfig{
        "CoefficientsDoNotScaleWithTile",
        /*tileSizes=*/{128},
        /*expectedVector=*/13,
        /*expectedFailure=*/"",
        "measured HvxVR=13. Quadrupling tile only affects the varying "
        "values, the coefficients contribute a flat 5 at both sizes",
        // 13 full registers * 128 B.
        /*expectedUsefulBytes=*/1664});

} // namespace
} // namespace mlir::iree_compiler::hexagon::codegen::planning

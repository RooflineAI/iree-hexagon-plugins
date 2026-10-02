// Copyright 2026 RooflineAI GmbH
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "TestUtils.h"

namespace mlir::iree_compiler::hexagon::codegen::planning {
namespace {

/// i8->i32 sign-extend
/// 1 loop (d0), parallel, identity map on both operands. Input is i8,
/// output is i32 - each operand's register count must use its own element
/// bit width, not a shared one.
REGISTER_ESTIMATION_TEST_SUITE(
    Convert,
    R"mlir(
      func.func @convert(%in: tensor<1605632xi8>, %out: tensor<1605632xi32>) -> tensor<1605632xi32> {
        %0 = linalg.generic {
          indexing_maps = [affine_map<(d0) -> (d0)>,
                            affine_map<(d0) -> (d0)>],
          iterator_types = ["parallel"]
        } ins(%in : tensor<1605632xi8>) outs(%out : tensor<1605632xi32>) {
        ^bb0(%in0: i8, %init: i32):
          %ext = arith.extsi %in0 : i8 to i32
          linalg.yield %ext : i32
        } -> tensor<1605632xi32>
        return %0 : tensor<1605632xi32>
      }
    )mlir",
    DispatchConfig{"IreeChosenTiling",
                   /*tileSizes=*/{32},
                   /*expectedVector=*/1,
                   /*expectedFailure=*/"",
                   "measured HvxVR=6. real usage is "
                   "higher because the i8->i32 sign-extend "
                   "uses additional registers for expanding and unpacking"},
    DispatchConfig{"SameTileDifferentBitwidthsScaleIndependently",
                   /*tileSizes=*/{128},
                   /*expectedVector=*/4,
                   /*expectedFailure=*/"", "measured: 4"}
    // TODO out of curiosity: investigate, why 128 can go with exactly 4 regs
);

} // namespace
} // namespace mlir::iree_compiler::hexagon::codegen::planning

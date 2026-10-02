// Copyright 2026 RooflineAI GmbH
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "TestUtils.h"

namespace mlir::iree_compiler::hexagon::codegen::planning {
namespace {

/// Elementwise Addition
/// 1 loop (d0), parallel. 2 f32 inputs + 1 f32 output, all with the identity
/// indexing map, so every operand's tile element count is simply
/// tileSize[d0].
REGISTER_ESTIMATION_TEST_SUITE(
    ElementwiseAdd,
    R"mlir(
      func.func @add(%a: tensor<150528xf32>, %b: tensor<150528xf32>,
                      %out: tensor<150528xf32>) -> tensor<150528xf32> {
        %0 = linalg.generic {
          indexing_maps = [affine_map<(d0) -> (d0)>,
                            affine_map<(d0) -> (d0)>,
                            affine_map<(d0) -> (d0)>],
          iterator_types = ["parallel"]
        } ins(%a, %b : tensor<150528xf32>, tensor<150528xf32>)
          outs(%out : tensor<150528xf32>) {
        ^bb0(%in0: f32, %in1: f32, %init: f32):
          %sum = arith.addf %in0, %in1 : f32
          linalg.yield %sum : f32
        } -> tensor<150528xf32>
        return %0 : tensor<150528xf32>
      }
    )mlir",
    DispatchConfig{"IreeChosenTiling",
                   /*tileSizes=*/{32},
                   /*expectedVector=*/2,
                   /*expectedFailure=*/"", "measured HvxVR=2",
                   // Two full tiles of 32*4 B, no padding.
                   /*expectedUsefulBytes=*/256},
    DispatchConfig{"RoundsUpJustOverOneRegister",
                   /*tileSizes=*/{33},
                   /*expectedVector=*/4,
                   /*expectedFailure=*/"",
                   "33 elements dont fit into one register anymore",
                   // 2 * 33*4 B = 264 of the 512 B the four registers hold.
                   /*expectedUsefulBytes=*/264},
    DispatchConfig{"TwoRegisters",
                   /*tileSizes=*/{64},
                   /*expectedVector=*/4,
                   /*expectedFailure=*/"", "two full registers",
                   /*expectedUsefulBytes=*/512},
    DispatchConfig{"UntiledTakesTheFullExtent",
                   /*tileSizes=*/{0},
                   // 0 resolves to the loop's extent: ceil(150528*32/1024) =
                   // 4704 per node, two input tiles live -> 9408.
                   /*expectedVector=*/9408,
                   /*expectedFailure=*/"", "0 means untiled",
                   // 150528 is a multiple of 32: 2 * 150528*4 B, all full.
                   /*expectedUsefulBytes=*/1204224});

/// The same op with its element type changed to i8 and nothing else, so the
/// only difference is how many elements a register holds: 1024/8 = 128.
REGISTER_ESTIMATION_TEST_SUITE(
    ElementwiseAddI8,
    R"mlir(
      func.func @add(%a: tensor<150528xi8>, %b: tensor<150528xi8>,
                      %out: tensor<150528xi8>) -> tensor<150528xi8> {
        %0 = linalg.generic {
          indexing_maps = [affine_map<(d0) -> (d0)>,
                            affine_map<(d0) -> (d0)>,
                            affine_map<(d0) -> (d0)>],
          iterator_types = ["parallel"]
        } ins(%a, %b : tensor<150528xi8>, tensor<150528xi8>)
          outs(%out : tensor<150528xi8>) {
        ^bb0(%in0: i8, %in1: i8, %init: i8):
          %sum = arith.addi %in0, %in1 : i8
          linalg.yield %sum : i8
        } -> tensor<150528xi8>
        return %0 : tensor<150528xi8>
      }
    )mlir",
    DispatchConfig{"PadsThePartialRegister",
                   /*tileSizes=*/{100},
                   // ceil(100*8/1024) = 1: one register with 28 lanes of
                   // padding. Two input tiles live -> 2.
                   /*expectedVector=*/2,
                   /*expectedFailure=*/"",
                   "100 i8 pad to a full 128-lane register",
                   // 2 * 100 B of the two 128 B registers.
                   /*expectedUsefulBytes=*/200},
    DispatchConfig{"ExactlyEightRegisters",
                   /*tileSizes=*/{1024},
                   // 1024*8/1024 = 8 per node, two input tiles live -> 16.
                   /*expectedVector=*/16,
                   /*expectedFailure=*/"",
                   "no padding at a whole multiple of 128 lanes",
                   /*expectedUsefulBytes=*/2048},
    DispatchConfig{"OneElementOverPadsAWholeRegister",
                   /*tileSizes=*/{1025},
                   // ceil(1025*8/1024) = 9 per node, two live -> 18.
                   /*expectedVector=*/18,
                   /*expectedFailure=*/"",
                   "a single extra lane costs a whole register",
                   // 2 * 1025 B of the 18 * 128 = 2304 B held.
                   /*expectedUsefulBytes=*/2050});

} // namespace
} // namespace mlir::iree_compiler::hexagon::codegen::planning

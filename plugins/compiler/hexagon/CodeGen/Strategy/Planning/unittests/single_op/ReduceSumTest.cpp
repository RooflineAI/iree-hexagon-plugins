// Copyright 2026 RooflineAI GmbH
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "TestUtils.h"

namespace mlir::iree_compiler::hexagon::codegen::planning {
namespace {

/// reduction sum operation
/// 2 loops (d0 parallel, d1 reduction). The input depends on both dims; the
/// output depends on d0 only.
REGISTER_ESTIMATION_TEST_SUITE(
    ReduceSum,
    R"mlir(
      func.func @reduce_sum(%in: tensor<1024x1024xf32>, %out: tensor<1024xf32>) -> tensor<1024xf32> {
        %0 = linalg.generic {
          indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                            affine_map<(d0, d1) -> (d0)>],
          iterator_types = ["parallel", "reduction"]
        } ins(%in : tensor<1024x1024xf32>) outs(%out : tensor<1024xf32>) {
        ^bb0(%in0: f32, %init: f32):
          %sum = arith.addf %in0, %init : f32
          linalg.yield %sum : f32
        } -> tensor<1024xf32>
        return %0 : tensor<1024xf32>
      }
    )mlir",
    DispatchConfig{"IreeChosenTiling",
                   /*tileSizes=*/{32, 8}, // [d0, d1]
                   /*expectedVector=*/3,
                   /*expectedFailure=*/"",
                   "Measured: HvxVR=1."
                   "Real usage directly accumulates into the reg, no need "
                   "to spend extra registers loading lhs first."},
    DispatchConfig{
        "OneRowAtATime",
        /*tileSizes=*/{1, 1024},
        // %in is vectorized along the reduction (d1 is its lane dim), so it
        // keeps d1 at its real tile size, flattened: 1024 elements ->
        // ceil(1024*32/1024)=32; the parallel d0=1 contributes nothing. The
        // accumulator is [1] -> 1. Peak is the load and the accumulator both
        // live at once, just before the addf consumes them: 32 + 1 = 33.
        /*expectedVector=*/33,
        /*expectedFailure=*/"",
        "measured HvxVR=33; the whole reduction tile has to be gathered before "
        "it can be folded, so it dominates over the accumulator",
        // 1024*4 B of %in fill their 32 registers; the [1] accumulator holds
        // 4 B of its register: 4096 + 4.
        /*expectedUsefulBytes=*/4100},
    DispatchConfig{
        "FullReductionExtent",
        /*tileSizes=*/{32, 1024}, // [d0, d1]: the whole reduction axis
        // Same d1=1024 flattened to 32 - the parallel d0 growing from 1 to 32
        // still contributes nothing, one row is live at a time. The
        // accumulator is [32] -> 1 (32 elements pack into a single register).
        // Peak: 32 + 1 = 33.
        /*expectedVector=*/33,
        /*expectedFailure=*/"",
        "measured HvxVR=33; widening the parallel dim doesn't change either "
        "number, only the reduction tile's own width does",
        // 1024*4 B of %in + 32*4 B of accumulator, all full.
        /*expectedUsefulBytes=*/4224},
    DispatchConfig{
        "QuaterReductionExtent",
        /*tileSizes=*/{32, 256}, // [d0, d1]
        /*expectedVector=*/16,
        /*expectedFailure=*/"",
        "Measured: 16 - 8 for the reduction tile and 8 accumulators"},
    DispatchConfig{"QuaterReductionExtent2",
                   /*tileSizes=*/{64, 256}, // [d0, d1]
                   /*expectedVector=*/16,
                   /*expectedFailure=*/"", "Measured: 16"});

/// reduction sum operation transposed (column sum)
REGISTER_ESTIMATION_TEST_SUITE(ReduceSumTransposed, R"mlir(
    func.func @reduce_sum_transposed(%in: tensor<1024x1024xf32>, %out: tensor<1024xf32>) -> tensor<1024xf32> {
      %0 = linalg.generic {
        indexing_maps = [affine_map<(d0, d1) -> (d1, d0)>,
                          affine_map<(d0, d1) -> (d0)>],
        iterator_types = ["parallel", "reduction"]
      } ins(%in : tensor<1024x1024xf32>) outs(%out : tensor<1024xf32>) {
      ^bb0(%in0: f32, %init: f32):
        %sum = arith.addf %in0, %init : f32
        linalg.yield %sum : f32
      } -> tensor<1024xf32>
      return %0 : tensor<1024xf32>
    }
  )mlir",
                               DispatchConfig{
                                   "Transposed",
                                   /*tileSizes=*/{32, 8}, // 8 reduction lanes
                                                          // accumulator + input
                                   /*expectedVector=*/2,
                                   /*expectedFailure=*/"",
                                   "Measured: 11, "
                                   "Shuffleing overhead dominates"});

} // namespace
} // namespace mlir::iree_compiler::hexagon::codegen::planning

// Copyright 2026 RooflineAI GmbH
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "TestUtils.h"

namespace mlir::iree_compiler::hexagon::codegen::planning {
namespace {

/// linalg.depthwise_conv_2d_nhwc_hwc over a 1x114x114x128 input /
/// 3x3x128 filter / 1x112x112x128 output.
/// 6 loops (d0=N, d1=OH, d2=OW, d3=C, d4=KH, d5=KW); d4/d5 are reduction,
/// the rest parallel. The input's map `(d0, d1+d4, d2+d5, d3)` depends on
/// all 6 dims (the sliding window sums OH+KH and OW+KW, same mechanism as
/// Conv2dTest's gather); the filter depends on (d3, d4, d5); the output
/// depends on (d0, d1, d2, d3).
REGISTER_ESTIMATION_TEST_SUITE(
    DepthwiseConv2d,
    R"mlir(
      func.func @depthwise_conv2d(%in: tensor<1x114x114x128xf32>,
                                   %filter: tensor<3x3x128xf32>,
                                   %out: tensor<1x112x112x128xf32>) -> tensor<1x112x112x128xf32> {
        %0 = linalg.generic {
          indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1 + d4, d2 + d5, d3)>,
                            affine_map<(d0, d1, d2, d3, d4, d5) -> (d4, d5, d3)>,
                            affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3)>],
          iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction"]
        } ins(%in, %filter : tensor<1x114x114x128xf32>, tensor<3x3x128xf32>)
          outs(%out : tensor<1x112x112x128xf32>) {
        ^bb0(%in0: f32, %in1: f32, %init: f32):
          %mul = arith.mulf %in0, %in1 : f32
          %add = arith.addf %init, %mul : f32
          linalg.yield %add : f32
        } -> tensor<1x112x112x128xf32>
        return %0 : tensor<1x112x112x128xf32>
      }
    )mlir",
    DispatchConfig{
        "IreeChosenTiling",
        /*tileSizes=*/{1, 1, 28, 32, 3, 3}, // [N, OH, OW, C, KH, KW]
        /*expectedVector=*/-1,
        /*expectedFailure=*/"",
        "measured: 1. In this case, the OP didn't properly vectorize"},
    DispatchConfig{"ChannelOnlyTileApproximatesSingleVectorGranularity",
                   /*tileSizes=*/{1, 1, 1, 32, 3, 3}, // [N, OH, OW, C, KH, KW]
                   /*expectedVector=*/3,
                   /*expectedFailure=*/"", "measured: 1."});

/// The same depthwise convolution with both inputs in a channels-first layout
REGISTER_ESTIMATION_TEST_SUITE(
    DepthwiseConv2dTransposed, R"mlir(
    func.func @transposed_depthwise_conv2d(%in: tensor<1x128x114x114xf32>,
                                 %filter: tensor<128x3x3xf32>,
                                 %out: tensor<1x112x112x128xf32>) -> tensor<1x112x112x128xf32> {
      %0 = linalg.generic {
        indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d3, d1 + d4, d2 + d5)>,
                          affine_map<(d0, d1, d2, d3, d4, d5) -> (d3, d4, d5)>,
                          affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3)>],
        iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction"]
      } ins(%in, %filter : tensor<1x128x114x114xf32>, tensor<128x3x3xf32>)
        outs(%out : tensor<1x112x112x128xf32>) {
      ^bb0(%in0: f32, %in1: f32, %init: f32):
        %mul = arith.mulf %in0, %in1 : f32
        %add = arith.addf %init, %mul : f32
        linalg.yield %add : f32
      } -> tensor<1x112x112x128xf32>
      return %0 : tensor<1x112x112x128xf32>
    }
  )mlir",
    DispatchConfig{
        "TransposedIreeChosenTiling",
        /*tileSizes=*/{1, 1, 28, 32, 3, 3}, // [N, OH, OW, C, KH, KW]
                                            // accumulator is the output
                                            // 28 accu + in 3 + filter 9
        /*expectedVector=*/-1,
        /*expectedFailure=*/"",
        "measured HvxVR=0; as above: the op doesnt vectorize at all"},
    DispatchConfig{"TransposedPartialKernelTile",
                   /*tileSizes=*/{1, 1, 28, 32, 1, 3}, // KH tiled to 1 of 3
                   // accumulator 28 + in 1 + filter 3
                   // + 3 for the filter tile's transposed copy, written while
                   // the gathered source is read
                   /*expectedVector=*/35,
                   /*expectedFailure=*/"", "Measured HvxVR=0; see above"});

} // namespace
} // namespace mlir::iree_compiler::hexagon::codegen::planning

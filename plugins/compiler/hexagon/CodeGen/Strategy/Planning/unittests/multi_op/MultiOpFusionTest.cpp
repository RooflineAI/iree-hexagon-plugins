// Copyright 2026 RooflineAI GmbH
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "TestUtils.h"

// Several linalg ops fused into one nest.

namespace mlir::iree_compiler::hexagon::codegen::planning {
namespace {

/// Three `ElementwiseAddTest.cpp` bodies chained in one func, same
/// shape and maps as that file's `add`.
REGISTER_ESTIMATION_TEST_SUITE(
    ElementwiseChain,
    R"mlir(
      func.func @chain(%a: tensor<150528xf32>, %b: tensor<150528xf32>,
                       %c: tensor<150528xf32>, %d: tensor<150528xf32>,
                       %e0: tensor<150528xf32>, %e1: tensor<150528xf32>,
                       %e2: tensor<150528xf32>) -> tensor<150528xf32> {
        %0 = linalg.generic {
          indexing_maps = [affine_map<(d0) -> (d0)>,
                            affine_map<(d0) -> (d0)>,
                            affine_map<(d0) -> (d0)>],
          iterator_types = ["parallel"]
        } ins(%a, %b : tensor<150528xf32>, tensor<150528xf32>)
          outs(%e0 : tensor<150528xf32>) {
        ^bb0(%in0: f32, %in1: f32, %init: f32):
          %sum = arith.addf %in0, %in1 : f32
          linalg.yield %sum : f32
        } -> tensor<150528xf32>
        %1 = linalg.generic {
          indexing_maps = [affine_map<(d0) -> (d0)>,
                            affine_map<(d0) -> (d0)>,
                            affine_map<(d0) -> (d0)>],
          iterator_types = ["parallel"]
        } ins(%0, %c : tensor<150528xf32>, tensor<150528xf32>)
          outs(%e1 : tensor<150528xf32>) {
        ^bb0(%in0: f32, %in1: f32, %init: f32):
          %sum = arith.addf %in0, %in1 : f32
          linalg.yield %sum : f32
        } -> tensor<150528xf32>
        %2 = linalg.generic {anchor,
          indexing_maps = [affine_map<(d0) -> (d0)>,
                            affine_map<(d0) -> (d0)>,
                            affine_map<(d0) -> (d0)>],
          iterator_types = ["parallel"]
        } ins(%1, %d : tensor<150528xf32>, tensor<150528xf32>)
          outs(%e2 : tensor<150528xf32>) {
        ^bb0(%in0: f32, %in1: f32, %init: f32):
          %sum = arith.addf %in0, %in1 : f32
          linalg.yield %sum : f32
        } -> tensor<150528xf32>
        return %2 : tensor<150528xf32>
      }
    )mlir",
    DispatchConfig{"FusedChainStreams",
                   /*tileSizes=*/{32},
                   // 1 register per tile. Peak at any add:
                   // 3 live - min(2 dying, 1 result) = 2.
                   /*expectedVector=*/2,
                   /*expectedFailure=*/"", "measured HvxVR=4"},
    DispatchConfig{
        "WiderTileScalesEveryLiveTile",
        /*tileSizes=*/{256},
        // 8 per tile, two live -> 16.
        /*expectedVector=*/16,
        /*expectedFailure=*/"",
        "measured HvxVR=32; still two live tiles, each eight registers wide"});

/// `ElementwiseAddTest.cpp`'s body as the producer and
/// `TransposeTest.cpp`'s `transpose` verbatim as the consumer
REGISTER_ESTIMATION_TEST_SUITE(
    FusedTranspose,
    R"mlir(
      func.func @fused_transpose(%a: tensor<1024x1024xf32>,
                                 %b: tensor<1024x1024xf32>,
                                 %e0: tensor<1024x1024xf32>,
                                 %out: tensor<1024x1024xf32>)
                                 -> tensor<1024x1024xf32> {
        %0 = linalg.generic {
          indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                            affine_map<(d0, d1) -> (d0, d1)>,
                            affine_map<(d0, d1) -> (d0, d1)>],
          iterator_types = ["parallel", "parallel"]
        } ins(%a, %b : tensor<1024x1024xf32>, tensor<1024x1024xf32>)
          outs(%e0 : tensor<1024x1024xf32>) {
        ^bb0(%in0: f32, %in1: f32, %init: f32):
          %sum = arith.addf %in0, %in1 : f32
          linalg.yield %sum : f32
        } -> tensor<1024x1024xf32>
        %1 = linalg.generic {anchor,
          indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                            affine_map<(d0, d1) -> (d1, d0)>],
          iterator_types = ["parallel", "parallel"]
        } ins(%0 : tensor<1024x1024xf32>) outs(%out : tensor<1024x1024xf32>) {
        ^bb0(%in0: f32, %init: f32):
          linalg.yield %in0 : f32
        } -> tensor<1024x1024xf32>
        return %1 : tensor<1024x1024xf32>
      }
    )mlir",
    DispatchConfig{"SquareTile",
                   /*tileSizes=*/{32, 32},
                   // source [32,32]: 32 rows * ceil(32*32/1024)=1 -> 32.
                   // destination [32,32] likewise 32. Both live -> 64.
                   /*expectedVector=*/64,
                   /*expectedFailure=*/"",
                   "measured HvxVR=124; shuffle overhead"},
    DispatchConfig{
        "ThinTileTransposesBadly",
        /*tileSizes=*/{4, 32},
        // source [4,32]: 4 * ceil(32*32/1024)=1 -> 4.
        // destination [32,4]: 32 rows * ceil(4*32/1024)=1 -> 32,
        // register. 4 + 32 -> 36.
        /*expectedVector=*/36,
        /*expectedFailure=*/"",
        "measured HvxVR=6; shuffle cost is hiding form vec registers"});

/// `BroadcastScaleShiftTest.cpp`'s `broadcast_scale_shift` verbatim:
/// a 3136x64 tile scaled and shifted by two 64-element vectors read through
/// `(d0, d1) -> (d1)`.
REGISTER_ESTIMATION_TEST_SUITE(
    BroadcastScaleShift,
    R"mlir(
      func.func @broadcast_scale_shift(%x: tensor<3136x64xf32>, %scale: tensor<64xf32>,
                                        %shift: tensor<64xf32>, %out: tensor<3136x64xf32>)
                                        -> tensor<3136x64xf32> {
        %0 = linalg.generic {
          indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                            affine_map<(d0, d1) -> (d1)>,
                            affine_map<(d0, d1) -> (d1)>,
                            affine_map<(d0, d1) -> (d0, d1)>],
          iterator_types = ["parallel", "parallel"]
        } ins(%x, %scale, %shift : tensor<3136x64xf32>, tensor<64xf32>, tensor<64xf32>)
          outs(%out : tensor<3136x64xf32>) {
        ^bb0(%in0: f32, %in1: f32, %in2: f32, %init: f32):
          %mul = arith.mulf %in0, %in1 : f32
          %add = arith.addf %mul, %in2 : f32
          linalg.yield %add : f32
        } -> tensor<3136x64xf32>
        return %0 : tensor<3136x64xf32>
      }
    )mlir",
    DispatchConfig{"BroadcastsStayFlat",
                   /*tileSizes=*/{8, 64},
                   // %x [8,64]: 8 * ceil(64*32/1024)=2 -> 16, and so do mulf
                   // and addf. %scale and %shift [64] -> 2 each.
                   // Peak at the mulf: 16+2+2+16 = 36, minus 16 dying
                   //-> 20.
                   /*expectedVector=*/20,
                   /*expectedFailure=*/"", "measured HvxVR=21"},
    DispatchConfig{"BroadcastDoesNotGrowWithTheTile",
                   /*tileSizes=*/{64, 64},
                   // %x and the two body values are now 64*2 = 128 each,
                   // %scale and %shift are still 2. Peak at the mulf:
                   // 128+2+2+128 = 260 - 128 dying -> 132.
                   /*expectedVector=*/132,
                   /*expectedFailure=*/"",
                   "measured HvxVR=133; eight times the rows, but the two "
                   "broadcasts are still two registers"});

/// matmul followed by a generic: bias & relu
REGISTER_ESTIMATION_TEST_SUITE(
    MatmulBiasRelu,
    R"mlir(
      func.func @matmul_bias_relu(%lhs: tensor<256x256xf32>, %rhs: tensor<256x256xf32>,
                                  %acc: tensor<256x256xf32>, %bias: tensor<256xf32>,
                                  %out: tensor<256x256xf32>) -> tensor<256x256xf32> {
        %0 = linalg.matmul ins(%lhs, %rhs : tensor<256x256xf32>, tensor<256x256xf32>)
                            outs(%acc : tensor<256x256xf32>) {anchor}
                            -> tensor<256x256xf32>
        %1 = linalg.generic {
          indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                            affine_map<(d0, d1) -> (d1)>,
                            affine_map<(d0, d1) -> (d0, d1)>],
          iterator_types = ["parallel", "parallel"]
        } ins(%0, %bias : tensor<256x256xf32>, tensor<256xf32>)
          outs(%out : tensor<256x256xf32>) {
        ^bb0(%in: f32, %b: f32, %init: f32):
          %zero = arith.constant 0.000000e+00 : f32
          %biased = arith.addf %in, %b : f32
          %relu = arith.maximumf %biased, %zero : f32
          linalg.yield %relu : f32
        } -> tensor<256x256xf32>
        return %1 : tensor<256x256xf32>
      }
    )mlir",
    DispatchConfig{"EpilogueIsCheaperThanTheLoop",
                   /*tileSizes=*/{32, 32, 8},
                   // Prologue: the zero splat -> 1.
                   // Loop: %lhs keeps its own lane dim k=8, flattened:
                   // ceil(8*32/1024)=1. %rhs keeps its own lane dim n=32:
                   // ceil(32*32/1024)=1. accumulator [32,32] -> 32. Peak once
                   // %rhs is loaded, with the splat still live:
                   // 1 + 1 + 1 + 32 = 35.
                   // Epilogue: the reduced value 32, %bias [32] 1 and the
                   // splat, the addf taking over the reduced value's
                   // registers: 34 - still cheaper than the loop.
                   /*expectedVector=*/35,
                   /*expectedFailure=*/"",
                   "measured HvxVR=65, The accumulator is "
                   "carried as one [32 x <32 x float>] loop phi here, not the "
                   "32 separate phis of a plain matmul at this tiling "
                   "(measured 35), and 32 live-in plus 32 live-out copies "
                   "are live together"}
    // TODO investigate if this is a lowering bug
);

} // namespace
} // namespace mlir::iree_compiler::hexagon::codegen::planning

// Copyright 2026 RooflineAI GmbH
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "TestUtils.h"

// Dispatches that make up one decoder layer of an LLM, as IREE forms them.

namespace mlir::iree_compiler::hexagon::codegen::planning {
namespace {

/// Attention, dispatch 1: the scores. Q times K^T batched over heads, with
/// the 1/sqrt(64) scale fused as an epilogue.
REGISTER_ESTIMATION_TEST_SUITE(
    AttentionQk,
    R"mlir(
      func.func @attention_qk(%q: tensor<8x128x64xf32>, %k: tensor<8x128x64xf32>)
                              -> tensor<8x128x128xf32> {
        %zero = arith.constant 0.000000e+00 : f32
        %scale = arith.constant 1.250000e-01 : f32
        %empty = tensor.empty() : tensor<8x128x128xf32>
        %acc = linalg.fill ins(%zero : f32) outs(%empty : tensor<8x128x128xf32>)
                           -> tensor<8x128x128xf32>
        %0 = linalg.generic {anchor,
          indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>,
                           affine_map<(d0, d1, d2, d3) -> (d0, d2, d3)>,
                           affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>],
          iterator_types = ["parallel", "parallel", "parallel", "reduction"]
        } ins(%q, %k : tensor<8x128x64xf32>, tensor<8x128x64xf32>)
          outs(%acc : tensor<8x128x128xf32>) {
        ^bb0(%a: f32, %b: f32, %c: f32):
          %mul = arith.mulf %a, %b : f32
          %add = arith.addf %c, %mul : f32
          linalg.yield %add : f32
        } -> tensor<8x128x128xf32>
        %1 = linalg.generic {
          indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
                           affine_map<(d0, d1, d2) -> (d0, d1, d2)>],
          iterator_types = ["parallel", "parallel", "parallel"]
        } ins(%0 : tensor<8x128x128xf32>) outs(%empty : tensor<8x128x128xf32>) {
        ^bb0(%s: f32, %init: f32):
          %scaled = arith.mulf %s, %scale : f32
          linalg.yield %scaled : f32
        } -> tensor<8x128x128xf32>
        return %1 : tensor<8x128x128xf32>
      }
    )mlir",
    DispatchConfig{"IreeChosenTiling",
                   /*tileSizes=*/{1, 1, 1, 32},
                   // %q's chunk [32] along k -> 1. %k is reused across m and
                   // holds the output's lane dim n without having it
                   // innermost, so it is fully materialized: [1,32] flat ->
                   // 1, plus 1 for its transposed copy while it is loaded.
                   // accumulator [1,1,1] -> 1, and the scale splat is hoisted
                   // and live across the nest -> 1. Peak 5.
                   /*expectedVector=*/5,
                   /*expectedFailure=*/"",
                   "measured HvxVR=2. IREE picks a one-element output tile "
                   "for the transposed-B form: a 32-wide dot product, a "
                   "quarter of a register"},
    DispatchConfig{"WholeHeadDim",
                   /*tileSizes=*/{1, 8, 32, 64},
                   // %q keeps only its lane dim k: ceil(64*32/1024) = 2.
                   // %k fully materialized as above: 32*64 f32 flat -> 64,
                   // and 64 more for the transposed copy written while the
                   // gathered one is read. accumulator [8,32] -> 8, scale
                   // splat 1. Peak 139.
                   /*expectedVector=*/139,
                   /*expectedFailure=*/"",
                   "measured HvxVR=156; shuffle overhead"});

/// Attention, dispatch 1b: scaled Q*K^T at a larger, real-model shape (batch
/// of 4, 1024 tokens, head dim 128), with f16 inputs upcast to f32 and the
/// 1/sqrt(128)-ish scale fused into the upcast rather than the epilogue.
REGISTER_ESTIMATION_TEST_SUITE(
    AttentionQkt,
    R"mlir(
      func.func @attention_qkt(%q: tensor<4x1024x128xf16>, %k: tensor<4x1024x128xf16>)
                               -> tensor<4x1024x1024xf32> {
        %cst = arith.constant 0.297301769 : f32
        %cst_0 = arith.constant 0.000000e+00 : f32
        %5 = tensor.empty() : tensor<4x1024x1024xf32>
        %6 = tensor.empty() : tensor<4x1024x128xf32>
        %7 = linalg.generic {
          indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
                           affine_map<(d0, d1, d2) -> (d0, d1, d2)>],
          iterator_types = ["parallel", "parallel", "parallel"]
        } ins(%q : tensor<4x1024x128xf16>) outs(%6 : tensor<4x1024x128xf32>) {
        ^bb0(%in: f16, %out: f32):
          %11 = arith.extf %in : f16 to f32
          %12 = arith.mulf %11, %cst : f32
          linalg.yield %12 : f32
        } -> tensor<4x1024x128xf32>
        %8 = linalg.generic {
          indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
                           affine_map<(d0, d1, d2) -> (d0, d1, d2)>],
          iterator_types = ["parallel", "parallel", "parallel"]
        } ins(%k : tensor<4x1024x128xf16>) outs(%6 : tensor<4x1024x128xf32>) {
        ^bb0(%in: f16, %out: f32):
          %11 = arith.extf %in : f16 to f32
          %12 = arith.mulf %11, %cst : f32
          linalg.yield %12 : f32
        } -> tensor<4x1024x128xf32>
        %9 = linalg.fill ins(%cst_0 : f32) outs(%5 : tensor<4x1024x1024xf32>) -> tensor<4x1024x1024xf32>
        %10 = linalg.batch_matmul
          indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>,
                           affine_map<(d0, d1, d2, d3) -> (d0, d2, d3)>,
                           affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>]
          {anchor}
          ins(%7, %8 : tensor<4x1024x128xf32>, tensor<4x1024x128xf32>) outs(%9 : tensor<4x1024x1024xf32>) -> tensor<4x1024x1024xf32>
        return %10 : tensor<4x1024x1024xf32>
      }
    )mlir",
    DispatchConfig{"IreeChosenTiling",
                   /*tileSizes=*/{1, 1, 1, 32},
                   /*expectedVector=*/5,
                   /*expectedFailure=*/"", "measured HvxVR=6",
                   /*expectedUsefulBytes=*/-1,
                   /*fusedRelayoutChunkPolicy=*/
                   FusedRelayoutChunkPolicy::EstimateAnyway},
    DispatchConfig{"WholeHeadDim",
                   /*tileSizes=*/{1, 8, 32, 128},
                   // needs 128 registers for k + 128 for its transposed copy
                   /*expectedVector=*/269,
                   /*expectedFailure=*/"",
                   "measured HvxVR=1250; far more shuffleing overhead"});

/// Attention, dispatch 2: the row max of the scores, the first half of
/// softmax. A plain reduction along the rows' innermost dim, like
/// single_op/ReduceSumTest.cpp's `ReduceSum` with maximumf and a -inf init.
REGISTER_ESTIMATION_TEST_SUITE(
    SoftmaxRowMax,
    R"mlir(
      func.func @softmax_row_max(%x: tensor<1024x128xf32>) -> tensor<1024xf32> {
        %ninf = arith.constant 0xFF800000 : f32
        %empty = tensor.empty() : tensor<1024xf32>
        %init = linalg.fill ins(%ninf : f32) outs(%empty : tensor<1024xf32>)
                            -> tensor<1024xf32>
        %0 = linalg.generic {anchor,
          indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                           affine_map<(d0, d1) -> (d0)>],
          iterator_types = ["parallel", "reduction"]
        } ins(%x : tensor<1024x128xf32>) outs(%init : tensor<1024xf32>) {
        ^bb0(%in: f32, %m: f32):
          %max = arith.maximumf %in, %m : f32
          linalg.yield %max : f32
        } -> tensor<1024xf32>
        return %0 : tensor<1024xf32>
      }
    )mlir",
    DispatchConfig{"IreeChosenTiling",
                   /*tileSizes=*/{32, 8},
                   // %x's chunk runs along the reduction: 8 f32 -> 1. The
                   // [32] accumulator -> 1. The row max folds across lanes,
                   // so the reduced value holds two copies of the
                   // accumulator while it runs: 1 + 2 -> 3.
                   /*expectedVector=*/3,
                   /*expectedFailure=*/"",
                   "measured HvxVR=1, peak outside the loop nest (bb.0)"},
    DispatchConfig{"OneWholeRow",
                   /*tileSizes=*/{1, 128},
                   // %x's chunk: 128 f32 -> 4, accumulator [1] -> 1. Peak 5.
                   /*expectedVector=*/5,
                   /*expectedFailure=*/"", "measured HvxVR=14"},
    DispatchConfig{"EightWholeRows",
                   /*tileSizes=*/{8, 128},
                   // As OneWholeRow: the parallel d0 adds nothing to %x's
                   // chunk, and an [8] accumulator is still one register.
                   /*expectedVector=*/5,
                   /*expectedFailure=*/"",
                   "measured HvxVR=14, same as OneWholeRow"});

/// Attention, dispatch 3: the rest of softmax. exp(x - max) summed along the
/// row, then every element divided by its row's sum, recomputing exp(x - max)
/// rather than keeping it. The sum is the anchor.
REGISTER_ESTIMATION_TEST_SUITE(
    SoftmaxExpSumDivide,
    R"mlir(
      func.func @softmax_exp_sum_divide(%x: tensor<1024x128xf32>, %max: tensor<1024xf32>)
                                        -> tensor<1024x128xf32> {
        %zero = arith.constant 0.000000e+00 : f32
        %sum_empty = tensor.empty() : tensor<1024xf32>
        %sum_init = linalg.fill ins(%zero : f32) outs(%sum_empty : tensor<1024xf32>)
                                -> tensor<1024xf32>
        %sum = linalg.generic {anchor,
          indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                           affine_map<(d0, d1) -> (d0)>,
                           affine_map<(d0, d1) -> (d0)>],
          iterator_types = ["parallel", "reduction"]
        } ins(%x, %max : tensor<1024x128xf32>, tensor<1024xf32>)
          outs(%sum_init : tensor<1024xf32>) {
        ^bb0(%in: f32, %m: f32, %s: f32):
          %shifted = arith.subf %in, %m : f32
          %e = math.exp %shifted : f32
          %acc = arith.addf %e, %s : f32
          linalg.yield %acc : f32
        } -> tensor<1024xf32>
        %empty = tensor.empty() : tensor<1024x128xf32>
        %0 = linalg.generic {
          indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                           affine_map<(d0, d1) -> (d0)>,
                           affine_map<(d0, d1) -> (d0)>,
                           affine_map<(d0, d1) -> (d0, d1)>],
          iterator_types = ["parallel", "parallel"]
        } ins(%x, %max, %sum : tensor<1024x128xf32>, tensor<1024xf32>, tensor<1024xf32>)
          outs(%empty : tensor<1024x128xf32>) {
        ^bb0(%in: f32, %m: f32, %s: f32, %init: f32):
          %shifted = arith.subf %in, %m : f32
          %e = math.exp %shifted : f32
          %p = arith.divf %e, %s : f32
          linalg.yield %p : f32
        } -> tensor<1024x128xf32>
        return %0 : tensor<1024x128xf32>
      }
    )mlir",
    DispatchConfig{"IreeChosenTiling",
                   /*tileSizes=*/{32, 8},
                   // At the divide's exp: exp's constants 12; the sum and
                   // %max, each [32] broadcast to the [32,8] chunk -> 32 each;
                   // exp(x - max) [32,8] -> 32 and its extra tile 32.
                   // Peak 140.
                   /*expectedVector=*/140,
                   /*expectedFailure=*/"",
                   "measured HvxVR=3 8-wide rows scalarize"},
    DispatchConfig{"OneWholeRow",
                   /*tileSizes=*/{1, 128},
                   // Either exp: constants 12, its [1,128] input 4 taken over
                   // by the result, one extra tile 4, %max 1 and the
                   // accumulator (or the sum) 1. Peak 22.
                   /*expectedVector=*/22,
                   /*expectedFailure=*/"",
                   "measured HvxVR=20; codegen computes exp(x - max) once "
                   "for both ops"},
    DispatchConfig{"EightWholeRows",
                   /*tileSizes=*/{8, 128},
                   // The reduction's exp runs on one row's chunk (22, as in
                   // OneWholeRow). The divide's runs on the whole [8,128]
                   // tile: constants 12, exp 32 and its extra tile 32, %max
                   // and the sum 1 each - the reduction is not tiled, so the
                   // phase runs once and nothing is held broadcast. Peak 78.
                   /*expectedVector=*/78,
                   /*expectedFailure=*/"", "measured HvxVR=77"});

/// Attention, dispatch 4: the probabilities times V. A plain batch matmul, V
/// read row-major, so this one is the row-times-matrix form.
REGISTER_ESTIMATION_TEST_SUITE(
    AttentionPv,
    R"mlir(
      func.func @attention_pv(%p: tensor<8x128x128xf32>, %v: tensor<8x128x64xf32>)
                              -> tensor<8x128x64xf32> {
        %zero = arith.constant 0.000000e+00 : f32
        %empty = tensor.empty() : tensor<8x128x64xf32>
        %acc = linalg.fill ins(%zero : f32) outs(%empty : tensor<8x128x64xf32>)
                           -> tensor<8x128x64xf32>
        %0 = linalg.batch_matmul {anchor}
               ins(%p, %v : tensor<8x128x128xf32>, tensor<8x128x64xf32>)
               outs(%acc : tensor<8x128x64xf32>) -> tensor<8x128x64xf32>
        return %0 : tensor<8x128x64xf32>
      }
    )mlir",
    DispatchConfig{"IreeChosenTiling",
                   /*tileSizes=*/{1, 8, 32, 8},
                   // %p's chunk along k: 8 f32 -> 1. %v keeps its lane dim
                   // n: 32 f32 -> 1. accumulator [8,32] -> 8. Peak 10.
                   /*expectedVector=*/10,
                   /*expectedFailure=*/"", "measured HvxVR=11"},
    DispatchConfig{"WholeHeadDim",
                   /*tileSizes=*/{1, 8, 64, 16},
                   // %p: 16 f32 -> 1. %v: n=64 -> 2. accumulator [8,64] ->
                   // 16. Peak 19.
                   /*expectedVector=*/19,
                   /*expectedFailure=*/"", "measured HvxVR=21"});

/// RMSNorm before attention and before the FFN: the sum of squares along the
/// hidden dim, then x * rsqrt(mean + eps) * weight, the weight read as a
/// broadcast along the rows.
REGISTER_ESTIMATION_TEST_SUITE(
    RmsNorm,
    R"mlir(
      func.func @rms_norm(%x: tensor<128x512xf32>, %w: tensor<512xf32>)
                          -> tensor<128x512xf32> {
        %zero = arith.constant 0.000000e+00 : f32
        %inv_n = arith.constant 1.953125e-03 : f32
        %eps = arith.constant 9.99999997E-7 : f32
        %empty = tensor.empty() : tensor<128x512xf32>
        %ss_empty = tensor.empty() : tensor<128xf32>
        %ss_init = linalg.fill ins(%zero : f32) outs(%ss_empty : tensor<128xf32>)
                               -> tensor<128xf32>
        %ss = linalg.generic {anchor,
          indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                           affine_map<(d0, d1) -> (d0)>],
          iterator_types = ["parallel", "reduction"]
        } ins(%x : tensor<128x512xf32>) outs(%ss_init : tensor<128xf32>) {
        ^bb0(%in: f32, %acc: f32):
          %sq = arith.mulf %in, %in : f32
          %sum = arith.addf %sq, %acc : f32
          linalg.yield %sum : f32
        } -> tensor<128xf32>
        %0 = linalg.generic {
          indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                           affine_map<(d0, d1) -> (d0)>,
                           affine_map<(d0, d1) -> (d1)>,
                           affine_map<(d0, d1) -> (d0, d1)>],
          iterator_types = ["parallel", "parallel"]
        } ins(%x, %ss, %w : tensor<128x512xf32>, tensor<128xf32>, tensor<512xf32>)
          outs(%empty : tensor<128x512xf32>) {
        ^bb0(%in: f32, %s: f32, %weight: f32, %init: f32):
          %mean = arith.mulf %s, %inv_n : f32
          %biased = arith.addf %mean, %eps : f32
          %rstd = math.rsqrt %biased : f32
          %norm = arith.mulf %in, %rstd : f32
          %scaled = arith.mulf %norm, %weight : f32
          linalg.yield %scaled : f32
        } -> tensor<128x512xf32>
        return %0 : tensor<128x512xf32>
      }
    )mlir",
    DispatchConfig{"IreeChosenTiling",
                   /*tileSizes=*/{32, 8},
                   // In the scale phase: the two constant splats 2, rsqrt's
                   // hoisted constants 2, %x's [32,8] chunk 32, %w [8] 1, and
                   // rstd [32] broadcast to [32,8] -> 32, written over the
                   // dying `+ eps`. Peak 69.
                   /*expectedVector=*/69,
                   /*expectedFailure=*/"",
                   "measured HvxVR=3, peak outside the loop nest (bb.0); 3 "
                   "8-wide rows scalarize"},
    DispatchConfig{"OneWholeRow",
                   /*tileSizes=*/{1, 512},
                   // Scale phase: splats 2, rsqrt's constants 2, %x's row 16,
                   // %w 16, rstd 1 - the reduction is not tiled, so it stays
                   // one splat. Peak 37.
                   /*expectedVector=*/37,
                   /*expectedFailure=*/"",
                   "measured HvxVR=33: %w 16 + %x 16 + the rstd splat; the "
                   "constants are folded into scalar code"},
    DispatchConfig{"EightRowsQuarterWidth",
                   /*tileSizes=*/{8, 128},
                   // Scale phase over 4 chunks of 128: splats 2, rsqrt's
                   // constants 2, %x's [8,128] chunk 32, %w [128] 4, rstd [8]
                   // broadcast to [8,128] 32. Peak 72.
                   /*expectedVector=*/72,
                   /*expectedFailure=*/"",
                   "measured HvxVR=54; codegen does hold rstd broadcast to "
                   "the [8,128] chunk across the loop. Where the other 16 "
                   "go is not investigated"},
    DispatchConfig{"SearchLegalChunk",
                   /*tileSizes=*/{32, 32},
                   // Same shape as IreeChosenTiling, but with the reduction
                   // chunk widened to 32 (one full native vector): splats 2,
                   // rsqrt's expansion constants 2, %x's [32,32] chunk 32,
                   // %w [32] 1, rstd [32] broadcast to [32,32] -> 32. Peak
                   // 69.
                   /*expectedVector=*/69,
                   /*expectedFailure=*/"",
                   "measured HvxVR=66 (get_reg_pressure.sh on this exact "
                   "tile, both dims search-legal - a multiple of the native "
                   "vector width, not the sub-register chunk that "
                   "IreeChosenTiling scalarizes). Confirms the broadcast "
                   "formula itself is sound: TileSelectionReport.md section "
                   "10.5 found no evidence for the ~20x broadcast bug the "
                   "IreeChosenTiling/[8,128] notes above suggested - both of "
                   "those are contaminated by scalarization, a separate, "
                   "already-documented gap the model deliberately does not "
                   "follow"});

/// FFN: the up projection x W_up^T with the SwiGLU gate fused as its
/// epilogue, silu(gate) * up. The weight is stored [out, in] as torch does,
/// so like AttentionQk both operands have the reduction dim innermost, and
/// the weight tile is transposed.
REGISTER_ESTIMATION_TEST_SUITE(
    FfnUpSwiGlu,
    R"mlir(
      func.func @ffn_up_swiglu(%x: tensor<128x512xf32>, %w_up: tensor<1408x512xf32>,
                               %gate: tensor<128x1408xf32>) -> tensor<128x1408xf32> {
        %zero = arith.constant 0.000000e+00 : f32
        %one = arith.constant 1.000000e+00 : f32
        %empty = tensor.empty() : tensor<128x1408xf32>
        %acc = linalg.fill ins(%zero : f32) outs(%empty : tensor<128x1408xf32>)
                           -> tensor<128x1408xf32>
        %up = linalg.generic {anchor,
          indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>,
                           affine_map<(d0, d1, d2) -> (d1, d2)>,
                           affine_map<(d0, d1, d2) -> (d0, d1)>],
          iterator_types = ["parallel", "parallel", "reduction"]
        } ins(%x, %w_up : tensor<128x512xf32>, tensor<1408x512xf32>)
          outs(%acc : tensor<128x1408xf32>) {
        ^bb0(%a: f32, %b: f32, %c: f32):
          %mul = arith.mulf %a, %b : f32
          %add = arith.addf %c, %mul : f32
          linalg.yield %add : f32
        } -> tensor<128x1408xf32>
        %0 = linalg.generic {
          indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                           affine_map<(d0, d1) -> (d0, d1)>,
                           affine_map<(d0, d1) -> (d0, d1)>],
          iterator_types = ["parallel", "parallel"]
        } ins(%gate, %up : tensor<128x1408xf32>, tensor<128x1408xf32>)
          outs(%empty : tensor<128x1408xf32>) {
        ^bb0(%g: f32, %u: f32, %init: f32):
          %neg = arith.negf %g : f32
          %e = math.exp %neg : f32
          %den = arith.addf %e, %one : f32
          %silu = arith.divf %g, %den : f32
          %prod = arith.mulf %silu, %u : f32
          linalg.yield %prod : f32
        } -> tensor<128x1408xf32>
        return %0 : tensor<128x1408xf32>
      }
    )mlir",
    DispatchConfig{"IreeChosenTiling",
                   /*tileSizes=*/{1, 1, 32},
                   // In the loop: exp's constants 12 and the 1.0 splat,
                   // hoisted; %x's chunk 32 f32 -> 1, %w_up fully materialized
                   // [1,32] -> 1 plus 1 for its transposed copy, accumulator
                   // 1. Peak 17.
                   /*expectedVector=*/17,
                   /*expectedFailure=*/"",
                   "measured HvxVR=2: one output element per tile, scalarizes"},
    DispatchConfig{"SquareOutputTile",
                   /*tileSizes=*/{8, 32, 64},
                   // In the loop: exp's constants 12, splat 1, %x's chunk 64
                   // f32 -> 2, %w_up fully materialized 32*64 f32 -> 64 plus
                   // 64 for its transposed copy, accumulator [8,32] 8.
                   // Peak 151.
                   /*expectedVector=*/151,
                   /*expectedFailure=*/"",
                   "measured HvxVR=164; shuffle overhead"});

} // namespace
} // namespace mlir::iree_compiler::hexagon::codegen::planning

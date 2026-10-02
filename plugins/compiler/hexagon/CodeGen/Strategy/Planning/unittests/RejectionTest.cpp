// Copyright 2026 RooflineAI GmbH
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "TestUtils.h"

// Dispatches the model refuses to model at all.

namespace mlir::iree_compiler::hexagon::codegen::planning {
namespace {

/// Case 17, dynamic shapes: case 1's IR with its extent replaced by `?`.
REGISTER_ESTIMATION_TEST_SUITE(
    DynamicExtent,
    R"mlir(
      func.func @add(%a: tensor<?xf32>, %b: tensor<?xf32>,
                      %out: tensor<?xf32>) -> tensor<?xf32> {
        %0 = linalg.generic {
          indexing_maps = [affine_map<(d0) -> (d0)>,
                            affine_map<(d0) -> (d0)>,
                            affine_map<(d0) -> (d0)>],
          iterator_types = ["parallel"]
        } ins(%a, %b : tensor<?xf32>, tensor<?xf32>)
          outs(%out : tensor<?xf32>) {
        ^bb0(%in0: f32, %in1: f32, %init: f32):
          %sum = arith.addf %in0, %in1 : f32
          linalg.yield %sum : f32
        } -> tensor<?xf32>
        return %0 : tensor<?xf32>
      }
    )mlir",
    DispatchConfig{"RejectedAtBuild",
                   /*tileSizes=*/{32},
                   /*expectedVector=*/0,
                   /*expectedFailure=*/"dynamic loop extent",
                   "no static extent means no tile shape to weigh"});

/// a non-linalg op in the dispatch: `tensor.extract_slice`
REGISTER_ESTIMATION_TEST_SUITE(
    NonLinalgOpInDispatch,
    R"mlir(
      func.func @sliced_chain(%a: tensor<150528xf32>, %b: tensor<150528xf32>,
                              %c: tensor<1024xf32>, %e0: tensor<150528xf32>,
                              %e1: tensor<1024xf32>) -> tensor<1024xf32> {
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
        %slice = tensor.extract_slice %0[0] [1024] [1]
          : tensor<150528xf32> to tensor<1024xf32>
        %1 = linalg.generic {anchor,
          indexing_maps = [affine_map<(d0) -> (d0)>,
                            affine_map<(d0) -> (d0)>,
                            affine_map<(d0) -> (d0)>],
          iterator_types = ["parallel"]
        } ins(%slice, %c : tensor<1024xf32>, tensor<1024xf32>)
          outs(%e1 : tensor<1024xf32>) {
        ^bb0(%in0: f32, %in1: f32, %init: f32):
          %sum = arith.addf %in0, %in1 : f32
          linalg.yield %sum : f32
        } -> tensor<1024xf32>
        return %1 : tensor<1024xf32>
      }
    )mlir",
    DispatchConfig{"RejectedByName",
                   /*tileSizes=*/{32},
                   /*expectedVector=*/0,
                   /*expectedFailure=*/"tensor.extract_slice",
                   "the message has to name the op, or the caller cannot "
                   "tell which part of its dispatch is unsupported"});

/// conflicting shared slices: one value read by two consumers in
/// different dim orders. Op 2 reads %0 transposed while op 1 reads it
/// straight, so no single tile of %0 can serve both
REGISTER_ESTIMATION_TEST_SUITE(
    ConflictingSharedSlices,
    R"mlir(
      func.func @two_readers(%a: tensor<1024x1024xf32>, %b: tensor<1024x1024xf32>,
                             %e0: tensor<1024x1024xf32>, %e1: tensor<1024x1024xf32>,
                             %out: tensor<1024x1024xf32>) -> tensor<1024x1024xf32> {
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
        %1 = linalg.generic {
          indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                            affine_map<(d0, d1) -> (d0, d1)>],
          iterator_types = ["parallel", "parallel"]
        } ins(%0 : tensor<1024x1024xf32>) outs(%e1 : tensor<1024x1024xf32>) {
        ^bb0(%in0: f32, %init: f32):
          %double = arith.addf %in0, %in0 : f32
          linalg.yield %double : f32
        } -> tensor<1024x1024xf32>
        %2 = linalg.generic {anchor,
          indexing_maps = [affine_map<(d0, d1) -> (d1, d0)>,
                            affine_map<(d0, d1) -> (d0, d1)>,
                            affine_map<(d0, d1) -> (d0, d1)>],
          iterator_types = ["parallel", "parallel"]
        } ins(%0, %1 : tensor<1024x1024xf32>, tensor<1024x1024xf32>)
          outs(%out : tensor<1024x1024xf32>) {
        ^bb0(%in0: f32, %in1: f32, %init: f32):
          %sum = arith.addf %in0, %in1 : f32
          linalg.yield %sum : f32
        } -> tensor<1024x1024xf32>
        return %2 : tensor<1024x1024xf32>
      }
    )mlir",
    DispatchConfig{"RejectedAtBuild",
                   /*tileSizes=*/{32, 32},
                   /*expectedVector=*/0,
                   /*expectedFailure=*/"two different tile mappings",
                   "one straight reader and one transposed reader of the "
                   "same value cannot share a tile"});

/// A fused consumer that reduces over a dim the anchor iterates in parallel
/// in this case, a wrong anchor is given
REGISTER_ESTIMATION_TEST_SUITE(
    ReductionOverParallelAnchorDim,
    R"mlir(
      func.func @square_then_row_sum(%a: tensor<64x1024xf32>,
                                     %e0: tensor<64x1024xf32>,
                                     %out: tensor<64xf32>) -> tensor<64xf32> {
        %0 = linalg.generic {anchor,
          indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                            affine_map<(d0, d1) -> (d0, d1)>],
          iterator_types = ["parallel", "parallel"]
        } ins(%a : tensor<64x1024xf32>) outs(%e0 : tensor<64x1024xf32>) {
        ^bb0(%x: f32, %init: f32):
          %square = arith.mulf %x, %x : f32
          linalg.yield %square : f32
        } -> tensor<64x1024xf32>
        %1 = linalg.generic {
          indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                            affine_map<(d0, d1) -> (d0)>],
          iterator_types = ["parallel", "reduction"]
        } ins(%0 : tensor<64x1024xf32>) outs(%out : tensor<64xf32>) {
        ^bb0(%x: f32, %acc: f32):
          %sum = arith.addf %x, %acc : f32
          linalg.yield %sum : f32
        } -> tensor<64xf32>
        return %1 : tensor<64xf32>
      }
    )mlir",
    DispatchConfig{"RejectedAtBuild",
                   /*tileSizes=*/{1, 256},
                   /*expectedVector=*/0,
                   /*expectedFailure=*/"reduces over anchor dim",
                   "the row sum's reduction loop lands on the anchor's "
                   "parallel d1; anchoring at the sum is the supported "
                   "form"});

} // namespace
} // namespace mlir::iree_compiler::hexagon::codegen::planning

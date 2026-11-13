// Copyright 2026 RooflineAI GmbH
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RUN: iree-opt --pass-pipeline='builtin.module(func.func(iree-hexagon-tile-and-fuse-producer-consumer{tiling-level=cache_parallel}))' --split-input-file %s | FileCheck %s

// A tile that covers the whole static dim leaves a single-iteration forall.
// Promoting it inserts the tile into the whole dynamically shaped result. The
// insert must fold away so that the result bufferizes in place instead of
// through a temporary of the whole result.

#config = #iree_hexagon.lowering_config<cache_parallel = [9, 0], distribution = [0, 0], vector_common_parallel = [1, 32]>
#map = affine_map<(d0, d1) -> (d0, d1)>
func.func @full_tile_dynamic(%arg0: tensor<9x?xf16>, %arg1: tensor<9x?xf16>) -> tensor<9x?xf16> {
  %c1 = arith.constant 1 : index
  %dim = tensor.dim %arg0, %c1 : tensor<9x?xf16>
  %empty = tensor.empty(%dim) : tensor<9x?xf16>
  %0 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%arg0, %arg1 : tensor<9x?xf16>, tensor<9x?xf16>) outs(%empty : tensor<9x?xf16>) attrs = {lowering_config = #config} {
  ^bb0(%in: f16, %in_0: f16, %out: f16):
    %1 = arith.addf %in, %in_0 : f16
    linalg.yield %1 : f16
  } -> tensor<9x?xf16>
  return %0 : tensor<9x?xf16>
}
// CHECK-LABEL: func.func @full_tile_dynamic(
// CHECK-NOT:     scf.forall
// CHECK-NOT:     tensor.insert_slice
// CHECK:         %[[GENERIC:.+]] = linalg.generic
// CHECK-NOT:     tensor.insert_slice
// CHECK:         return %[[GENERIC]]

// -----

// An insert that covers only part of its destination is kept.

#config = #iree_hexagon.lowering_config<cache_parallel = [9, 0], distribution = [0, 0], vector_common_parallel = [1, 32]>
#map = affine_map<(d0, d1) -> (d0, d1)>
func.func @partial_insert_kept(%arg0: tensor<9x?xf16>, %arg1: tensor<9x?xf16>, %dest: tensor<9x?xf16>) -> tensor<9x?xf16> {
  %c1 = arith.constant 1 : index
  %dim = tensor.dim %arg0, %c1 : tensor<9x?xf16>
  %empty = tensor.empty(%dim) : tensor<9x?xf16>
  %0 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%arg0, %arg1 : tensor<9x?xf16>, tensor<9x?xf16>) outs(%empty : tensor<9x?xf16>) attrs = {lowering_config = #config} {
  ^bb0(%in: f16, %in_0: f16, %out: f16):
    %1 = arith.addf %in, %in_0 : f16
    linalg.yield %1 : f16
  } -> tensor<9x?xf16>
  %2 = tensor.insert_slice %0 into %dest[0, 0] [9, %dim] [1, 1] : tensor<9x?xf16> into tensor<9x?xf16>
  return %2 : tensor<9x?xf16>
}
// CHECK-LABEL: func.func @partial_insert_kept(
// CHECK-SAME:    %[[DEST:[a-zA-Z0-9_]+]]: tensor<9x?xf16>)
// CHECK:         %[[GENERIC:.+]] = linalg.generic
// CHECK:         %[[INSERT:.+]] = tensor.insert_slice %[[GENERIC]] into %[[DEST]]
// CHECK:         return %[[INSERT]]

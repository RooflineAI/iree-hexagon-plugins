// Copyright 2026 RooflineAI GmbH
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Additional local coverage. The planner expects fills never being refined
// by this pass (see canRefineComputeTileDownstream in the planning library).
// If this test fails, consider updating the planner as well.
// RUN: iree-opt --pass-pipeline='builtin.module(func.func(iree-hexagon-tile-to-vector-size))' --split-input-file %s | FileCheck %s

// A configured fill is skipped; an unconfigured consumer is also left alone.
// CHECK-LABEL: func.func @skip_fill_and_unconfigured_consumer(
// CHECK-NOT: scf.for
// CHECK: linalg.fill
// CHECK-NOT: scf.for
// CHECK: linalg.generic
// CHECK-NOT: scf.for
// CHECK: return
func.func @skip_fill_and_unconfigured_consumer(%input: tensor<16xf32>,
    %init: tensor<16xf32>) -> tensor<16xf32> {
  %c0 = arith.constant 0.0 : f32
  %filled = linalg.fill
      {lowering_config = #iree_hexagon.lowering_config<vector_common_parallel = [4]>}
      ins(%c0 : f32) outs(%init : tensor<16xf32>) -> tensor<16xf32>
  %consumer = linalg.generic {
      indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
      iterator_types = ["parallel"]}
      ins(%input : tensor<16xf32>) outs(%filled : tensor<16xf32>) {
  ^bb0(%in: f32, %out: f32):
    %sum = arith.addf %in, %out : f32
    linalg.yield %sum : f32
  } -> tensor<16xf32>
  return %consumer : tensor<16xf32>
}

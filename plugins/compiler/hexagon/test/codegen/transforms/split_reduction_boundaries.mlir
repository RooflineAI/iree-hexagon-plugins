// Copyright 2026 RooflineAI GmbH
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Additional local coverage for the configuration lifecycle.
// RUN: iree-opt --pass-pipeline='builtin.module(func.func(iree-hexagon-split-reduction,cse,canonicalize))' --split-input-file %s | FileCheck %s
// RUN: iree-opt --pass-pipeline='builtin.module(func.func(iree-hexagon-split-reduction))' --split-input-file %s | FileCheck %s --check-prefix=LIFECYCLE

// Integer reductions split without enabling FP reordering. Partial reductions
// start at the neutral element, and the final reduction uses the original init.
// CHECK-LABEL: func.func @static_integer_reduction(
// CHECK-SAME: %[[INPUT:.+]]: tensor<32xi32>, %[[INIT:.+]]: tensor<i32>
// CHECK: %[[ZERO:.+]] = arith.constant 0 : i32
// CHECK: %[[EXPANDED:.+]] = tensor.expand_shape %[[INPUT]]
// CHECK-SAME: tensor<32xi32> into tensor<4x8xi32>
// CHECK: %[[FILLED:.+]] = linalg.fill ins(%[[ZERO]]
// CHECK: scf.for {{.*}} iter_args({{.*}} = %[[FILLED]]) -> (tensor<8xi32>)
// CHECK: linalg.generic
// CHECK-SAME: iterator_types = ["reduction", "parallel"]
// CHECK: %[[FINAL:.+]] = linalg.generic
// CHECK-SAME: outs(%[[INIT]] : tensor<i32>)
// CHECK: return %[[FINAL]]

// Newly created operations have different iteration spaces; do not clone the
// original configuration onto them.
// LIFECYCLE-LABEL: func.func @static_integer_reduction(
// LIFECYCLE-NOT: lowering_config
// LIFECYCLE: return
func.func @static_integer_reduction(%input: tensor<32xi32>, %init: tensor<i32>)
    -> tensor<i32> {
  %result = linalg.generic {
      indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> ()>],
      iterator_types = ["reduction"]}
      ins(%input : tensor<32xi32>) outs(%init : tensor<i32>)
      attrs = {lowering_config = #iree_cpu.lowering_config<vector_reduction = [8]>} {
  ^bb0(%in: i32, %out: i32):
    %sum = arith.addi %in, %out : i32
    linalg.yield %sum : i32
  } -> tensor<i32>
  return %result : tensor<i32>
}

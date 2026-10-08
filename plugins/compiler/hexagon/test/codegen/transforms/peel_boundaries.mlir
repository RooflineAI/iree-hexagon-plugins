// Copyright 2026 RooflineAI GmbH
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Additional local coverage for the iterator-rank limit.
// RUN: iree-opt --pass-pipeline='builtin.module(func.func(iree-hexagon-peel))' --split-input-file %s | FileCheck %s

// The configured fill has one iterator. Peel only the immediately enclosing
// loop; the outer loop is outside its iterator space and must keep its bound.
// CHECK-LABEL: func.func @respect_iterator_rank(
// CHECK-DAG: %[[C10:.+]] = arith.constant 10 : index
// CHECK-DAG: %[[C8:.+]] = arith.constant 8 : index
// CHECK-DAG: %[[C4:.+]] = arith.constant 4 : index
// CHECK: %[[OUTER:.+]] = scf.for {{.*}} to %[[C10]] step %[[C4]]
// CHECK: scf.for {{.*}} to %[[C8]] step %[[C4]]
// CHECK: linalg.fill
// CHECK-SAME: -> tensor<4xf32>
// CHECK: linalg.fill
// CHECK-SAME: -> tensor<2xf32>
// CHECK-NOT: scf.for {{.*}}
// CHECK: return %[[OUTER]]
func.func @respect_iterator_rank(%init: tensor<10xf32>) -> tensor<10xf32> {
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %c10 = arith.constant 10 : index
  %zero = arith.constant 0.0 : f32
  %result = scf.for %outer = %c0 to %c10 step %c4 iter_args(%outer_acc = %init)
      -> (tensor<10xf32>) {
    %inner_result = scf.for %iv = %c0 to %c10 step %c4
        iter_args(%acc = %outer_acc) -> (tensor<10xf32>) {
      %size = affine.min affine_map<(d0) -> (10 - d0, 4)>(%iv)
      %slice = tensor.extract_slice %acc[%iv] [%size] [1]
          : tensor<10xf32> to tensor<?xf32>
      %filled = linalg.fill
          {lowering_config = #iree_hexagon.lowering_config<vector_common_parallel = [4]>}
          ins(%zero : f32) outs(%slice : tensor<?xf32>) -> tensor<?xf32>
      %inserted = tensor.insert_slice %filled into %acc[%iv] [%size] [1]
          : tensor<?xf32> into tensor<10xf32>
      scf.yield %inserted : tensor<10xf32>
    }
    scf.yield %inner_result : tensor<10xf32>
  }
  return %result : tensor<10xf32>
}

// Copyright 2026 RooflineAI GmbH
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RUN: iree-opt --pass-pipeline='builtin.module(func.func(iree-hexagon-tile-and-fuse-producer-consumer{tiling-level=vector_reduction only-fuse-producer-input-operands=true}),cse)' --split-input-file %s | FileCheck %s --check-prefix=INPUT
// RUN: iree-opt --pass-pipeline='builtin.module(func.func(iree-hexagon-tile-and-fuse-producer-consumer{tiling-level=vector_common_parallel}),cse)' --split-input-file %s | FileCheck %s --check-prefix=PARALLEL

// Reduction input fusion must leave the fill destination outside the serial
// loop and the epilogue outside the root's reduction loop. Parallel fusion may
// fuse both into its forall.
// INPUT-LABEL: func.func @input_only_excludes_destination_and_consumer(
// INPUT-DAG: %[[C4:.+]] = arith.constant 4 : index
// INPUT: %[[FILL:.+]] = linalg.fill
// INPUT-NOT: scf.forall
// INPUT: %[[LOOP:.+]] = scf.for {{.*}} step %[[C4]] iter_args({{.*}} = %[[FILL]])
// INPUT: linalg.generic
// INPUT-SAME: tensor<8x4xf32>
// INPUT-NOT: linalg.fill
// INPUT: %[[ROOT:.+]] = linalg.matmul
// INPUT: scf.yield %[[ROOT]]
// INPUT: }
// INPUT: %[[CONSUMER:.+]] = linalg.generic
// INPUT-SAME: ins(%[[LOOP]]
// INPUT: return %[[CONSUMER]]
// PARALLEL-LABEL: func.func @input_only_excludes_destination_and_consumer(
// PARALLEL: %[[LOOP:.+]] = scf.forall
// PARALLEL: linalg.generic
// PARALLEL: linalg.fill
// PARALLEL: linalg.matmul
// PARALLEL: linalg.generic
// PARALLEL: scf.forall.in_parallel
// PARALLEL: return %[[LOOP]]
func.func @input_only_excludes_destination_and_consumer(
    %lhs: tensor<8x16xf32>, %rhs: tensor<16x8xf32>, %init: tensor<8x8xf32>)
    -> tensor<8x8xf32> {
  %c0 = arith.constant 0.0 : f32
  %producer_empty = tensor.empty() : tensor<8x16xf32>
  %producer = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                       affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]}
      ins(%lhs : tensor<8x16xf32>) outs(%producer_empty : tensor<8x16xf32>) {
  ^bb0(%in: f32, %out: f32):
    %scaled = arith.mulf %in, %in : f32
    linalg.yield %scaled : f32
  } -> tensor<8x16xf32>
  %filled = linalg.fill ins(%c0 : f32) outs(%init : tensor<8x8xf32>)
      -> tensor<8x8xf32>
  %product = linalg.matmul
      {lowering_config = #iree_cpu.lowering_config<distribution = [8, 8, 0], vector_common_parallel = [4, 4, 0], vector_reduction = [0, 0, 4]>}
      ins(%producer, %rhs : tensor<8x16xf32>, tensor<16x8xf32>)
      outs(%filled : tensor<8x8xf32>) -> tensor<8x8xf32>
  %consumer = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                       affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]}
      ins(%product : tensor<8x8xf32>) outs(%init : tensor<8x8xf32>) {
  ^bb0(%in: f32, %out: f32):
    %sum = arith.addf %in, %in : f32
    linalg.yield %sum : f32
  } -> tensor<8x8xf32>
  return %consumer : tensor<8x8xf32>
}

// -----

// A producer behind tensor-level VTCM staging must remain outside the fused
// loop: slices use the staged value, preserving its allocation/copy boundary.
// PARALLEL-LABEL: func.func @vtcm_stage_is_a_fusion_barrier(
// PARALLEL: %[[PRODUCER:.+]] = linalg.generic
// PARALLEL: %[[STAGED:.+]] = iree_hexagon.stage_to_vtcm %[[PRODUCER]]
// PARALLEL: scf.forall
// PARALLEL: tensor.extract_slice %[[STAGED]]
// PARALLEL-NOT: arith.mulf
// PARALLEL: linalg.generic
// PARALLEL: arith.addf
// PARALLEL: scf.forall.in_parallel
// PARALLEL: return
func.func @vtcm_stage_is_a_fusion_barrier(%input: tensor<8xf32>,
    %init: tensor<8xf32>) -> tensor<8xf32> {
  %producer = linalg.generic {
      indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
      iterator_types = ["parallel"]}
      ins(%input : tensor<8xf32>) outs(%init : tensor<8xf32>) {
  ^bb0(%in: f32, %out: f32):
    %scaled = arith.mulf %in, %in : f32
    linalg.yield %scaled : f32
  } -> tensor<8xf32>
  %staged = iree_hexagon.stage_to_vtcm %producer : tensor<8xf32>
  %consumer = linalg.generic {
      indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
      iterator_types = ["parallel"],
      lowering_config = #iree_cpu.lowering_config<distribution = [8], vector_common_parallel = [4]>}
      ins(%staged : tensor<8xf32>) outs(%init : tensor<8xf32>) {
  ^bb0(%in: f32, %out: f32):
    %sum = arith.addf %in, %in : f32
    linalg.yield %sum : f32
  } -> tensor<8xf32>
  return %consumer : tensor<8xf32>
}

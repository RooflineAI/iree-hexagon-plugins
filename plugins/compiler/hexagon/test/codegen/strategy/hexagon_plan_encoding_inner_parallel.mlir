// A consumer dimension that the root does not have must reach
// VectorInnerParallel and must be realized as a loop by the downstream
// last-operation tiling pass. This is the accepted counterpart to
// hexagon_plan_verification_last_tiling_anchor.mlir covers the opposite
// ordering, which the current pipeline cannot consume.
//
// RUN: iree-opt \
// RUN:   --iree-hexagon-enable-vtcm-tiling=false \
// RUN:   --iree-hexagon-enable-hmx-matmul=false \
// RUN:   --pass-pipeline='builtin.module(iree-hexagon-select-lowering-strategy)' \
// RUN:   %s | FileCheck %s

#target = #hal.executable.target<"hexagon", "embedded-elf-hexagon", {cpu = "hexagonv79", cpu_features = "+hvxv79,+hvx-length128b", max_stack_allocation_size = 16384 : i64, target_triple = "hexagon-unknown-unknown-elf"}>

func.func @broadcast_consumer_after_root(%lhs: tensor<128x64xf32>, %rhs: tensor<64x32xf32>) -> tensor<128x32x64xf32> attributes {hal.executable.target = #target} {
  %cst = arith.constant 0.0 : f32
  %empty = tensor.empty() : tensor<128x32xf32>
  %init = linalg.fill ins(%cst : f32) outs(%empty : tensor<128x32xf32>) -> tensor<128x32xf32>
  %matmul = linalg.matmul ins(%lhs, %rhs : tensor<128x64xf32>, tensor<64x32xf32>) outs(%init : tensor<128x32xf32>) -> tensor<128x32xf32>
  %result_empty = tensor.empty() : tensor<128x32x64xf32>
  %result = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1)>,
                       affine_map<(d0, d1, d2) -> (d0, d1, d2)>],
      iterator_types = ["parallel", "parallel", "parallel"]}
      ins(%matmul : tensor<128x32xf32>) outs(%result_empty : tensor<128x32x64xf32>) {
  ^bb0(%value: f32, %out: f32):
    %scaled = arith.mulf %value, %value : f32
    linalg.yield %scaled : f32
  } -> tensor<128x32x64xf32>
  return %result : tensor<128x32x64xf32>
}

// d0/d1 are shared with the root and stay Common. d2 exists only on the
// consumer, so it is the only dimension at the Inner level, and it is zero in
// Common: no dimension may appear at two vector levels or getVectorSizes()
// returns nullopt and both TileToVectorSize and the configured-vector-size
// path are silently skipped.
// CHECK-DAG: #[[FILL:.+]] = #iree_cpu.lowering_config<vector_common_parallel = [8, 32]>
// CHECK-DAG: #[[ROOT:.+]] = #iree_cpu.lowering_config<cache_parallel = [64, 32, 0], distribution = [0, 0, 0], vector_common_parallel = [8, 32, 0], vector_reduction = [0, 0, 8]>
// CHECK-DAG: #[[CONSUMER:.+]] = #iree_cpu.lowering_config<vector_common_parallel = [1, 1, 0], vector_inner_parallel = [0, 0, 32]>
// CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = #iree_cpu.pipeline<DoubleTilingExpert>, {enable_loop_peeling}>
// CHECK: func.func @broadcast_consumer_after_root(
// CHECK-SAME: translation_info = #[[TRANSLATION]]
// CHECK: linalg.fill {lowering_config = #[[FILL]]}
// CHECK: linalg.matmul
// CHECK-SAME: lowering_config = #[[ROOT]]
// CHECK: linalg.generic
// CHECK-SAME: lowering_config = #[[CONSUMER]]

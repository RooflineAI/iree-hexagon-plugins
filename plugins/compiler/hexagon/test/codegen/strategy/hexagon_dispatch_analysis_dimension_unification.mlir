// Two operations may share one tensor.empty as their destination - CSE makes
// that happen for any two results of equal type. That is an allocation, not a
// shared value, so it must not unify their iteration dimensions.
//
// Here the LHS producer's innermost dimension is the root's k and the fill's is
// the root's n. Merging them would collapse the root's own n and k into one
// unified dimension, and the producer would then be bounded by n's tile of 1
// instead of k's tile of 32 - scalarizing its innermost dimension.
//
// RUN: iree-opt --iree-load-plugin=hexagon=$ROOF_HEXAGON_COMPILER_PLUGIN \
// RUN:   --iree-hexagon-enable-vtcm-tiling=false \
// RUN:   --pass-pipeline='builtin.module(iree-hexagon-select-lowering-strategy)' \
// RUN:   %s | FileCheck %s

#target = #hal.executable.target<"hexagon", "embedded-elf-hexagon", {cpu = "hexagonv79", cpu_features = "+hvxv79,+hvx-length128b", max_stack_allocation_size = 16384 : i64, target_triple = "hexagon-unknown-unknown-elf"}>
#identity3 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#lhs_map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
#rhs_transposed = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3)>
#out_map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>

func.func @shared_init_between_producer_and_fill(%q: tensor<16x128x128xf32>, %rhs: tensor<16x128x128xf32>) -> tensor<16x128x128xf32> attributes {hal.executable.target = #target} {
  %zero = arith.constant 0.000000e+00 : f32
  // One allocation, used as the destination of both the producer and the fill.
  %shared_empty = tensor.empty() : tensor<16x128x128xf32>
  %lhs = linalg.generic {indexing_maps = [#identity3, #identity3], iterator_types = ["parallel", "parallel", "parallel"]} ins(%q : tensor<16x128x128xf32>) outs(%shared_empty : tensor<16x128x128xf32>) {
  ^bb0(%in: f32, %out: f32):
    %doubled = arith.addf %in, %in : f32
    linalg.yield %doubled : f32
  } -> tensor<16x128x128xf32>
  %init = linalg.fill ins(%zero : f32) outs(%shared_empty : tensor<16x128x128xf32>) -> tensor<16x128x128xf32>
  %result = linalg.batch_matmul indexing_maps = [#lhs_map, #rhs_transposed, #out_map]
      ins(%lhs, %rhs : tensor<16x128x128xf32>, tensor<16x128x128xf32>)
      outs(%init : tensor<16x128x128xf32>) -> tensor<16x128x128xf32>
  return %result : tensor<16x128x128xf32>
}

// CHECK-DAG: #[[ROOT:.+]] = #iree_cpu.lowering_config<cache_parallel = [1, 64, 64, 0], distribution = [0, 0, 0, 0], vector_common_parallel = [1, 1, 1, 0], vector_reduction = [0, 0, 0, 32]>
// The producer's innermost dimension is k, so it keeps the root's 32 there.
// CHECK-DAG: #[[PRODUCER:.+]] = #iree_cpu.lowering_config<vector_common_parallel = [1, 1, 32]>
// The fill's innermost dimension is n, whose tile really is 1.
// CHECK-DAG: #[[FILL:.+]] = #iree_cpu.lowering_config<vector_common_parallel = [1, 1, 1]>
// CHECK: func.func @shared_init_between_producer_and_fill(
// CHECK: linalg.generic
// CHECK-SAME: lowering_config = #[[PRODUCER]]
// CHECK: linalg.fill {lowering_config = #[[FILL]]}
// CHECK: linalg.batch_matmul
// CHECK-SAME: lowering_config = #[[ROOT]]

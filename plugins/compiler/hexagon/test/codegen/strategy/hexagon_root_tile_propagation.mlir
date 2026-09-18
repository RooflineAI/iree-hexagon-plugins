// Test written to reflect current Hexagon selector policy.
//
// This file exercises the non-VTCM, non-HMX strategy policies. Dedicated tests
// cover those opt-in paths below the planner boundary.
//
// RUN: iree-opt --iree-load-plugin=hexagon=$ROOF_HEXAGON_COMPILER_PLUGIN \
// RUN:   --iree-hexagon-enable-vtcm-tiling=false \
// RUN:   --iree-hexagon-enable-hmx-matmul=false \
// RUN:   --pass-pipeline='builtin.module(iree-hexagon-select-lowering-strategy)' \
// RUN:   --split-input-file %s | FileCheck %s

#target = #hal.executable.target<"hexagon", "embedded-elf-hexagon", {cpu = "hexagonv79", cpu_features = "+hvxv79,+hvx-length128b", data_layout = "e-m:e-p:32:32:32-a:0-n16:32-i64:64:64-i32:32:32-i16:16:16-i1:8:8-f32:32:32-f64:64:64-v32:32:32-v64:64:64-v512:512:512-v1024:1024:1024-v2048:2048:2048", hexagon.version = "79", iree.encoding.resolver = #iree_hexagon.hexagon_encoding_resolver<>, link_embedded = false, max_stack_allocation_size = 16384 : i64, native_vector_size = 32 : i64, target_triple = "hexagon-unknown-unknown-elf"}>
func.func @matmul_epilogue_dispatch(%lhs: tensor<128x128xf32>, %rhs: tensor<128x128xf32>, %bias: tensor<128x128xf32>) -> tensor<128x128xf32> attributes {hal.executable.target = #target} {
  %cst = arith.constant 0.0 : f32
  %empty = tensor.empty() : tensor<128x128xf32>
  %init = linalg.fill ins(%cst : f32) outs(%empty : tensor<128x128xf32>) -> tensor<128x128xf32>
  %matmul = linalg.matmul ins(%lhs, %rhs : tensor<128x128xf32>, tensor<128x128xf32>) outs(%init : tensor<128x128xf32>) -> tensor<128x128xf32>
  %epilogue_empty = tensor.empty() : tensor<128x128xf32>
  %epilogue_init = linalg.fill ins(%cst : f32) outs(%epilogue_empty : tensor<128x128xf32>) -> tensor<128x128xf32>
  %result = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%matmul, %bias : tensor<128x128xf32>, tensor<128x128xf32>) outs(%epilogue_init : tensor<128x128xf32>) {
  ^bb0(%acc: f32, %bias_in: f32, %out0: f32):
    %sum = arith.addf %acc, %bias_in : f32
    linalg.yield %sum : f32
  } -> tensor<128x128xf32>
  return %result : tensor<128x128xf32>
}
// The fill takes the root's fusion tile, while the epilogue keeps its own
// narrower shape: TileToVectorSize can still tile the epilogue down to it, so
// only the divisibility bound applies there.

// CHECK-DAG: #[[FILL:.+]] = #iree_cpu.lowering_config<vector_common_parallel = [8, 32]>
// CHECK-DAG: #[[MATMUL:.+]] = #iree_cpu.lowering_config<cache_parallel = [64, 64, 0], distribution = [0, 0, 0], vector_common_parallel = [8, 32, 0], vector_reduction = [0, 0, 8]>
// CHECK-DAG: #[[EPILOGUE:.+]] = #iree_cpu.lowering_config<vector_common_parallel = [1, 32]>
// CHECK-NOT: #iree_cpu.lowering_config
// CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = #iree_cpu.pipeline<DoubleTilingExpert>, {enable_loop_peeling}>
// CHECK: func.func @matmul_epilogue_dispatch(
// CHECK-SAME: translation_info = #[[TRANSLATION]]
// CHECK: linalg.fill {lowering_config = #[[FILL]]}
// CHECK: linalg.matmul {lowering_config = #[[MATMUL]]}
// CHECK: linalg.generic
// CHECK-SAME: iterator_types = ["parallel", "parallel"]
// CHECK-SAME: lowering_config = #[[EPILOGUE]]

// -----

// Transposed-RHS batch_matmul with producers. Producer plans are selected from
// their own shapes and do not inherit the root contraction tile.
#map_lhs = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
#map_rhs_t = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3)>
#map_out = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
#producer_map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#target = #hal.executable.target<"hexagon", "embedded-elf-hexagon", {cpu = "hexagonv79", cpu_features = "+hvxv79,+hvx-length128b", data_layout = "e-m:e-p:32:32:32-a:0-n16:32-i64:64:64-i32:32:32-i16:16:16-i1:8:8-f32:32:32-f64:64:64-v32:32:32-v64:64:64-v512:512:512-v1024:1024:1024-v2048:2048:2048", hexagon.version = "79", iree.encoding.resolver = #iree_hexagon.hexagon_encoding_resolver<>, link_embedded = false, max_stack_allocation_size = 16384 : i64, native_vector_size = 32 : i64, target_triple = "hexagon-unknown-unknown-elf"}>
func.func @batch_matmul_transposed_rhs_with_producers_dispatch(%lhs: tensor<4x64x128xf32>, %rhs: tensor<4x64x128xf32>) -> tensor<4x64x64xf32> attributes {hal.executable.target = #target} {
  %cst = arith.constant 0.0 : f32
  %lhs_empty = tensor.empty() : tensor<4x64x128xf32>
  %rhs_empty = tensor.empty() : tensor<4x64x128xf32>
  %lhs_producer = linalg.generic {indexing_maps = [#producer_map, #producer_map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%lhs : tensor<4x64x128xf32>) outs(%lhs_empty : tensor<4x64x128xf32>) {
  ^bb0(%in: f32, %out0: f32):
    linalg.yield %in : f32
  } -> tensor<4x64x128xf32>
  %rhs_producer = linalg.generic {indexing_maps = [#producer_map, #producer_map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%rhs : tensor<4x64x128xf32>) outs(%rhs_empty : tensor<4x64x128xf32>) {
  ^bb0(%in: f32, %out0: f32):
    linalg.yield %in : f32
  } -> tensor<4x64x128xf32>
  %empty = tensor.empty() : tensor<4x64x64xf32>
  %init = linalg.fill ins(%cst : f32) outs(%empty : tensor<4x64x64xf32>) -> tensor<4x64x64xf32>
  %result = linalg.batch_matmul indexing_maps = [#map_lhs, #map_rhs_t, #map_out]
      ins(%lhs_producer, %rhs_producer : tensor<4x64x128xf32>, tensor<4x64x128xf32>)
      outs(%init : tensor<4x64x64xf32>) -> tensor<4x64x64xf32>
  return %result : tensor<4x64x64xf32>
}

// The producers keep their own 32-wide tile: they map onto the root's
// reduction dimension, which is not part of the root-anchored fusion loop, so
// reconciliation does not bound them. The fill maps onto the root's parallel
// dimensions and is matched to them.

// CHECK-DAG: #[[ATTN_PRODUCER_CONFIG:.+]] = #iree_cpu.lowering_config<vector_common_parallel = [1, 1, 32]>
// CHECK-DAG: #[[ATTN_FILL_CONFIG:.+]] = #iree_cpu.lowering_config<vector_common_parallel = [1, 1, 1]>
// CHECK-DAG: #[[ATTN_ROOT_CONFIG:.+]] = #iree_cpu.lowering_config<cache_parallel = [1, 64, 64, 0], distribution = [0, 0, 0, 0], vector_common_parallel = [1, 1, 1, 0], vector_reduction = [0, 0, 0, 32]>
// CHECK-NOT: #iree_cpu.lowering_config
// CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = #iree_cpu.pipeline<DoubleTilingExpert>, {enable_loop_peeling}>
// CHECK: func.func @batch_matmul_transposed_rhs_with_producers_dispatch(
// CHECK-SAME: translation_info = #[[TRANSLATION]]
// CHECK: linalg.generic
// CHECK-SAME: iterator_types = ["parallel", "parallel", "parallel"]
// CHECK-SAME: lowering_config = #[[ATTN_PRODUCER_CONFIG]]
// CHECK: linalg.generic
// CHECK-SAME: iterator_types = ["parallel", "parallel", "parallel"]
// CHECK-SAME: lowering_config = #[[ATTN_PRODUCER_CONFIG]]
// CHECK: linalg.fill {lowering_config = #[[ATTN_FILL_CONFIG]]}
// CHECK: linalg.batch_matmul
// CHECK-SAME: lowering_config = #[[ATTN_ROOT_CONFIG]]

// -----

// Elementwise conversions feeding a matmul. Both have the same loop structure
// and both select [1, 32] on their own, but they are bounded differently: the
// LHS conversion's innermost dimension is the root's reduction dimension, and
// TileRootAndFuseInputOperands fuses input-operand producers into that loop, so
// it is bounded to the root's k tile of 8. The RHS conversion's innermost
// dimension is the root's n dimension, whose Common tile is already 32.

#target = #hal.executable.target<"hexagon", "embedded-elf-hexagon", {cpu = "hexagonv79", cpu_features = "+hvxv79,+hvx-length128b", max_stack_allocation_size = 16384 : i64, target_triple = "hexagon-unknown-unknown-elf"}>
#identity2 = affine_map<(d0, d1) -> (d0, d1)>

func.func @matmul_with_converted_inputs(%lhs: tensor<128x256xf16>, %rhs: tensor<256x64xf16>) -> tensor<128x64xf32> attributes {hal.executable.target = #target} {
  %zero = arith.constant 0.0 : f32
  %lhs_empty = tensor.empty() : tensor<128x256xf32>
  %lhs_f32 = linalg.generic {indexing_maps = [#identity2, #identity2], iterator_types = ["parallel", "parallel"]} ins(%lhs : tensor<128x256xf16>) outs(%lhs_empty : tensor<128x256xf32>) {
  ^bb0(%in: f16, %out: f32):
    %extended = arith.extf %in : f16 to f32
    linalg.yield %extended : f32
  } -> tensor<128x256xf32>
  %rhs_empty = tensor.empty() : tensor<256x64xf32>
  %rhs_f32 = linalg.generic {indexing_maps = [#identity2, #identity2], iterator_types = ["parallel", "parallel"]} ins(%rhs : tensor<256x64xf16>) outs(%rhs_empty : tensor<256x64xf32>) {
  ^bb0(%in: f16, %out: f32):
    %extended = arith.extf %in : f16 to f32
    linalg.yield %extended : f32
  } -> tensor<256x64xf32>
  %empty = tensor.empty() : tensor<128x64xf32>
  %init = linalg.fill ins(%zero : f32) outs(%empty : tensor<128x64xf32>) -> tensor<128x64xf32>
  %result = linalg.matmul ins(%lhs_f32, %rhs_f32 : tensor<128x256xf32>, tensor<256x64xf32>) outs(%init : tensor<128x64xf32>) -> tensor<128x64xf32>
  return %result : tensor<128x64xf32>
}
// CHECK-DAG: #[[LHS_CONVERT:.+]] = #iree_cpu.lowering_config<vector_common_parallel = [1, 8]>
// CHECK-DAG: #[[RHS_CONVERT:.+]] = #iree_cpu.lowering_config<vector_common_parallel = [1, 32]>
// CHECK-DAG: #[[CONVERT_FILL:.+]] = #iree_cpu.lowering_config<vector_common_parallel = [8, 32]>
// CHECK-DAG: #[[CONVERT_ROOT:.+]] = #iree_cpu.lowering_config<cache_parallel = [64, 64, 0], distribution = [0, 0, 0], vector_common_parallel = [8, 32, 0], vector_reduction = [0, 0, 8]>
// CHECK: func.func @matmul_with_converted_inputs(
// CHECK: linalg.generic
// CHECK-SAME: lowering_config = #[[LHS_CONVERT]]
// CHECK: linalg.generic
// CHECK-SAME: lowering_config = #[[RHS_CONVERT]]
// CHECK: linalg.fill {lowering_config = #[[CONVERT_FILL]]}
// CHECK: linalg.matmul
// CHECK-SAME: lowering_config = #[[CONVERT_ROOT]]

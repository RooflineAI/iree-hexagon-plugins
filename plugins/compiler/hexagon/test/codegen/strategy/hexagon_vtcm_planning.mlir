// RUN: iree-opt \
// RUN:   --iree-hexagon-enable-vtcm-tiling \
// RUN:   --pass-pipeline='builtin.module(iree-hexagon-select-lowering-strategy)' \
// RUN:   --split-input-file %s | FileCheck %s

#target = #hal.executable.target<"hexagon", "embedded-elf-hexagon", {cpu = "hexagonv79", cpu_features = "+hvxv79,+hvx-length128b", data_layout = "e-m:e-p:32:32:32-a:0-n16:32-i64:64:64-i32:32:32-i16:16:16-i1:8:8-f32:32:32-f64:64:64-v32:32:32-v64:64:64-v512:512:512-v1024:1024:1024-v2048:2048:2048", hexagon.version = "79", iree.encoding.resolver = #iree_hexagon.hexagon_encoding_resolver<>, link_embedded = false, max_stack_allocation_size = 16384 : i64, native_vector_size = 32 : i64, target_triple = "hexagon-unknown-unknown-elf"}>
func.func @generic_dispatch(%src: tensor<4x128x128xf32>) -> tensor<4x128xf32> attributes {hal.executable.target = #target} {
  %cst = arith.constant 0.0 : f32
  %empty = tensor.empty() : tensor<4x128xf32>
  %init = linalg.fill ins(%cst : f32) outs(%empty : tensor<4x128xf32>) -> tensor<4x128xf32>
  %result = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%src : tensor<4x128x128xf32>) outs(%init : tensor<4x128xf32>) {
  ^bb0(%in: f32, %out0: f32):
    %sum = arith.addf %in, %out0 : f32
    linalg.yield %sum : f32
  } -> tensor<4x128xf32>
  return %result : tensor<4x128xf32>
}
// CHECK-DAG: #[[FILL:.+]] = #iree_cpu.lowering_config<vector_common_parallel = [1, 1]>
// CHECK-DAG: #[[ROOT:.+]] = #iree_cpu.lowering_config<distribution = [0, 0, 0], vector_common_parallel = [1, 1, 0], vector_reduction = [0, 0, 32]>
// CHECK-NOT: #iree_cpu.lowering_config
// CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = #iree_cpu.pipeline<DoubleTilingExpert>, {enable_loop_peeling}>
// CHECK-DAG: #[[VTCM:.+]] = #iree_hexagon.vtcm_tiling_config<tile_sizes = [4, 128, 128]>
// CHECK: func.func @generic_dispatch(
// CHECK-SAME: translation_info = #[[TRANSLATION]]
// CHECK: linalg.fill {lowering_config = #[[FILL]]}
// CHECK: linalg.generic
// CHECK-SAME: hexagon_vtcm_tiling_config = #[[VTCM]]
// CHECK-SAME: lowering_config = #[[ROOT]]

// -----

#target = #hal.executable.target<"hexagon", "embedded-elf-hexagon", {cpu = "hexagonv79", cpu_features = "+hvxv79,+hvx-length128b", data_layout = "e-m:e-p:32:32:32-a:0-n16:32-i64:64:64-i32:32:32-i16:16:16-i1:8:8-f32:32:32-f64:64:64-v32:32:32-v64:64:64-v512:512:512-v1024:1024:1024-v2048:2048:2048", hexagon.version = "79", iree.encoding.resolver = #iree_hexagon.hexagon_encoding_resolver<>, link_embedded = false, max_stack_allocation_size = 16384 : i64, native_vector_size = 32 : i64, target_triple = "hexagon-unknown-unknown-elf"}>
func.func @matmul_dispatch(%lhs: tensor<128x128xf32>, %rhs: tensor<128x128xf32>) -> tensor<128x128xf32> attributes {hal.executable.target = #target} {
  %cst = arith.constant 0.0 : f32
  %empty = tensor.empty() : tensor<128x128xf32>
  %init = linalg.fill ins(%cst : f32) outs(%empty : tensor<128x128xf32>) -> tensor<128x128xf32>
  %result = linalg.matmul ins(%lhs, %rhs : tensor<128x128xf32>, tensor<128x128xf32>) outs(%init : tensor<128x128xf32>) -> tensor<128x128xf32>
  return %result : tensor<128x128xf32>
}
// CHECK-DAG: #[[FILL:.+]] = #iree_cpu.lowering_config<vector_common_parallel = [8, 32]>
// CHECK-DAG: #[[ROOT:.+]] = #iree_cpu.lowering_config<distribution = [0, 0, 0], vector_common_parallel = [8, 32, 0], vector_reduction = [0, 0, 8]>
// CHECK-NOT: #iree_cpu.lowering_config
// CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = #iree_cpu.pipeline<DoubleTilingExpert>, {enable_loop_peeling}>
// CHECK-DAG: #[[VTCM:.+]] = #iree_hexagon.vtcm_tiling_config<tile_sizes = [128, 128, 128]>
// CHECK: func.func @matmul_dispatch(
// CHECK-SAME: translation_info = #[[TRANSLATION]]
// CHECK: linalg.fill {lowering_config = #[[FILL]]}
// CHECK: linalg.matmul
// CHECK-SAME: hexagon_vtcm_tiling_config = #[[VTCM]]
// CHECK-SAME: lowering_config = #[[ROOT]]

// -----

// Two fused elementwise ops: only the root (last) op carries the dispatch-wide
// VTCM tiling config; the producer must not.
#target = #hal.executable.target<"hexagon", "embedded-elf-hexagon", {cpu = "hexagonv79", cpu_features = "+hvxv79,+hvx-length128b", data_layout = "e-m:e-p:32:32:32-a:0-n16:32-i64:64:64-i32:32:32-i16:16:16-i1:8:8-f32:32:32-f64:64:64-v32:32:32-v64:64:64-v512:512:512-v1024:1024:1024-v2048:2048:2048", hexagon.version = "79", iree.encoding.resolver = #iree_hexagon.hexagon_encoding_resolver<>, link_embedded = false, max_stack_allocation_size = 16384 : i64, native_vector_size = 32 : i64, target_triple = "hexagon-unknown-unknown-elf"}>
func.func @multi_op_dispatch(%src: tensor<8x8xf32>) -> tensor<8x8xf32> attributes {hal.executable.target = #target} {
  %cst = arith.constant 1.0 : f32
  %empty = tensor.empty() : tensor<8x8xf32>
  %init = linalg.fill ins(%cst : f32) outs(%empty : tensor<8x8xf32>) -> tensor<8x8xf32>
  %producer = linalg.generic {indexing_maps = [affine_map<(i, j) -> (i, j)>, affine_map<(i, j) -> (i, j)>], iterator_types = ["parallel", "parallel"]} ins(%src : tensor<8x8xf32>) outs(%init : tensor<8x8xf32>) {
  ^bb0(%in: f32, %out0: f32):
    %sum = arith.addf %in, %out0 : f32
    linalg.yield %sum : f32
  } -> tensor<8x8xf32>
  %result = linalg.generic {indexing_maps = [affine_map<(i, j) -> (i, j)>, affine_map<(i, j) -> (i, j)>], iterator_types = ["parallel", "parallel"]} ins(%producer : tensor<8x8xf32>) outs(%init : tensor<8x8xf32>) {
  ^bb0(%in: f32, %out0: f32):
    %sum = arith.addf %in, %out0 : f32
    linalg.yield %sum : f32
  } -> tensor<8x8xf32>
  return %result : tensor<8x8xf32>
}
// CHECK-DAG: #[[PRODUCER:.+]] = #iree_cpu.lowering_config<vector_common_parallel = [1, 8]>
// CHECK-DAG: #[[ROOT:.+]] = #iree_cpu.lowering_config<distribution = [0, 0], vector_common_parallel = [1, 8]>
// CHECK-NOT: #iree_cpu.lowering_config
// CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = #iree_cpu.pipeline<DoubleTilingExpert>, {enable_loop_peeling}>
// CHECK-DAG: #[[VTCM:.+]] = #iree_hexagon.vtcm_tiling_config<tile_sizes = [8, 8]>
// CHECK: func.func @multi_op_dispatch(
// CHECK-SAME: translation_info = #[[TRANSLATION]]
// CHECK: linalg.generic
// CHECK-SAME: lowering_config = #[[PRODUCER]]
// CHECK-NOT: hexagon_vtcm_tiling_config
// CHECK: linalg.generic
// CHECK-SAME: hexagon_vtcm_tiling_config = #[[VTCM]]
// CHECK-SAME: lowering_config = #[[ROOT]]

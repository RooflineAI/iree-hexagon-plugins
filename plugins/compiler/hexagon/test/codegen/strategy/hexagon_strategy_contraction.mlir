// Test written to reflect current Hexagon selector policy.
//
// This file exercises the non-VTCM, non-HMX strategy policies.
//
// RUN: iree-opt --iree-load-plugin=hexagon=$ROOF_HEXAGON_COMPILER_PLUGIN \
// RUN:   --iree-hexagon-enable-vtcm-tiling=false \
// RUN:   --pass-pipeline='builtin.module(iree-hexagon-select-lowering-strategy)' \
// RUN:   --split-input-file %s | FileCheck %s
#target = #hal.executable.target<"hexagon", "embedded-elf-hexagon", {cpu = "hexagonv79", cpu_features = "+hvxv79,+hvx-length128b", data_layout = "e-m:e-p:32:32:32-a:0-n16:32-i64:64:64-i32:32:32-i16:16:16-i1:8:8-f32:32:32-f64:64:64-v32:32:32-v64:64:64-v512:512:512-v1024:1024:1024-v2048:2048:2048", hexagon.version = "79", iree.encoding.resolver = #iree_hexagon.hexagon_encoding_resolver<>, link_embedded = false, max_stack_allocation_size = 16384 : i64, native_vector_size = 32 : i64, target_triple = "hexagon-unknown-unknown-elf"}>
func.func @matmul_dispatch(%lhs: tensor<128x128xf32>, %rhs: tensor<128x128xf32>) -> tensor<128x128xf32> attributes {hal.executable.target = #target} {
  %cst = arith.constant 0.0 : f32
  %empty = tensor.empty() : tensor<128x128xf32>
  %init = linalg.fill ins(%cst : f32) outs(%empty : tensor<128x128xf32>) -> tensor<128x128xf32>
  %result = linalg.matmul ins(%lhs, %rhs : tensor<128x128xf32>, tensor<128x128xf32>) outs(%init : tensor<128x128xf32>) -> tensor<128x128xf32>
  return %result : tensor<128x128xf32>
}
// The fill is fused into the root's 8x32 tile and cannot be refined further by
// TileToVectorSize, so it takes the root's fusion tile exactly.

// CHECK-DAG: #[[FILL:.+]] = #iree_cpu.lowering_config<vector_common_parallel = [8, 32]>
// CHECK-DAG: #[[MATMUL:.+]] = #iree_cpu.lowering_config<cache_parallel = [64, 64, 0], distribution = [0, 0, 0], vector_common_parallel = [8, 32, 0], vector_reduction = [0, 0, 8]>
// CHECK-NOT: #iree_cpu.lowering_config
// CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = #iree_cpu.pipeline<DoubleTilingExpert>, {enable_loop_peeling}>
// CHECK: func.func @matmul_dispatch(
// CHECK-SAME: translation_info = #[[TRANSLATION]]
// CHECK: linalg.fill {lowering_config = #[[FILL]]}
// CHECK: linalg.matmul {lowering_config = #[[MATMUL]]}

// -----

// matmul with n=49 (not divisible by the HVX vector width of 32).
#target = #hal.executable.target<"hexagon", "embedded-elf-hexagon", {cpu = "hexagonv79", cpu_features = "+hvxv79,+hvx-length128b", data_layout = "e-m:e-p:32:32:32-a:0-n16:32-i64:64:64-i32:32:32-i16:16:16-i1:8:8-f32:32:32-f64:64:64-v32:32:32-v64:64:64-v512:512:512-v1024:1024:1024-v2048:2048:2048", hexagon.version = "79", iree.encoding.resolver = #iree_hexagon.hexagon_encoding_resolver<>, link_embedded = false, max_stack_allocation_size = 16384 : i64, native_vector_size = 32 : i64, target_triple = "hexagon-unknown-unknown-elf"}>
func.func @matmul_512x49x4608_dispatch(%lhs: tensor<512x4608xf32>, %rhs: tensor<4608x49xf32>) -> tensor<512x49xf32> attributes {hal.executable.target = #target} {
  %cst = arith.constant 0.0 : f32
  %empty = tensor.empty() : tensor<512x49xf32>
  %init = linalg.fill ins(%cst : f32) outs(%empty : tensor<512x49xf32>) -> tensor<512x49xf32>
  %result = linalg.matmul ins(%lhs, %rhs : tensor<512x4608xf32>, tensor<4608x49xf32>) outs(%init : tensor<512x49xf32>) -> tensor<512x49xf32>
  return %result : tensor<512x49xf32>
}
// CHECK-DAG: #[[FILL:.+]] = #iree_cpu.lowering_config<vector_common_parallel = [8, 32]>
// CHECK-DAG: #[[MATMUL:.+]] = #iree_cpu.lowering_config<cache_parallel = [64, 49, 0], distribution = [0, 0, 0], vector_common_parallel = [8, 32, 0], vector_reduction = [0, 0, 8]>
// CHECK-NOT: #iree_cpu.lowering_config
// CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = #iree_cpu.pipeline<DoubleTilingExpert>, {enable_loop_peeling}>
// CHECK: func.func @matmul_512x49x4608_dispatch(
// CHECK-SAME: translation_info = #[[TRANSLATION]]
// CHECK: linalg.fill {lowering_config = #[[FILL]]}
// CHECK: linalg.matmul {lowering_config = #[[MATMUL]]}

// -----

#target = #hal.executable.target<"hexagon", "embedded-elf-hexagon", {cpu = "hexagonv79", cpu_features = "+hvxv79,+hvx-length128b", data_layout = "e-m:e-p:32:32:32-a:0-n16:32-i64:64:64-i32:32:32-i16:16:16-i1:8:8-f32:32:32-f64:64:64-v32:32:32-v64:64:64-v512:512:512-v1024:1024:1024-v2048:2048:2048", hexagon.version = "79", iree.encoding.resolver = #iree_hexagon.hexagon_encoding_resolver<>, link_embedded = false, max_stack_allocation_size = 16384 : i64, native_vector_size = 32 : i64, target_triple = "hexagon-unknown-unknown-elf"}>
func.func @batch_matmul_dispatch(%lhs: tensor<4x128x128xf32>, %rhs: tensor<4x128x128xf32>) -> tensor<4x128x128xf32> attributes {hal.executable.target = #target} {
  %cst = arith.constant 0.0 : f32
  %empty = tensor.empty() : tensor<4x128x128xf32>
  %init = linalg.fill ins(%cst : f32) outs(%empty : tensor<4x128x128xf32>) -> tensor<4x128x128xf32>
  %result = linalg.batch_matmul ins(%lhs, %rhs : tensor<4x128x128xf32>, tensor<4x128x128xf32>) outs(%init : tensor<4x128x128xf32>) -> tensor<4x128x128xf32>
  return %result : tensor<4x128x128xf32>
}
// CHECK-DAG: #[[FILL:.+]] = #iree_cpu.lowering_config<vector_common_parallel = [1, 8, 32]>
// CHECK-DAG: #[[MATMUL:.+]] = #iree_cpu.lowering_config<cache_parallel = [1, 64, 64, 0], distribution = [0, 0, 0, 0], vector_common_parallel = [1, 8, 32, 0], vector_reduction = [0, 0, 0, 8]>
// CHECK-NOT: #iree_cpu.lowering_config
// CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = #iree_cpu.pipeline<DoubleTilingExpert>, {enable_loop_peeling}>
// CHECK: func.func @batch_matmul_dispatch(
// CHECK-SAME: translation_info = #[[TRANSLATION]]
// CHECK: linalg.fill {lowering_config = #[[FILL]]}
// CHECK: linalg.batch_matmul {lowering_config = #[[MATMUL]]}

// -----

// Unsupported contraction (dot): expect the CPUDefault fallback with no tiling
// selected on any op.
#target = #hal.executable.target<"hexagon", "embedded-elf-hexagon", {cpu = "hexagonv79", cpu_features = "+hvxv79,+hvx-length128b", data_layout = "e-m:e-p:32:32:32-a:0-n16:32-i64:64:64-i32:32:32-i16:16:16-i1:8:8-f32:32:32-f64:64:64-v32:32:32-v64:64:64-v512:512:512-v1024:1024:1024-v2048:2048:2048", hexagon.version = "79", iree.encoding.resolver = #iree_hexagon.hexagon_encoding_resolver<>, link_embedded = false, max_stack_allocation_size = 16384 : i64, native_vector_size = 32 : i64, target_triple = "hexagon-unknown-unknown-elf"}>
func.func @dot_dispatch(%lhs: tensor<128xf32>, %rhs: tensor<128xf32>) -> tensor<f32> attributes {hal.executable.target = #target} {
  %cst = arith.constant 0.0 : f32
  %empty = tensor.empty() : tensor<f32>
  %init = linalg.fill ins(%cst : f32) outs(%empty : tensor<f32>) -> tensor<f32>
  %result = linalg.dot ins(%lhs, %rhs : tensor<128xf32>, tensor<128xf32>) outs(%init : tensor<f32>) -> tensor<f32>
  return %result : tensor<f32>
}
// CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = #iree_cpu.pipeline<Default>>
// CHECK: func.func @dot_dispatch(
// CHECK-SAME: translation_info = #[[TRANSLATION]]
// CHECK: linalg.fill
// CHECK-NOT: lowering_config
// CHECK: linalg.dot
// CHECK-NOT: lowering_config
// CHECK: return

// -----

// Transposed-RHS batch_matmul: RHS layout is B[b, n, k]
// This is the shape of attention's Q*K^T dispatch (b=4, m=1024, n=1024, k=128).
#map_lhs = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
#map_rhs_t = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3)>
#map_out = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
#target = #hal.executable.target<"hexagon", "embedded-elf-hexagon", {cpu = "hexagonv79", cpu_features = "+hvxv79,+hvx-length128b", data_layout = "e-m:e-p:32:32:32-a:0-n16:32-i64:64:64-i32:32:32-i16:16:16-i1:8:8-f32:32:32-f64:64:64-v32:32:32-v64:64:64-v512:512:512-v1024:1024:1024-v2048:2048:2048", hexagon.version = "79", iree.encoding.resolver = #iree_hexagon.hexagon_encoding_resolver<>, link_embedded = false, max_stack_allocation_size = 16384 : i64, native_vector_size = 32 : i64, target_triple = "hexagon-unknown-unknown-elf"}>
func.func @batch_matmul_transposed_rhs_dispatch(%lhs: tensor<4x64x128xf32>, %rhs: tensor<4x64x128xf32>) -> tensor<4x64x64xf32> attributes {hal.executable.target = #target} {
  %cst = arith.constant 0.0 : f32
  %empty = tensor.empty() : tensor<4x64x64xf32>
  %init = linalg.fill ins(%cst : f32) outs(%empty : tensor<4x64x64xf32>) -> tensor<4x64x64xf32>
  %result = linalg.batch_matmul indexing_maps = [#map_lhs, #map_rhs_t, #map_out]
      ins(%lhs, %rhs : tensor<4x64x128xf32>, tensor<4x64x128xf32>)
      outs(%init : tensor<4x64x64xf32>) -> tensor<4x64x64xf32>
  return %result : tensor<4x64x64xf32>
}
// CHECK-DAG: #[[FILL:.+]] = #iree_cpu.lowering_config<vector_common_parallel = [1, 1, 1]>
// CHECK-DAG: #[[MATMUL:.+]] = #iree_cpu.lowering_config<cache_parallel = [1, 64, 64, 0], distribution = [0, 0, 0, 0], vector_common_parallel = [1, 1, 1, 0], vector_reduction = [0, 0, 0, 32]>
// CHECK-NOT: #iree_cpu.lowering_config
// CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = #iree_cpu.pipeline<DoubleTilingExpert>, {enable_loop_peeling}>
// CHECK: func.func @batch_matmul_transposed_rhs_dispatch(
// CHECK-SAME: translation_info = #[[TRANSLATION]]
// CHECK: linalg.fill {lowering_config = #[[FILL]]}
// CHECK: linalg.batch_matmul {{.*}} {lowering_config = #[[MATMUL]]}

// Non-root compute tiles are selected from each operation's own shape. They do
// not inherit root tiles, cache tiles, or distribution tiles.
//
// RUN: iree-opt \
// RUN:   --iree-hexagon-enable-hmx-matmul \
// RUN:   --pass-pipeline='builtin.module(iree-hexagon-select-lowering-strategy)' \
// RUN:   --split-input-file %s | FileCheck %s


#target = #hal.executable.target<"hexagon", "embedded-elf-hexagon", {cpu = "hexagonv79", cpu_features = "+hvxv79,+hvx-length128b", data_layout = "e-m:e-p:32:32:32-a:0-n16:32-i64:64:64-i32:32:32-i16:16:16-i1:8:8-f32:32:32-f64:64:64-v32:32:32-v64:64:64-v512:512:512-v1024:1024:1024-v2048:2048:2048", hexagon.version = "79", iree.encoding.resolver = #iree_hexagon.hexagon_encoding_resolver<>, link_embedded = false, max_stack_allocation_size = 16384 : i64, native_vector_size = 32 : i64, target_triple = "hexagon-unknown-unknown-elf"}>

// HMX requires VTCM but preserves its root-only cache batch tile. The cache
// level is currently consumed before HMX packing to rank-reduce batch_matmul.
func.func @hmx_batch_matmul_preserves_batch_tile(%lhs: tensor<4x32x32xf16>, %rhs: tensor<4x32x32xf16>) -> tensor<4x32x32xf16> attributes {hal.executable.target = #target} {
  %cst = arith.constant 0.0 : f16
  %empty = tensor.empty() : tensor<4x32x32xf16>
  %init = linalg.fill ins(%cst : f16) outs(%empty : tensor<4x32x32xf16>) -> tensor<4x32x32xf16>
  %result = linalg.batch_matmul ins(%lhs, %rhs : tensor<4x32x32xf16>, tensor<4x32x32xf16>) outs(%init : tensor<4x32x32xf16>) -> tensor<4x32x32xf16>
  return %result : tensor<4x32x32xf16>
}
// CHECK-DAG: #[[BATCH_FILL:.+]] = #iree_cpu.lowering_config<vector_common_parallel = [1, 32, 32]>
// CHECK-DAG: #[[BATCH_ROOT:.+]] = #iree_cpu.lowering_config<cache_parallel = [1, 0, 0, 0], distribution = [0, 0, 0, 0], vector_common_parallel = [1, 32, 32, 0]>
// CHECK-DAG: #[[BATCH_TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = #iree_cpu.pipeline<Mmt4dTilingExpert>>
// CHECK-DAG: #[[BATCH_VTCM:.+]] = #iree_hexagon.vtcm_tiling_config
// CHECK: func.func @hmx_batch_matmul_preserves_batch_tile(
// CHECK-SAME: translation_info = #[[BATCH_TRANSLATION]]
// CHECK: linalg.fill {lowering_config = #[[BATCH_FILL]]}
// CHECK: linalg.batch_matmul
// CHECK-SAME: hexagon_vtcm_tiling_config = #[[BATCH_VTCM]]
// CHECK-SAME: lowering_config = #[[BATCH_ROOT]]

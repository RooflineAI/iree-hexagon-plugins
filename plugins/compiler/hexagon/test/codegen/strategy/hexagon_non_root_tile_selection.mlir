// Non-root compute tiles are selected from each operation's own shape. They do
// not inherit root tiles, cache tiles, or distribution tiles.
//
// RUN: iree-opt --iree-load-plugin=hexagon=$ROOF_HEXAGON_COMPILER_PLUGIN \
// RUN:   --iree-hexagon-enable-hmx-matmul \
// RUN:   --pass-pipeline='builtin.module(iree-hexagon-select-lowering-strategy)' \
// RUN:   --split-input-file %s | FileCheck %s

#target = #hal.executable.target<"hexagon", "embedded-elf-hexagon", {cpu = "hexagonv79", cpu_features = "+hvxv79,+hvx-length128b", data_layout = "e-m:e-p:32:32:32-a:0-n16:32-i64:64:64-i32:32:32-i16:16:16-i1:8:8-f32:32:32-f64:64:64-v32:32:32-v64:64:64-v512:512:512-v1024:1024:1024-v2048:2048:2048", hexagon.version = "79", iree.encoding.resolver = #iree_hexagon.hexagon_encoding_resolver<>, link_embedded = false, max_stack_allocation_size = 16384 : i64, native_vector_size = 32 : i64, target_triple = "hexagon-unknown-unknown-elf"}>

// The hardware-fixed HMX root remains 32x32. The epilogue selects [1, 64] for
// f16 independently and is then bounded to [1, 32] by the root's fusion tile.
// The fill is matched to the fusion tile outright: TileToVectorSize skips fills,
// so a narrower shape would never be realized.
func.func @hmx_matmul_bias_consumer(%lhs: tensor<128x128xf16>, %rhs: tensor<128x128xf16>, %bias: tensor<128x128xf16>) -> tensor<128x128xf16> attributes {hal.executable.target = #target} {
  %cst = arith.constant 0.0 : f16
  %empty = tensor.empty() : tensor<128x128xf16>
  %init = linalg.fill ins(%cst : f16) outs(%empty : tensor<128x128xf16>) -> tensor<128x128xf16>
  %matmul = linalg.matmul ins(%lhs, %rhs : tensor<128x128xf16>, tensor<128x128xf16>) outs(%init : tensor<128x128xf16>) -> tensor<128x128xf16>
  %result_empty = tensor.empty() : tensor<128x128xf16>
  %result = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]}
      ins(%matmul, %bias : tensor<128x128xf16>, tensor<128x128xf16>)
      outs(%result_empty : tensor<128x128xf16>) {
  ^bb0(%value: f16, %bias_value: f16, %out: f16):
    %sum = arith.addf %value, %bias_value : f16
    linalg.yield %sum : f16
  } -> tensor<128x128xf16>
  return %result : tensor<128x128xf16>
}
// CHECK-DAG: #[[FILL:.+]] = #iree_cpu.lowering_config<vector_common_parallel = [32, 32]>
// CHECK-DAG: #[[EPILOGUE:.+]] = #iree_cpu.lowering_config<vector_common_parallel = [1, 32]>
// CHECK-DAG: #[[ROOT:.+]] = #iree_cpu.lowering_config<distribution = [0, 0, 0], vector_common_parallel = [32, 32, 0]>
// CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = #iree_cpu.pipeline<Mmt4dTilingExpert>>
// CHECK-DAG: #[[VTCM:.+]] = #iree_hexagon.vtcm_tiling_config<tile_sizes = [128, 128, 128]>
// CHECK: func.func @hmx_matmul_bias_consumer(
// CHECK-SAME: translation_info = #[[TRANSLATION]]
// CHECK: linalg.fill {lowering_config = #[[FILL]]}
// CHECK: linalg.matmul
// CHECK-SAME: hexagon_vtcm_tiling_config = #[[VTCM]]
// CHECK-SAME: lowering_config = #[[ROOT]]
// CHECK: linalg.generic
// CHECK-SAME: lowering_config = #[[EPILOGUE]]

// -----

#target = #hal.executable.target<"hexagon", "embedded-elf-hexagon", {cpu = "hexagonv79", cpu_features = "+hvxv79,+hvx-length128b", data_layout = "e-m:e-p:32:32:32-a:0-n16:32-i64:64:64-i32:32:32-i16:16:16-i1:8:8-f32:32:32-f64:64:64-v32:32:32-v64:64:64-v512:512:512-v1024:1024:1024-v2048:2048:2048", hexagon.version = "79", iree.encoding.resolver = #iree_hexagon.hexagon_encoding_resolver<>, link_embedded = false, max_stack_allocation_size = 16384 : i64, native_vector_size = 32 : i64, target_triple = "hexagon-unknown-unknown-elf"}>

// d2 is private to the consumer. Encoding therefore assigns its independently
// selected 64-wide tile to Inner while d0/d1 remain Common. The exact config
// also proves that no dimension occurs at both vector levels.
func.func @hmx_matmul_broadcast_consumer(%lhs: tensor<128x32xf16>, %rhs: tensor<32x32xf16>) -> tensor<128x32x64xf16> attributes {hal.executable.target = #target} {
  %cst = arith.constant 0.0 : f16
  %empty = tensor.empty() : tensor<128x32xf16>
  %init = linalg.fill ins(%cst : f16) outs(%empty : tensor<128x32xf16>) -> tensor<128x32xf16>
  %matmul = linalg.matmul ins(%lhs, %rhs : tensor<128x32xf16>, tensor<32x32xf16>) outs(%init : tensor<128x32xf16>) -> tensor<128x32xf16>
  %result_empty = tensor.empty() : tensor<128x32x64xf16>
  %result = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>],
      iterator_types = ["parallel", "parallel", "parallel"]}
      ins(%matmul : tensor<128x32xf16>) outs(%result_empty : tensor<128x32x64xf16>) {
  ^bb0(%value: f16, %out: f16):
    linalg.yield %value : f16
  } -> tensor<128x32x64xf16>
  return %result : tensor<128x32x64xf16>
}
// CHECK-DAG: #[[FILL:.+]] = #iree_cpu.lowering_config<vector_common_parallel = [32, 32]>
// CHECK-DAG: #[[ROOT:.+]] = #iree_cpu.lowering_config<distribution = [0, 0, 0], vector_common_parallel = [32, 32, 0]>
// CHECK-DAG: #[[CONSUMER:.+]] = #iree_cpu.lowering_config<vector_common_parallel = [1, 1, 0], vector_inner_parallel = [0, 0, 64]>
// CHECK-NOT: cache_parallel
// CHECK: func.func @hmx_matmul_broadcast_consumer(
// CHECK: linalg.fill {lowering_config = #[[FILL]]}
// CHECK: linalg.matmul
// CHECK-SAME: lowering_config = #[[ROOT]]
// CHECK: linalg.generic
// CHECK-SAME: lowering_config = #[[CONSUMER]]

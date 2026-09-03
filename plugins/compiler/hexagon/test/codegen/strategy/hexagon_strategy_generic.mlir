// Test written to reflect current Hexagon selector policy.
//
// This file exercises the non-VTCM, non-HMX strategy policies. Dedicated tests
// cover those opt-in paths below the planner boundary.
//
// RUN: iree-opt --iree-load-plugin=hexagon=$ROOF_HEXAGON_COMPILER_PLUGIN \
// RUN:   --iree-hexagon-enable-vtcm-tiling=false \
// RUN:   --pass-pipeline='builtin.module(iree-hexagon-select-lowering-strategy)' \
// RUN:   --split-input-file %s | FileCheck %s

#target = #hal.executable.target<"hexagon", "embedded-elf-hexagon", {cpu = "hexagonv79", cpu_features = "+hvxv79,+hvx-length128b", data_layout = "e-m:e-p:32:32:32-a:0-n16:32-i64:64:64-i32:32:32-i16:16:16-i1:8:8-f32:32:32-f64:64:64-v32:32:32-v64:64:64-v512:512:512-v1024:1024:1024-v2048:2048:2048", hexagon.version = "79", iree.encoding.resolver = #iree_hexagon.hexagon_encoding_resolver<>, link_embedded = false, max_stack_allocation_size = 16384 : i64, native_vector_size = 32 : i64, target_triple = "hexagon-unknown-unknown-elf"}>
func.func @generic_dispatch(%src: tensor<4x128x128xf32>) -> tensor<4x128xf32> attributes {hal.executable.target = #target} {
  %cst = arith.constant 0.0 : f32
  %red_empty = tensor.empty() : tensor<4x128xf32>
  %red_init = linalg.fill ins(%cst : f32) outs(%red_empty : tensor<4x128xf32>) -> tensor<4x128xf32>
  %reduced = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%src : tensor<4x128x128xf32>) outs(%red_init : tensor<4x128xf32>) {
  ^bb0(%in: f32, %out0: f32):
    %sum = arith.addf %in, %out0 : f32
    linalg.yield %sum : f32
  } -> tensor<4x128xf32>
  %ew_empty = tensor.empty() : tensor<4x128xf32>
  %ew_init = linalg.fill ins(%cst : f32) outs(%ew_empty : tensor<4x128xf32>) -> tensor<4x128xf32>
  %result = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%reduced : tensor<4x128xf32>) outs(%ew_init : tensor<4x128xf32>) {
  ^bb0(%in: f32, %out0: f32):
    %sum = arith.addf %in, %out0 : f32
    linalg.yield %sum : f32
  } -> tensor<4x128xf32>
  return %result : tensor<4x128xf32>
}
// The reduction root's Common tile is [1, 1], so everything fused into it is
// bounded to [1, 1] as well: correctness-first reconciliation cannot widen a
// consumer beyond the loop it lives in. Recovering a wide elementwise tail here
// needs a better root tile, not a wider consumer tile.
// CHECK-DAG: #[[FILL:.+]] = #iree_cpu.lowering_config<vector_common_parallel = [1, 1]>
// CHECK-DAG: #[[REDUCE:.+]] = #iree_cpu.lowering_config<cache_parallel = [4, 0, 0], distribution = [0, 0, 0], vector_common_parallel = [1, 1, 0], vector_reduction = [0, 0, 32]>
// CHECK-NOT: #iree_cpu.lowering_config
// CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = #iree_cpu.pipeline<DoubleTilingExpert>, {enable_loop_peeling}>
// CHECK: func.func @generic_dispatch(
// CHECK-SAME: translation_info = #[[TRANSLATION]]
// CHECK: linalg.fill {lowering_config = #[[FILL]]}
// CHECK: linalg.generic
// CHECK-SAME: lowering_config = #[[REDUCE]]
// CHECK: linalg.fill {lowering_config = #[[FILL]]}
// CHECK: linalg.generic
// CHECK-SAME: lowering_config = #[[FILL]]

// -----

#target = #hal.executable.target<"hexagon", "embedded-elf-hexagon", {cpu = "hexagonv79", cpu_features = "+hvxv79,+hvx-length128b", data_layout = "e-m:e-p:32:32:32-a:0-n16:32-i64:64:64-i32:32:32-i16:16:16-i1:8:8-f32:32:32-f64:64:64-v32:32:32-v64:64:64-v512:512:512-v1024:1024:1024-v2048:2048:2048", hexagon.version = "79", iree.encoding.resolver = #iree_hexagon.hexagon_encoding_resolver<>, link_embedded = false, max_stack_allocation_size = 16384 : i64, native_vector_size = 32 : i64, target_triple = "hexagon-unknown-unknown-elf"}>
func.func @transpose_root_dispatch(%src: tensor<64x128xf32>) -> tensor<128x64xf32> attributes {hal.executable.target = #target} {
  %empty = tensor.empty() : tensor<128x64xf32>
  %transposed = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d1, d0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%src : tensor<64x128xf32>) outs(%empty : tensor<128x64xf32>) {
  ^bb0(%in: f32, %out0: f32):
    linalg.yield %in : f32
  } -> tensor<128x64xf32>
  return %transposed : tensor<128x64xf32>
}
// CHECK-DAG: #[[TRANSPOSE:.+]] = #iree_cpu.lowering_config<cache_parallel = [64, 0], distribution = [0, 0], vector_common_parallel = [1, 32]>
// CHECK-NOT: #iree_cpu.lowering_config
// CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = #iree_cpu.pipeline<DoubleTilingExpert>, {enable_loop_peeling}>
// CHECK: func.func @transpose_root_dispatch(
// CHECK-SAME: translation_info = #[[TRANSLATION]]
// CHECK: linalg.generic
// CHECK-SAME: iterator_types = ["parallel", "parallel"]
// CHECK-SAME: lowering_config = #[[TRANSPOSE]]

// -----

// Softmax dispatch from the attention path.
#target = #hal.executable.target<"hexagon", "embedded-elf-hexagon", {cpu = "hexagonv79", cpu_features = "+hvxv79,+hvx-length128b", data_layout = "e-m:e-p:32:32:32-a:0-n16:32-i64:64:64-i32:32:32-i16:16:16-i1:8:8-f32:32:32-f64:64:64-v32:32:32-v64:64:64-v512:512:512-v1024:1024:1024-v2048:2048:2048", hexagon.version = "79", iree.encoding.resolver = #iree_hexagon.hexagon_encoding_resolver<>, link_embedded = false, max_stack_allocation_size = 16384 : i64, native_vector_size = 32 : i64, target_triple = "hexagon-unknown-unknown-elf"}>
func.func @softmax_dispatch(%src: tensor<4x1024x1024xf32>, %mask: tensor<4x1024xi8>) -> tensor<4x1024x1024xf32> attributes {hal.executable.target = #target} {
  %cst = arith.constant 0xFFC00000 : f32
  %cst_0 = arith.constant 0.000000e+00 : f32
  %empty = tensor.empty() : tensor<4x1024x1024xf32>
  %red_empty = tensor.empty() : tensor<4x1024xf32>
  %max_init = linalg.fill ins(%cst : f32) outs(%red_empty : tensor<4x1024xf32>) -> tensor<4x1024xf32>
  %max = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%src : tensor<4x1024x1024xf32>) outs(%max_init : tensor<4x1024xf32>) {
  ^bb0(%in: f32, %out0: f32):
    %m = arith.maxnumf %in, %out0 : f32
    linalg.yield %m : f32
  } -> tensor<4x1024xf32>
  %sum_init = linalg.fill ins(%cst_0 : f32) outs(%red_empty : tensor<4x1024xf32>) -> tensor<4x1024xf32>
  %sum = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%src, %max : tensor<4x1024x1024xf32>, tensor<4x1024xf32>) outs(%sum_init : tensor<4x1024xf32>) {
  ^bb0(%in: f32, %max_in: f32, %out0: f32):
    %shifted = arith.subf %in, %max_in : f32
    %exp = math.exp %shifted : f32
    %s = arith.addf %exp, %out0 : f32
    linalg.yield %s : f32
  } -> tensor<4x1024xf32>
  %result = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%mask, %src, %max, %sum : tensor<4x1024xi8>, tensor<4x1024x1024xf32>, tensor<4x1024xf32>, tensor<4x1024xf32>) outs(%empty : tensor<4x1024x1024xf32>) {
  ^bb0(%mask_in: i8, %in: f32, %max_in: f32, %sum_in: f32, %out0: f32):
    %shifted = arith.subf %in, %max_in : f32
    %exp = math.exp %shifted : f32
    %normalized = arith.divf %exp, %sum_in : f32
    %is_masked = arith.trunci %mask_in : i8 to i1
    %selected = arith.select %is_masked, %cst_0, %normalized : f32
    linalg.yield %selected : f32
  } -> tensor<4x1024x1024xf32>
  return %result : tensor<4x1024x1024xf32>
}
// CHECK-DAG: #[[FILL:.+]] = #iree_cpu.lowering_config<vector_common_parallel = [1, 1]>
// CHECK-DAG: #[[MAX:.+]] = #iree_cpu.lowering_config<vector_common_parallel = [1, 1, 0], vector_reduction = [0, 0, 32]>
// CHECK-DAG: #[[SUM:.+]] = #iree_cpu.lowering_config<cache_parallel = [4, 0, 0], distribution = [0, 0, 0], vector_common_parallel = [1, 1, 0], vector_reduction = [0, 0, 32]>
// CHECK-DAG: #[[NORM:.+]] = #iree_cpu.lowering_config<vector_common_parallel = [1, 1, 32]>
// CHECK-NOT: #iree_cpu.lowering_config
// CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = #iree_cpu.pipeline<DoubleTilingExpert>, {enable_loop_peeling}>
// CHECK: func.func @softmax_dispatch(
// CHECK-SAME: translation_info = #[[TRANSLATION]]
// CHECK: linalg.fill {lowering_config = #[[FILL]]}
// CHECK: linalg.generic
// CHECK-SAME: iterator_types = ["parallel", "parallel", "reduction"]
// CHECK-SAME: lowering_config = #[[MAX]]
// CHECK: linalg.fill {lowering_config = #[[FILL]]}
// CHECK: linalg.generic
// CHECK-SAME: iterator_types = ["parallel", "parallel", "reduction"]
// CHECK-SAME: lowering_config = #[[SUM]]
// CHECK: linalg.generic
// CHECK-SAME: iterator_types = ["parallel", "parallel", "parallel"]
// CHECK-SAME: lowering_config = #[[NORM]]

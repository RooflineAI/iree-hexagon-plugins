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
func.func @fallback_dispatch(%src: tensor<96x96xf32>) -> tensor<98x98xf32> attributes {hal.executable.target = #target} {
  %padded = tensor.pad %src low[1, 1] high[1, 1] {
  ^bb0(%arg0: index, %arg1: index):
    %cst = arith.constant 0.0 : f32
    tensor.yield %cst : f32
  } : tensor<96x96xf32> to tensor<98x98xf32>
  return %padded : tensor<98x98xf32>
}
// CHECK-DAG: #[[PAD:.+]] = #iree_cpu.lowering_config<vector_common_parallel = [1, 32]>
// CHECK-NOT: #iree_cpu.lowering_config
// CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = #iree_cpu.pipeline<Default>>
// CHECK: func.func @fallback_dispatch(
// CHECK-SAME: translation_info = #[[TRANSLATION]]
// CHECK: tensor.pad
// CHECK: lowering_config = #[[PAD]]

// -----

#target = #hal.executable.target<"hexagon", "embedded-elf-hexagon", {cpu = "hexagonv79", cpu_features = "+hvxv79,+hvx-length128b", data_layout = "e-m:e-p:32:32:32-a:0-n16:32-i64:64:64-i32:32:32-i16:16:16-i1:8:8-f32:32:32-f64:64:64-v32:32:32-v64:64:64-v512:512:512-v1024:1024:1024-v2048:2048:2048", hexagon.version = "79", iree.encoding.resolver = #iree_hexagon.hexagon_encoding_resolver<>, link_embedded = false, max_stack_allocation_size = 16384 : i64, native_vector_size = 32 : i64, target_triple = "hexagon-unknown-unknown-elf"}>
func.func @fill_root_dispatch() -> tensor<64x128xf32> attributes {hal.executable.target = #target} {
  %cst = arith.constant 0.0 : f32
  %empty = tensor.empty() : tensor<64x128xf32>
  %filled = linalg.fill ins(%cst : f32) outs(%empty : tensor<64x128xf32>) -> tensor<64x128xf32>
  return %filled : tensor<64x128xf32>
}
// CHECK-DAG: #[[FILL:.+]] = #iree_cpu.lowering_config<distribution = [0, 0], vector_common_parallel = [1, 32]>
// CHECK-NOT: #iree_cpu.lowering_config
// CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = #iree_cpu.pipeline<DoubleTilingExpert>, {enable_loop_peeling}>
// CHECK: func.func @fill_root_dispatch(
// CHECK-SAME: translation_info = #[[TRANSLATION]]
// CHECK: linalg.fill {lowering_config = #[[FILL]]}

// -----

#target = #hal.executable.target<"hexagon", "embedded-elf-hexagon", {cpu = "hexagonv79", cpu_features = "+hvxv79,+hvx-length128b", data_layout = "e-m:e-p:32:32:32-a:0-n16:32-i64:64:64-i32:32:32-i16:16:16-i1:8:8-f32:32:32-f64:64:64-v32:32:32-v64:64:64-v512:512:512-v1024:1024:1024-v2048:2048:2048", hexagon.version = "79", iree.encoding.resolver = #iree_hexagon.hexagon_encoding_resolver<>, link_embedded = false, max_stack_allocation_size = 16384 : i64, native_vector_size = 32 : i64, target_triple = "hexagon-unknown-unknown-elf"}>
func.func @fft_fallback_dispatch(%twiddle_real: tensor<2xf32>, %twiddle_imag: tensor<2xf32>) -> (tensor<32xf32>, tensor<32xf32>) attributes {hal.executable.target = #target} {
  %c2 = arith.constant 2 : index
  %empty_real = tensor.empty() : tensor<32xf32>
  %empty_imag = tensor.empty() : tensor<32xf32>
  %fft_real, %fft_imag = iree_linalg_ext.fft ins(%c2, %twiddle_real, %twiddle_imag : index, tensor<2xf32>, tensor<2xf32>) outs(%empty_real, %empty_imag : tensor<32xf32>, tensor<32xf32>) : tensor<32xf32>, tensor<32xf32>
  return %fft_real, %fft_imag : tensor<32xf32>, tensor<32xf32>
}
// CHECK-DAG: #[[FFT:.+]] = #iree_cpu.lowering_config<vector_common_parallel = [32]>
// CHECK-NOT: #iree_cpu.lowering_config
// CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = #iree_cpu.pipeline<Default>>
// CHECK: func.func @fft_fallback_dispatch(
// CHECK-SAME: translation_info = #[[TRANSLATION]]
// CHECK: iree_linalg_ext.fft
// CHECK-SAME: lowering_config = #[[FFT]]

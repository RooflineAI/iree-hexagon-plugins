// Test written to reflect current Hexagon selector policy.
//
// This file exercises the non-VTCM, non-HMX strategy policies. Dedicated tests
// cover those opt-in paths below the planner boundary.
//
// RUN: iree-opt \
// RUN:   --iree-hexagon-enable-vtcm-tiling=false \
// RUN:   --iree-hexagon-enable-hmx-matmul=false \
// RUN:   --pass-pipeline='builtin.module(iree-hexagon-select-lowering-strategy)' \
// RUN:   --split-input-file %s | FileCheck %s
//
// Same selection, but generalize named ops first so that convolution/pooling
// selection is exercised on their linalg.generic form too (mirrors upstream
// IREE). inferOpLoweringPlan classifies by structure before op type, so a
// generalized conv/pool is still routed through inferConvPlan and receives the
// same root tiling as the named op. The GENERIC checks assert only the root op:
// the fused fill is generalized into a linalg.generic and tiled as one, so its
// config differs from the named form and is not asserted.
//
// RUN: iree-opt \
// RUN:   --iree-hexagon-enable-vtcm-tiling=false \
// RUN:   --iree-hexagon-enable-hmx-matmul=false \
// RUN:   --pass-pipeline='builtin.module(func.func(linalg-generalize-named-ops),iree-hexagon-select-lowering-strategy)' \
// RUN:   --split-input-file %s | FileCheck %s --check-prefix=GENERIC


// Note that we still expect 32 as tile target despite it not being a divisor of the tensor shape
#target = #hal.executable.target<"hexagon", "embedded-elf-hexagon", {cpu = "hexagonv79", cpu_features = "+hvxv79,+hvx-length128b", data_layout = "e-m:e-p:32:32:32-a:0-n16:32-i64:64:64-i32:32:32-i16:16:16-i1:8:8-f32:32:32-f64:64:64-v32:32:32-v64:64:64-v512:512:512-v1024:1024:1024-v2048:2048:2048", hexagon.version = "79", iree.encoding.resolver = #iree_hexagon.hexagon_encoding_resolver<>, link_embedded = false, max_stack_allocation_size = 16384 : i64, native_vector_size = 32 : i64, target_triple = "hexagon-unknown-unknown-elf"}>
func.func @pooling_root_dispatch(%src: tensor<1x64x114x114xf32>) -> tensor<1x64x56x56xf32> attributes {hal.executable.target = #target} {
  %cst = arith.constant 0xFF800000 : f32
  %empty = tensor.empty() : tensor<1x64x56x56xf32>
  %kernel = tensor.empty() : tensor<3x3xf32>
  %init = linalg.fill ins(%cst : f32) outs(%empty : tensor<1x64x56x56xf32>) -> tensor<1x64x56x56xf32>
  %result = linalg.pooling_nchw_max {dilations = dense<1> : vector<2xi64>, strides = dense<2> : vector<2xi64>} ins(%src, %kernel : tensor<1x64x114x114xf32>, tensor<3x3xf32>) outs(%init : tensor<1x64x56x56xf32>) -> tensor<1x64x56x56xf32>
  return %result : tensor<1x64x56x56xf32>
}
// CHECK-DAG: #[[POOL:.+]] = #iree_cpu.lowering_config<distribution = [1, 32, 1, 28, 0, 0], vector_common_parallel = [1, 32, 1, 28, 0, 0]>
// CHECK-NOT: #iree_cpu.lowering_config
// CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = #iree_cpu.pipeline<ConvTileAndDecomposeExpert>>
// CHECK: func.func @pooling_root_dispatch(
// CHECK-SAME: translation_info = #[[TRANSLATION]]
// CHECK: linalg.fill
// CHECK-NOT: lowering_config
// CHECK: linalg.pooling_nchw_max
// CHECK-SAME: lowering_config = #[[POOL]]
// GENERIC-DAG: #[[GPOOL:.+]] = #iree_cpu.lowering_config<distribution = [1, 32, 1, 28, 0, 0], vector_common_parallel = [1, 32, 1, 28, 0, 0]>
// GENERIC-DAG: #[[GTRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = #iree_cpu.pipeline<ConvTileAndDecomposeExpert>>
// GENERIC: func.func @pooling_root_dispatch(
// GENERIC-SAME: translation_info = #[[GTRANSLATION]]
// GENERIC: linalg.generic
// GENERIC: lowering_config = #[[GPOOL]]

// -----

#target = #hal.executable.target<"hexagon", "embedded-elf-hexagon", {cpu = "hexagonv79", cpu_features = "+hvxv79,+hvx-length128b", data_layout = "e-m:e-p:32:32:32-a:0-n16:32-i64:64:64-i32:32:32-i16:16:16-i1:8:8-f32:32:32-f64:64:64-v32:32:32-v64:64:64-v512:512:512-v1024:1024:1024-v2048:2048:2048", hexagon.version = "79", iree.encoding.resolver = #iree_hexagon.hexagon_encoding_resolver<>, link_embedded = false, max_stack_allocation_size = 16384 : i64, native_vector_size = 32 : i64, target_triple = "hexagon-unknown-unknown-elf"}>
func.func @conv_fallback_dispatch(%input: tensor<1x32x32x8xf32>, %filter: tensor<3x3x8x16xf32>) -> tensor<1x30x30x16xf32> attributes {hal.executable.target = #target} {
  %cst = arith.constant 0.0 : f32
  %empty = tensor.empty() : tensor<1x30x30x16xf32>
  %init = linalg.fill ins(%cst : f32) outs(%empty : tensor<1x30x30x16xf32>) -> tensor<1x30x30x16xf32>
  %result = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%input, %filter : tensor<1x32x32x8xf32>, tensor<3x3x8x16xf32>) outs(%init : tensor<1x30x30x16xf32>) -> tensor<1x30x30x16xf32>
  return %result : tensor<1x30x30x16xf32>
}
// CHECK-DAG: #[[CONV:.+]] = #iree_cpu.lowering_config<distribution = [1, 1, 30, 16, 0, 0, 0], vector_common_parallel = [1, 1, 30, 16, 0, 0, 0]>
// CHECK-NOT: #iree_cpu.lowering_config
// CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = #iree_cpu.pipeline<ConvTileAndDecomposeExpert>>
// CHECK: func.func @conv_fallback_dispatch(
// CHECK-SAME: translation_info = #[[TRANSLATION]]
// CHECK: linalg.fill
// CHECK-NOT: lowering_config
// CHECK: linalg.conv_2d_nhwc_hwcf
// CHECK-SAME: lowering_config = #[[CONV]]
// GENERIC-DAG: #[[GCONV:.+]] = #iree_cpu.lowering_config<distribution = [1, 1, 30, 16, 0, 0, 0], vector_common_parallel = [1, 1, 30, 16, 0, 0, 0]>
// GENERIC-DAG: #[[GTRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = #iree_cpu.pipeline<ConvTileAndDecomposeExpert>>
// GENERIC: func.func @conv_fallback_dispatch(
// GENERIC-SAME: translation_info = #[[GTRANSLATION]]
// GENERIC: linalg.generic
// GENERIC: lowering_config = #[[GCONV]]

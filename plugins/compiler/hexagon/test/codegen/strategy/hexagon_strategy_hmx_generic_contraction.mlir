// HMX lowering converts only named linalg.matmul and linalg.batch_matmul. A
// matmul-shaped linalg.generic that still reaches strategy selection (the
// configuration pipeline specializes them first) must not be planned for HMX:
// its K dimension would be left untiled for HMX packing that never happens,
// and the whole reduction vectorized instead. The planner warns and falls
// back to the regular contraction strategy.
//
// RUN: iree-opt \
// RUN:   --iree-hexagon-enable-vtcm-tiling --iree-hexagon-enable-hmx-matmul \
// RUN:   --pass-pipeline='builtin.module(iree-hexagon-select-lowering-strategy)' \
// RUN:   --split-input-file --verify-diagnostics %s | FileCheck %s

#target = #hal.executable.target<"hexagon", "embedded-elf-hexagon", {cpu = "hexagonv79", cpu_features = "+hvxv79,+hvx-length128b", data_layout = "e-m:e-p:32:32:32-a:0-n16:32-i64:64:64-i32:32:32-i16:16:16-i1:8:8-f32:32:32-f64:64:64-v32:32:32-v64:64:64-v512:512:512-v1024:1024:1024-v2048:2048:2048", hexagon.version = "79", max_stack_allocation_size = 16384 : i64, native_vector_size = 128 : i64, target_triple = "hexagon-unknown-unknown-elf"}>
func.func @hmx_generic_contraction(%lhs: tensor<256x1536xf16>, %rhs: tensor<1024x1536xf16>) -> tensor<256x1024xf32> attributes {hal.executable.target = #target} {
  %zero = arith.constant 0.0 : f32
  %empty = tensor.empty() : tensor<256x1024xf32>
  %init = linalg.fill ins(%zero : f32) outs(%empty : tensor<256x1024xf32>) -> tensor<256x1024xf32>
  // expected-warning @+1 {{matmul-like 'linalg.generic' is not lowered to HMX: only named linalg.matmul and linalg.batch_matmul with an MxN output are; planning it without HMX}}
  %result = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%lhs, %rhs : tensor<256x1536xf16>, tensor<1024x1536xf16>) outs(%init : tensor<256x1024xf32>) {
  ^bb0(%a: f16, %b: f16, %c: f32):
    %a32 = arith.extf %a : f16 to f32
    %b32 = arith.extf %b : f16 to f32
    %m = arith.mulf %a32, %b32 : f32
    %s = arith.addf %c, %m : f32
    linalg.yield %s : f32
  } -> tensor<256x1024xf32>
  return %result : tensor<256x1024xf32>
}
// The contraction strategy tiles K (vector_reduction), unlike the HMX plan.
// CHECK-NOT: HmxMatmulExpert
// CHECK-DAG: #[[ROOT:.+]] = #iree_hexagon.lowering_config<{{.*}}vector_reduction = [0, 0, {{[1-9][0-9]*}}]
// CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = #iree_hexagon.pipeline<MultiTilingExpert>
// CHECK: func.func @hmx_generic_contraction(
// CHECK-SAME: translation_info = #[[TRANSLATION]]
// CHECK: linalg.generic
// CHECK-SAME: lowering_config = #[[ROOT]]

// -----

// The same contraction as a named op is planned for HMX, with no warning.
#target = #hal.executable.target<"hexagon", "embedded-elf-hexagon", {cpu = "hexagonv79", cpu_features = "+hvxv79,+hvx-length128b", data_layout = "e-m:e-p:32:32:32-a:0-n16:32-i64:64:64-i32:32:32-i16:16:16-i1:8:8-f32:32:32-f64:64:64-v32:32:32-v64:64:64-v512:512:512-v1024:1024:1024-v2048:2048:2048", hexagon.version = "79", max_stack_allocation_size = 16384 : i64, native_vector_size = 128 : i64, target_triple = "hexagon-unknown-unknown-elf"}>
func.func @hmx_named_contraction(%lhs: tensor<256x1536xf16>, %rhs: tensor<1024x1536xf16>) -> tensor<256x1024xf32> attributes {hal.executable.target = #target} {
  %zero = arith.constant 0.0 : f32
  %empty = tensor.empty() : tensor<256x1024xf32>
  %init = linalg.fill ins(%zero : f32) outs(%empty : tensor<256x1024xf32>) -> tensor<256x1024xf32>
  %result = linalg.matmul indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>] ins(%lhs, %rhs : tensor<256x1536xf16>, tensor<1024x1536xf16>) outs(%init : tensor<256x1024xf32>) -> tensor<256x1024xf32>
  return %result : tensor<256x1024xf32>
}
// CHECK-DAG: #[[NAMED_TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = #iree_hexagon.pipeline<HmxMatmulExpert>>
// CHECK: func.func @hmx_named_contraction(
// CHECK-SAME: translation_info = #[[NAMED_TRANSLATION]]
// CHECK: linalg.matmul

// -----

// A named matmul with a transposed (N x M) output is rejected by HMX lowering,
// so it is planned without HMX as well.
#target = #hal.executable.target<"hexagon", "embedded-elf-hexagon", {cpu = "hexagonv79", cpu_features = "+hvxv79,+hvx-length128b", data_layout = "e-m:e-p:32:32:32-a:0-n16:32-i64:64:64-i32:32:32-i16:16:16-i1:8:8-f32:32:32-f64:64:64-v32:32:32-v64:64:64-v512:512:512-v1024:1024:1024-v2048:2048:2048", hexagon.version = "79", max_stack_allocation_size = 16384 : i64, native_vector_size = 128 : i64, target_triple = "hexagon-unknown-unknown-elf"}>
func.func @hmx_named_transposed_output(%lhs: tensor<256x1536xf16>, %rhs: tensor<1024x1536xf16>) -> tensor<1024x256xf32> attributes {hal.executable.target = #target} {
  %zero = arith.constant 0.0 : f32
  %empty = tensor.empty() : tensor<1024x256xf32>
  %init = linalg.fill ins(%zero : f32) outs(%empty : tensor<1024x256xf32>) -> tensor<1024x256xf32>
  // expected-warning @+1 {{matmul-like 'linalg.matmul' is not lowered to HMX: only named linalg.matmul and linalg.batch_matmul with an MxN output are; planning it without HMX}}
  %result = linalg.matmul indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d1, d0)>] ins(%lhs, %rhs : tensor<256x1536xf16>, tensor<1024x1536xf16>) outs(%init : tensor<1024x256xf32>) -> tensor<1024x256xf32>
  return %result : tensor<1024x256xf32>
}
// CHECK-DAG: #[[TRANSPOSED_TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = #iree_hexagon.pipeline<MultiTilingExpert>
// CHECK: func.func @hmx_named_transposed_output(
// CHECK-SAME: translation_info = #[[TRANSPOSED_TRANSLATION]]

// -----

// HMX accumulates into f16 or f32 only, so a named matmul with another output
// type is not an HMX candidate and is planned without HMX or a warning.
#target = #hal.executable.target<"hexagon", "embedded-elf-hexagon", {cpu = "hexagonv79", cpu_features = "+hvxv79,+hvx-length128b", data_layout = "e-m:e-p:32:32:32-a:0-n16:32-i64:64:64-i32:32:32-i16:16:16-i1:8:8-f32:32:32-f64:64:64-v32:32:32-v64:64:64-v512:512:512-v1024:1024:1024-v2048:2048:2048", hexagon.version = "79", max_stack_allocation_size = 16384 : i64, native_vector_size = 128 : i64, target_triple = "hexagon-unknown-unknown-elf"}>
func.func @hmx_named_unsupported_output_type(%lhs: tensor<256x1536xf16>, %rhs: tensor<1536x1024xf16>) -> tensor<256x1024xbf16> attributes {hal.executable.target = #target} {
  %zero = arith.constant 0.0 : bf16
  %empty = tensor.empty() : tensor<256x1024xbf16>
  %init = linalg.fill ins(%zero : bf16) outs(%empty : tensor<256x1024xbf16>) -> tensor<256x1024xbf16>
  %result = linalg.matmul ins(%lhs, %rhs : tensor<256x1536xf16>, tensor<1536x1024xf16>) outs(%init : tensor<256x1024xbf16>) -> tensor<256x1024xbf16>
  return %result : tensor<256x1024xbf16>
}
// CHECK-DAG: #[[BF16_TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = #iree_hexagon.pipeline<MultiTilingExpert>
// CHECK: func.func @hmx_named_unsupported_output_type(
// CHECK-SAME: translation_info = #[[BF16_TRANSLATION]]

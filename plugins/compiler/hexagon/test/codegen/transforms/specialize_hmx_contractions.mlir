// HMX lowering converts only named linalg.matmul and linalg.batch_matmul, so
// HMX-eligible contractions that reach codegen as linalg.generic are rewritten
// as named ops before strategy selection. Everything else is left alone.
//
// RUN: iree-opt \
// RUN:   --pass-pipeline='builtin.module(func.func(iree-hexagon-specialize-hmx-contractions))' \
// RUN:   --split-input-file %s | FileCheck %s

// The Qwen3-VL patch embedding: a Conv3d whose window covers its whole input
// becomes A * B^T with f16 inputs extended to f32 inside the body.
// CHECK-DAG: #[[$A:.+]] = affine_map<(d0, d1, d2) -> (d0, d2)>
// CHECK-DAG: #[[$BT:.+]] = affine_map<(d0, d1, d2) -> (d1, d2)>
// CHECK-DAG: #[[$C:.+]] = affine_map<(d0, d1, d2) -> (d0, d1)>
// CHECK-LABEL: func.func @f16_mixed_precision_transposed_rhs(
// CHECK-NOT:     linalg.generic
// CHECK:         linalg.matmul indexing_maps = [#[[$A]], #[[$BT]], #[[$C]]]
// CHECK-SAME:      ins(%{{.*}}, %{{.*}} : tensor<64x96xf16>, tensor<128x96xf16>)
// CHECK-SAME:      outs(%{{.*}} : tensor<64x128xf32>)
// CHECK-NOT:     linalg.generic
func.func @f16_mixed_precision_transposed_rhs(%lhs: tensor<64x96xf16>, %rhs: tensor<128x96xf16>, %init: tensor<64x128xf32>) -> tensor<64x128xf32> {
  %0 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%lhs, %rhs : tensor<64x96xf16>, tensor<128x96xf16>) outs(%init : tensor<64x128xf32>) {
  ^bb0(%a: f16, %b: f16, %c: f32):
    %a32 = arith.extf %a : f16 to f32
    %b32 = arith.extf %b : f16 to f32
    %m = arith.mulf %a32, %b32 : f32
    %s = arith.addf %c, %m : f32
    linalg.yield %s : f32
  } -> tensor<64x128xf32>
  return %0 : tensor<64x128xf32>
}

// -----

// CHECK-LABEL: func.func @f16_batch_contraction(
// CHECK:         linalg.batch_matmul
// CHECK-SAME:      ins(%{{.*}}, %{{.*}} : tensor<4x32x64xf16>, tensor<4x64x48xf16>)
// CHECK-NOT:     linalg.generic
func.func @f16_batch_contraction(%lhs: tensor<4x32x64xf16>, %rhs: tensor<4x64x48xf16>, %init: tensor<4x32x48xf16>) -> tensor<4x32x48xf16> {
  %0 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%lhs, %rhs : tensor<4x32x64xf16>, tensor<4x64x48xf16>) outs(%init : tensor<4x32x48xf16>) {
  ^bb0(%a: f16, %b: f16, %c: f16):
    %m = arith.mulf %a, %b : f16
    %s = arith.addf %c, %m : f16
    linalg.yield %s : f16
  } -> tensor<4x32x48xf16>
  return %0 : tensor<4x32x48xf16>
}

// -----

// Not HMX-eligible: f32 inputs.
// CHECK-LABEL: func.func @f32_contraction_is_kept(
// CHECK:         linalg.generic
// CHECK-NOT:     linalg.matmul
func.func @f32_contraction_is_kept(%lhs: tensor<64x96xf32>, %rhs: tensor<96x128xf32>, %init: tensor<64x128xf32>) -> tensor<64x128xf32> {
  %0 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%lhs, %rhs : tensor<64x96xf32>, tensor<96x128xf32>) outs(%init : tensor<64x128xf32>) {
  ^bb0(%a: f32, %b: f32, %c: f32):
    %m = arith.mulf %a, %b : f32
    %s = arith.addf %c, %m : f32
    linalg.yield %s : f32
  } -> tensor<64x128xf32>
  return %0 : tensor<64x128xf32>
}

// -----

// A transposed (N x M) output would specialize to a linalg.matmul that HMX
// lowering rejects outright, so it stays a generic.
// CHECK-LABEL: func.func @transposed_output_is_kept(
// CHECK:         linalg.generic
// CHECK-NOT:     linalg.matmul
func.func @transposed_output_is_kept(%lhs: tensor<64x96xf16>, %rhs: tensor<96x128xf16>, %init: tensor<128x64xf32>) -> tensor<128x64xf32> {
  %0 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d1, d0)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%lhs, %rhs : tensor<64x96xf16>, tensor<96x128xf16>) outs(%init : tensor<128x64xf32>) {
  ^bb0(%a: f16, %b: f16, %c: f32):
    %a32 = arith.extf %a : f16 to f32
    %b32 = arith.extf %b : f16 to f32
    %m = arith.mulf %a32, %b32 : f32
    %s = arith.addf %c, %m : f32
    linalg.yield %s : f32
  } -> tensor<128x64xf32>
  return %0 : tensor<128x64xf32>
}

// -----

// Two reduction dimensions: a contraction, but not matmul-shaped.
// CHECK-LABEL: func.func @two_reduction_dims_are_kept(
// CHECK:         linalg.generic
// CHECK-NOT:     linalg.matmul
// CHECK-NOT:     linalg.contract
func.func @two_reduction_dims_are_kept(%lhs: tensor<32x8x16xf16>, %rhs: tensor<16x8x48xf16>, %init: tensor<32x48xf16>) -> tensor<32x48xf16> {
  %0 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d3, d2, d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction", "reduction"]} ins(%lhs, %rhs : tensor<32x8x16xf16>, tensor<16x8x48xf16>) outs(%init : tensor<32x48xf16>) {
  ^bb0(%a: f16, %b: f16, %c: f16):
    %m = arith.mulf %a, %b : f16
    %s = arith.addf %c, %m : f16
    linalg.yield %s : f16
  } -> tensor<32x48xf16>
  return %0 : tensor<32x48xf16>
}

// -----

// Elementwise generics (the bias epilogue next to the contraction) are not
// touched, even where upstream specialization would match them.
// CHECK-LABEL: func.func @elementwise_is_kept(
// CHECK:         linalg.generic
// CHECK-NOT:     linalg.add
func.func @elementwise_is_kept(%lhs: tensor<64x128xf16>, %rhs: tensor<64x128xf16>, %init: tensor<64x128xf16>) -> tensor<64x128xf16> {
  %0 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%lhs, %rhs : tensor<64x128xf16>, tensor<64x128xf16>) outs(%init : tensor<64x128xf16>) {
  ^bb0(%a: f16, %b: f16, %c: f16):
    %s = arith.addf %a, %b : f16
    linalg.yield %s : f16
  } -> tensor<64x128xf16>
  return %0 : tensor<64x128xf16>
}

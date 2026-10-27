// RUN: iree-opt --pass-pipeline="builtin.module(iree-hexagon-sort-to-topk)" --split-input-file %s | FileCheck %s

// torch.topk as torch-mlir decomposes it: a descending sort of the values and
// an iota, then the leading k of both results.

util.global private @device = #hal.device.target<"hexagon", [#hal.executable.target<"hexagon", "embedded-elf-hexagon">]> : !hal.device

func.func @topk_descending(%values: tensor<1x2048xf16>) -> (tensor<1x32xf16>, tensor<1x32xi64>) {
  %empty = tensor.empty() : tensor<1x2048xi64>
  %iota = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} outs(%empty : tensor<1x2048xi64>) {
  ^bb0(%out: i64):
    %index = linalg.index 1 : index
    %cast = arith.index_cast %index : index to i64
    linalg.yield %cast : i64
  } -> tensor<1x2048xi64>
  %sorted:2 = iree_linalg_ext.sort dimension(1) outs(%values, %iota : tensor<1x2048xf16>, tensor<1x2048xi64>) {
  ^bb0(%lhs: f16, %rhs: f16, %lhs_index: i64, %rhs_index: i64):
    %cmp = arith.cmpf oge, %lhs, %rhs : f16
    iree_linalg_ext.yield %cmp : i1
  } -> tensor<1x2048xf16>, tensor<1x2048xi64>
  %top_values = tensor.extract_slice %sorted#0[0, 0] [1, 32] [1, 1] : tensor<1x2048xf16> to tensor<1x32xf16>
  %top_indices = tensor.extract_slice %sorted#1[0, 0] [1, 32] [1, 1] : tensor<1x2048xi64> to tensor<1x32xi64>
  return %top_values, %top_indices : tensor<1x32xf16>, tensor<1x32xi64>
}

// CHECK-LABEL: func.func @topk_descending
//  CHECK-SAME:   %[[VALUES:[a-zA-Z0-9]+]]: tensor<1x2048xf16>
//   CHECK-DAG:   %[[NEG_INF:.+]] = arith.constant 0xFC00 : f16
//   CHECK-DAG:   %[[MAX_I32:.+]] = arith.constant 2147483647 : i32
//       CHECK:   %[[OUT_VALUES:.+]] = linalg.fill ins(%[[NEG_INF]] : f16) outs(%{{.+}} : tensor<1x32xf16>)
//       CHECK:   %[[OUT_INDICES:.+]] = linalg.fill ins(%[[MAX_I32]] : i32) outs(%{{.+}} : tensor<1x32xi32>)
//       CHECK:   %[[TOPK:.+]]:2 = iree_linalg_ext.topk dimension(1)
//  CHECK-SAME:       ins(%[[VALUES]] : tensor<1x2048xf16>)
//  CHECK-SAME:       outs(%[[OUT_VALUES]], %[[OUT_INDICES]] : tensor<1x32xf16>, tensor<1x32xi32>)
//       CHECK:     arith.cmpf ogt
//       CHECK:   %[[INDICES:.+]] = linalg.generic
//  CHECK-SAME:       ins(%[[TOPK]]#1 : tensor<1x32xi32>)
//       CHECK:     arith.extsi %{{.+}} : i32 to i64
//   CHECK-NOT:   iree_linalg_ext.sort
//       CHECK:   return %[[TOPK]]#0, %[[INDICES]]

// -----

// Ascending integer sort without indices: a min-k with the signed max as init.

util.global private @device = #hal.device.target<"hexagon", [#hal.executable.target<"hexagon", "embedded-elf-hexagon">]> : !hal.device

func.func @mink_int_values_only(%values: tensor<64x8xi32>) -> tensor<4x8xi32> {
  %sorted = iree_linalg_ext.sort dimension(0) outs(%values : tensor<64x8xi32>) {
  ^bb0(%lhs: i32, %rhs: i32):
    %cmp = arith.cmpi sle, %lhs, %rhs : i32
    iree_linalg_ext.yield %cmp : i1
  } -> tensor<64x8xi32>
  %slice = tensor.extract_slice %sorted[0, 0] [4, 8] [1, 1] : tensor<64x8xi32> to tensor<4x8xi32>
  return %slice : tensor<4x8xi32>
}

// CHECK-LABEL: func.func @mink_int_values_only
//       CHECK:   arith.constant 2147483647 : i32
//       CHECK:   %[[TOPK:.+]]:2 = iree_linalg_ext.topk dimension(0)
//       CHECK:     arith.cmpi slt
//   CHECK-NOT:   linalg.generic
//       CHECK:   return %[[TOPK]]#0

// -----

// Not rewritten: the slice does not start at 0, a sort result is used
// directly, or the indices are not an iota.

util.global private @device = #hal.device.target<"hexagon", [#hal.executable.target<"hexagon", "embedded-elf-hexagon">]> : !hal.device

func.func @offset_slice(%values: tensor<2048xf32>) -> tensor<32xf32> {
  %sorted = iree_linalg_ext.sort dimension(0) outs(%values : tensor<2048xf32>) {
  ^bb0(%lhs: f32, %rhs: f32):
    %cmp = arith.cmpf oge, %lhs, %rhs : f32
    iree_linalg_ext.yield %cmp : i1
  } -> tensor<2048xf32>
  %slice = tensor.extract_slice %sorted[1] [32] [1] : tensor<2048xf32> to tensor<32xf32>
  return %slice : tensor<32xf32>
}

func.func @full_use(%values: tensor<2048xf32>) -> (tensor<32xf32>, tensor<2048xf32>) {
  %sorted = iree_linalg_ext.sort dimension(0) outs(%values : tensor<2048xf32>) {
  ^bb0(%lhs: f32, %rhs: f32):
    %cmp = arith.cmpf oge, %lhs, %rhs : f32
    iree_linalg_ext.yield %cmp : i1
  } -> tensor<2048xf32>
  %slice = tensor.extract_slice %sorted[0] [32] [1] : tensor<2048xf32> to tensor<32xf32>
  return %slice, %sorted : tensor<32xf32>, tensor<2048xf32>
}

func.func @indices_not_iota(%values: tensor<2048xf32>, %indices: tensor<2048xi64>) -> tensor<32xi64> {
  %sorted:2 = iree_linalg_ext.sort dimension(0) outs(%values, %indices : tensor<2048xf32>, tensor<2048xi64>) {
  ^bb0(%lhs: f32, %rhs: f32, %lhs_index: i64, %rhs_index: i64):
    %cmp = arith.cmpf oge, %lhs, %rhs : f32
    iree_linalg_ext.yield %cmp : i1
  } -> tensor<2048xf32>, tensor<2048xi64>
  %slice = tensor.extract_slice %sorted#1[0] [32] [1] : tensor<2048xi64> to tensor<32xi64>
  return %slice : tensor<32xi64>
}

// CHECK-LABEL: func.func @offset_slice
//       CHECK:   iree_linalg_ext.sort
// CHECK-LABEL: func.func @full_use
//       CHECK:   iree_linalg_ext.sort
// CHECK-LABEL: func.func @indices_not_iota
//       CHECK:   iree_linalg_ext.sort
//   CHECK-NOT:   iree_linalg_ext.topk

// -----

// Not rewritten: the plugin is active by default, so modules for other
// devices keep their sort.

util.global private @device = #hal.device.target<"local", [#hal.executable.target<"llvm-cpu", "embedded-elf-x86_64">]> : !hal.device

func.func @other_device(%values: tensor<2048xf32>) -> tensor<32xf32> {
  %sorted = iree_linalg_ext.sort dimension(0) outs(%values : tensor<2048xf32>) {
  ^bb0(%lhs: f32, %rhs: f32):
    %cmp = arith.cmpf oge, %lhs, %rhs : f32
    iree_linalg_ext.yield %cmp : i1
  } -> tensor<2048xf32>
  %slice = tensor.extract_slice %sorted[0] [32] [1] : tensor<2048xf32> to tensor<32xf32>
  return %slice : tensor<32xf32>
}

// CHECK-LABEL: func.func @other_device
//       CHECK:   iree_linalg_ext.sort
//   CHECK-NOT:   iree_linalg_ext.topk

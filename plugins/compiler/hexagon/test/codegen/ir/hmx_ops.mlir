// RUN: iree-opt \
// RUN:   --split-input-file %s | FileCheck %s

// This file tests the HMX dialect surface in isolation.

// CHECK-LABEL: func.func @accumulator_primitives(
// CHECK: iree_hexagon.hmx.acc.setup_read
// CHECK: %[[ZERO:.+]] = iree_hexagon.hmx.acc.zero : !iree_hexagon.hmx.acc<32x32xf32>
// CHECK: %[[NEXT:.+]] = iree_hexagon.hmx.mma {{.+}}, {{.+}}, %[[ZERO]]
// CHECK-SAME: !iree_hexagon.hmx.acc<32x32xf32> -> !iree_hexagon.hmx.acc<32x32xf32>
// CHECK: iree_hexagon.hmx.acc.read %[[NEXT]]
func.func @accumulator_primitives(
    %config: memref<2048xi8, 1>,
    %lhs: memref<16x32x2xf16, 1>,
    %rhs: memref<16x32x2xf16, 1>,
    %dest: memref<16x32x2xf16, 1>) {
  iree_hexagon.hmx.acc.setup_read %config : memref<2048xi8, 1>
  %zero = iree_hexagon.hmx.acc.zero
      : !iree_hexagon.hmx.acc<32x32xf32>
  %next = iree_hexagon.hmx.mma %lhs, %rhs, %zero
      : memref<16x32x2xf16, 1>, memref<16x32x2xf16, 1>,
        !iree_hexagon.hmx.acc<32x32xf32>
      -> !iree_hexagon.hmx.acc<32x32xf32>
  iree_hexagon.hmx.acc.read %next, %dest
      : !iree_hexagon.hmx.acc<32x32xf32>, memref<16x32x2xf16, 1>
  return
}

// -----

// CHECK-LABEL: func.func @buffer_ops(
// CHECK: iree_hexagon.hmx.pack
// CHECK-SAME: {dim = 0 : i64}
// CHECK: iree_hexagon.hmx.matmul
// CHECK: iree_hexagon.hmx.unpack
// CHECK-SAME: {dim = 0 : i64}
func.func @buffer_ops(
    %source: memref<32x32xf16, 1>,
    %lhs: memref<1x1x16x32x2xf16, 1>,
    %rhs: memref<1x1x16x32x2xf16, 1>,
    %packed: memref<1x1x16x32x2xf16, 1>,
    %result: memref<32x32xf32, 1>) {
  iree_hexagon.hmx.pack
      ins(%source : memref<32x32xf16, 1>)
      outs(%packed : memref<1x1x16x32x2xf16, 1>)
      {dim = 0 : i64}
  iree_hexagon.hmx.matmul
      ins(%lhs, %rhs : memref<1x1x16x32x2xf16, 1>,
                       memref<1x1x16x32x2xf16, 1>)
      outs(%packed : memref<1x1x16x32x2xf16, 1>)
  iree_hexagon.hmx.unpack
      ins(%packed : memref<1x1x16x32x2xf16, 1>)
      outs(%result : memref<32x32xf32, 1>)
      {dim = 0 : i64}
  return
}

// -----

// CHECK-LABEL: func.func @tensor_ops(
// CHECK: iree_hexagon.hmx.tensor_pack
// CHECK: iree_hexagon.hmx.tensor_matmul
// CHECK: iree_hexagon.hmx.tensor_unpack
func.func @tensor_ops(
    %source: tensor<32x32xf16>,
    %lhs: tensor<1x1x16x32x2xf16>,
    %rhs: tensor<1x1x16x32x2xf16>,
    %packed: tensor<1x1x16x32x2xf16>,
    %result: tensor<32x32xf32>) -> tensor<32x32xf32> {
  %packed_result = iree_hexagon.hmx.tensor_pack
      ins(%source : tensor<32x32xf16>)
      outs(%packed : tensor<1x1x16x32x2xf16>)
      {dim = 0 : i64}
      -> tensor<1x1x16x32x2xf16>
  %product = iree_hexagon.hmx.tensor_matmul
      ins(%lhs, %rhs : tensor<1x1x16x32x2xf16>,
                       tensor<1x1x16x32x2xf16>)
      outs(%packed_result : tensor<1x1x16x32x2xf16>)
      -> tensor<1x1x16x32x2xf16>
  %unpacked = iree_hexagon.hmx.tensor_unpack
      ins(%product : tensor<1x1x16x32x2xf16>)
      outs(%result : tensor<32x32xf32>)
      {dim = 0 : i64}
      -> tensor<32x32xf32>
  return %unpacked : tensor<32x32xf32>
}

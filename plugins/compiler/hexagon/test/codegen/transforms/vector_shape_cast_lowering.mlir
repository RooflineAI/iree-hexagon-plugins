// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Adapted from mlir/test/Dialect/Vector/vector-shape-cast-lowering-transforms.mlir
// at LLVM revision 9550cd76cadea9f25a5c0bf3ee06e9f7ce5631ed.
// First five fixed-width cases; transform script replaced by the owned pass.
// RUN: iree-opt --pass-pipeline="builtin.module(func.func(iree-hexagon-vector-shape-cast-lowering))" %s | FileCheck %s

// CHECK-LABEL: func @nop_shape_cast
//  CHECK-SAME:    %[[A:.*]]: vector<16xf32>
//       CHECK: return %[[A]] : vector<16xf32>
func.func @nop_shape_cast(%arg0: vector<16xf32>) -> vector<16xf32> {
  %0 = vector.shape_cast %arg0 : vector<16xf32> to vector<16xf32>
  return %0 : vector<16xf32>
}

// CHECK-LABEL: func @cancel_shape_cast
//  CHECK-SAME:    %[[A:.*]]: vector<16xf32>
//       CHECK: return %[[A]] : vector<16xf32>
func.func @cancel_shape_cast(%arg0: vector<16xf32>) -> vector<16xf32> {
  %0 = vector.shape_cast %arg0 : vector<16xf32> to vector<4x4xf32>
  %1 = vector.shape_cast %0 : vector<4x4xf32> to vector<16xf32>
  return %1 : vector<16xf32>
}

// Collapse 2-D to 1-D.
// CHECK-LABEL: func @shape_cast_2d1d
//  CHECK-SAME:    %[[A:.*]]: vector<2x2xf32>) -> vector<4xf32> {
//       CHECK: %[[UB:.*]] = ub.poison : vector<4xf32>
//
//       CHECK: %[[EX0:.*]] = vector.extract %[[A]][0] : vector<2xf32> from vector<2x2xf32>
//       CHECK: %[[IN0:.*]] = vector.insert_strided_slice %[[EX0]], %[[UB]]
//  CHECK-SAME:    offsets = [0], strides = [1] : vector<2xf32> into vector<4xf32>
//
//       CHECK: %[[EX1:.*]] = vector.extract %{{.*}}[1] : vector<2xf32> from vector<2x2xf32>
//       CHECK: %[[IN2:.*]] = vector.insert_strided_slice %[[EX1]], %[[IN0]]
//  CHECK-SAME:    offsets = [2], strides = [1] : vector<2xf32> into vector<4xf32>
//       CHECK: return %[[IN2]] : vector<4xf32>
func.func @shape_cast_2d1d(%a: vector<2x2xf32>) -> (vector<4xf32>) {
  %0 = vector.shape_cast %a : vector<2x2xf32> to vector<4xf32>
  return %0 : vector<4xf32>
}

// Collapse 3-D to 1-D.
// CHECK-LABEL: func @shape_cast_3d1d
//  CHECK-SAME:    %[[A:.*]]: vector<1x3x2xf32>
//       CHECK: %[[UB:.*]] = ub.poison : vector<6xf32>
//
//       CHECK: %[[T0:.*]] = vector.extract %[[A]][0, 0] : vector<2xf32> from vector<1x3x2xf32>
//       CHECK: %[[T1:.*]] = vector.insert_strided_slice %[[T0]], %[[UB]]
//  CHECK-SAME:    offsets = [0], strides = [1] : vector<2xf32> into vector<6xf32>
//
//       CHECK: %[[T2:.*]] = vector.extract %[[A]][0, 1] : vector<2xf32> from vector<1x3x2xf32>
//       CHECK: %[[T3:.*]] = vector.insert_strided_slice %[[T2]], %[[T1]]
//  CHECK-SAME:    offsets = [2], strides = [1] : vector<2xf32> into vector<6xf32>
//
//       CHECK: %[[T4:.*]] = vector.extract %[[A]][0, 2] : vector<2xf32> from vector<1x3x2xf32>
//       CHECK: %[[T5:.*]] = vector.insert_strided_slice %[[T4]], %[[T3]]
//  CHECK-SAME:    offsets = [4], strides = [1] : vector<2xf32> into vector<6xf32>
//       CHECK: return %[[T5]] : vector<6xf32>
func.func @shape_cast_3d1d(%arg0 : vector<1x3x2xf32>) -> vector<6xf32> {
  %s = vector.shape_cast %arg0 : vector<1x3x2xf32> to vector<6xf32>
  return %s : vector<6xf32>
}

// Expand 1-D to 2-D.
// CHECK-LABEL: func.func @shape_cast_1d2d(
//  CHECK-SAME:    %[[A:.*]]: vector<4xf32>) -> vector<2x2xf32> {
//       CHECK: %[[UB:.*]] = ub.poison : vector<2x2xf32>
//
//       CHECK: %[[SS0:.*]] = vector.extract_strided_slice %[[A]]
//  CHECK-SAME:    offsets = [0], sizes = [2], strides = [1] :
//  CHECK-SAME:    vector<4xf32> to vector<2xf32>
//       CHECK: %[[res0:.*]] = vector.insert %[[SS0]], %[[UB]] [0] :
//  CHECK-SAME:    vector<2xf32> into vector<2x2xf32>
//
//       CHECK: %[[SS2:.*]] = vector.extract_strided_slice %[[A]]
//  CHECK-SAME:    offsets = [2], sizes = [2], strides = [1] :
//  CHECK-SAME:    vector<4xf32> to vector<2xf32>
//       CHECK: %[[res1:.*]] = vector.insert %[[SS2]], %[[res0]] [1] :
//  CHECK-SAME:    vector<2xf32> into vector<2x2xf32>
//       CHECK: return  %[[res1]] :  vector<2x2xf32>
func.func @shape_cast_1d2d(%a: vector<4xf32>) -> (vector<2x2xf32>) {
  %1 = vector.shape_cast %a: vector<4xf32> to vector<2x2xf32>
  return %1 : vector<2x2xf32>
}

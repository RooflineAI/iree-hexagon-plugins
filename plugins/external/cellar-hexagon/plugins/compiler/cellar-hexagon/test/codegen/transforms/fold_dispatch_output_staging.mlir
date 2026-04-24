// RUN: iree-opt --iree-load-plugin=cellar_hexagon=$CELLAR_HEXAGON_COMPILER_PLUGIN \
// RUN:   --pass-pipeline='builtin.module(func.func(iree-hexagon-fold-dispatch-output-staging))' \
// RUN:   %s | FileCheck %s

func.func @folds_memory_space_zero(%target: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<4x4xf32>>) {
  %src = arith.constant dense<0.0> : tensor<4x4xf32>
  %staged = bufferization.alloc_tensor() copy(%src) {memory_space = 0 : i64} : tensor<4x4xf32>
  iree_tensor_ext.dispatch.tensor.store %staged, %target, offsets = [0, 0], sizes = [4, 4], strides = [1, 1] : tensor<4x4xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<4x4xf32>>
  return
}

// CHECK-LABEL: func.func @folds_memory_space_zero(
// CHECK-NOT: bufferization.alloc_tensor() copy
// CHECK: iree_tensor_ext.dispatch.tensor.store %[[SRC:.*]], %[[TARGET:.*]], offsets = [0, 0], sizes = [4, 4], strides = [1, 1] : tensor<4x4xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<4x4xf32>>

func.func @keeps_nonzero_memory_space(%target: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<4x4xf32>>) {
  %src = arith.constant dense<1.0> : tensor<4x4xf32>
  %staged = bufferization.alloc_tensor() copy(%src) {memory_space = 1 : i64} : tensor<4x4xf32>
  iree_tensor_ext.dispatch.tensor.store %staged, %target, offsets = [0, 0], sizes = [4, 4], strides = [1, 1] : tensor<4x4xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<4x4xf32>>
  return
}

// CHECK-LABEL: func.func @keeps_nonzero_memory_space(
// CHECK: %[[SRC:.*]] = arith.constant dense<1.000000e+00> : tensor<4x4xf32>
// CHECK: %[[STAGED:.*]] = bufferization.alloc_tensor() copy(%[[SRC]]) {memory_space = 1 : i64} : tensor<4x4xf32>
// CHECK: iree_tensor_ext.dispatch.tensor.store %[[STAGED]], %[[TARGET:.*]], offsets = [0, 0], sizes = [4, 4], strides = [1, 1] : tensor<4x4xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<4x4xf32>>

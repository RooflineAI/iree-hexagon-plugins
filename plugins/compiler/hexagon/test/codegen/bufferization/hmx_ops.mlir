// RUN: iree-opt --iree-load-plugin=hexagon=$ROOF_HEXAGON_COMPILER_PLUGIN \
// RUN:   --one-shot-bufferize='bufferize-function-boundaries' \
// RUN:   --split-input-file %s | FileCheck %s

// Check the tensor-to-buffer boundary owned by this branch. Later branches
// separately test expansion and runtime-call lowering of the buffer ops.

// CHECK-LABEL: func.func @bufferize_hmx_chain(
// CHECK-SAME: %[[SOURCE:[^:]+]]: memref<32x32xf16
// CHECK-SAME: %[[LHS:[^:]+]]: memref<1x1x16x32x2xf16
// CHECK-SAME: %[[RHS:[^:]+]]: memref<1x1x16x32x2xf16
// CHECK-SAME: %[[PACKED:[^:]+]]: memref<1x1x16x32x2xf16
// CHECK-SAME: %[[RESULT:[^:]+]]: memref<32x32xf32
// CHECK: iree_hexagon.hmx.pack
// CHECK-SAME: ins(%[[SOURCE]]
// CHECK-SAME: outs(%[[PACKED]]
// CHECK: iree_hexagon.hmx.matmul
// CHECK-SAME: ins(%[[LHS]], %[[RHS]]
// CHECK-SAME: outs(%[[PACKED]]
// CHECK: iree_hexagon.hmx.unpack
// CHECK-SAME: ins(%[[PACKED]]
// CHECK-SAME: outs(%[[RESULT]]
// CHECK: return %[[RESULT]]
func.func @bufferize_hmx_chain(
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

// -----

// Returning both the destination's old tensor value and the HMX result forces
// one-shot bufferization to allocate a separate result buffer. Pack overwrites
// every physical tile, so the old destination contents must not be copied.
// CHECK-LABEL: func.func @bufferize_write_only_pack_dest(
// CHECK:       %[[ALLOC:.+]] = memref.alloc
// CHECK-NOT:   memref.copy
// CHECK:       iree_hexagon.hmx.pack
// CHECK-SAME:  outs(%[[ALLOC]]
func.func @bufferize_write_only_pack_dest(
    %source: tensor<32x32xf16>,
    %dest: tensor<1x1x16x32x2xf16>)
    -> (tensor<1x1x16x32x2xf16>, tensor<1x1x16x32x2xf16>) {
  %packed = iree_hexagon.hmx.tensor_pack
      ins(%source : tensor<32x32xf16>)
      outs(%dest : tensor<1x1x16x32x2xf16>)
      {dim = 0 : i64} -> tensor<1x1x16x32x2xf16>
  return %packed, %dest
      : tensor<1x1x16x32x2xf16>, tensor<1x1x16x32x2xf16>
}

// -----

// Matmul's accumulator-read tile is produced from register state and likewise
// overwrites its destination instead of loading the old f16 tile contents.
// CHECK-LABEL: func.func @bufferize_write_only_matmul_dest(
// CHECK:       %[[ALLOC:.+]] = memref.alloc
// CHECK-NOT:   memref.copy
// CHECK:       iree_hexagon.hmx.matmul
// CHECK-SAME:  outs(%[[ALLOC]]
func.func @bufferize_write_only_matmul_dest(
    %lhs: tensor<1x1x16x32x2xf16>,
    %rhs: tensor<1x1x16x32x2xf16>,
    %dest: tensor<1x1x16x32x2xf16>)
    -> (tensor<1x1x16x32x2xf16>, tensor<1x1x16x32x2xf16>) {
  %product = iree_hexagon.hmx.tensor_matmul
      ins(%lhs, %rhs : tensor<1x1x16x32x2xf16>,
                       tensor<1x1x16x32x2xf16>)
      outs(%dest : tensor<1x1x16x32x2xf16>)
      -> tensor<1x1x16x32x2xf16>
  return %product, %dest
      : tensor<1x1x16x32x2xf16>, tensor<1x1x16x32x2xf16>
}

// -----

// Unpack performs destination-style accumulation. When preserving the old
// initializer forces an out-of-place result, its contents must be copied into
// the new buffer before the runtime unpack updates it.
// CHECK-LABEL: func.func @bufferize_read_write_unpack_dest(
// CHECK:       %[[ALLOC:.+]] = memref.alloc
// CHECK-NEXT:  memref.copy %{{.+}}, %[[ALLOC]]
// CHECK-NEXT:  iree_hexagon.hmx.unpack
// CHECK-SAME:  outs(%[[ALLOC]]
func.func @bufferize_read_write_unpack_dest(
    %source: tensor<1x1x16x32x2xf16>, %dest: tensor<32x32xf32>)
    -> (tensor<32x32xf32>, tensor<32x32xf32>) {
  %unpacked = iree_hexagon.hmx.tensor_unpack
      ins(%source : tensor<1x1x16x32x2xf16>)
      outs(%dest : tensor<32x32xf32>)
      {dim = 0 : i64} -> tensor<32x32xf32>
  return %unpacked, %dest : tensor<32x32xf32>, tensor<32x32xf32>
}

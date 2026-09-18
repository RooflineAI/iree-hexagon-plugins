// RUN: iree-opt --iree-load-plugin=hexagon=$ROOF_HEXAGON_COMPILER_PLUGIN \
// RUN:   --iree-hexagon-mark-native-runtime-links %s | FileCheck %s

// Every call emitted by HexagonLowerHmxToCalls must remain a native unresolved
// reference for the DSP loader instead of becoming a HAL import thunk.
// CHECK-DAG: llvm.func @iree_hexagon_hmx_acc_setup_read_f16(i32) attributes {hexagon.native_runtime_link}
// CHECK-DAG: llvm.func @iree_hexagon_hmx_acc_clear_f16() attributes {hexagon.native_runtime_link}
// CHECK-DAG: llvm.func @iree_hexagon_hmx_pack_f16(i32, i32, i32, i32, i32, i32, i32) attributes {hexagon.native_runtime_link}
// CHECK-DAG: llvm.func @iree_hexagon_hmx_pack_transposed_f16(i32, i32, i32, i32, i32, i32, i32) attributes {hexagon.native_runtime_link}
// CHECK-DAG: llvm.func @iree_hexagon_hmx_unpack_acc_f16_to_f32(i32, i32, i32, i32, i32, i32, i32) attributes {hexagon.native_runtime_link}
// CHECK-DAG: llvm.func @iree_hexagon_hmx_unpack_acc_f16_to_f16(i32, i32, i32, i32, i32, i32, i32) attributes {hexagon.native_runtime_link}
// CHECK-DAG: llvm.func @iree_hexagon_hmx_mma_f16(i32, i32) attributes {hexagon.native_runtime_link}
// CHECK-DAG: llvm.func @iree_hexagon_hmx_acc_read_f16(i32) attributes {hexagon.native_runtime_link}

module {
  llvm.func @iree_hexagon_hmx_acc_setup_read_f16(i32)
  llvm.func @iree_hexagon_hmx_acc_clear_f16()
  llvm.func @iree_hexagon_hmx_pack_f16(i32, i32, i32, i32, i32, i32, i32)
  llvm.func @iree_hexagon_hmx_pack_transposed_f16(i32, i32, i32, i32, i32, i32, i32)
  llvm.func @iree_hexagon_hmx_unpack_acc_f16_to_f32(i32, i32, i32, i32, i32, i32, i32)
  llvm.func @iree_hexagon_hmx_unpack_acc_f16_to_f16(i32, i32, i32, i32, i32, i32, i32)
  llvm.func @iree_hexagon_hmx_mma_f16(i32, i32)
  llvm.func @iree_hexagon_hmx_acc_read_f16(i32)
}

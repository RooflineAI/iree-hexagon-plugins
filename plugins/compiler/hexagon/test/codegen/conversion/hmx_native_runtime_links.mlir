// RUN: iree-opt \
// RUN:   --pass-pipeline='builtin.module(iree-hexagon-convert-to-llvm)' \
// RUN:   %s | FileCheck %s

// Every runtime function called by HexagonLowerHmxToCalls must be classified as
// a native DSP runtime symbol: tagged `hal.import.static` and left unresolved
// for the DSP loader. An unclassified declaration would instead be rejected by
// the external-call validation.
// CHECK-DAG: llvm.func @iree_hexagon_hmx_acc_setup_read_f16(!llvm.ptr) attributes {hal.import.static}
// CHECK-DAG: llvm.func @iree_hexagon_hmx_acc_clear_f16() attributes {hal.import.static}
// CHECK-DAG: llvm.func @iree_hexagon_hmx_pack_f16(!llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32) attributes {hal.import.static}
// CHECK-DAG: llvm.func @iree_hexagon_hmx_pack_transposed_f16(!llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32) attributes {hal.import.static}
// CHECK-DAG: llvm.func @iree_hexagon_hmx_unpack_acc_f16_to_f32(!llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32) attributes {hal.import.static}
// CHECK-DAG: llvm.func @iree_hexagon_hmx_unpack_acc_f16_to_f16(!llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32) attributes {hal.import.static}
// CHECK-DAG: llvm.func @iree_hexagon_hmx_mma_f16(!llvm.ptr, !llvm.ptr) attributes {hal.import.static}
// CHECK-DAG: llvm.func @iree_hexagon_hmx_acc_read_f16(!llvm.ptr) attributes {hal.import.static}

module {
  llvm.func @iree_hexagon_hmx_acc_setup_read_f16(!llvm.ptr)
  llvm.func @iree_hexagon_hmx_acc_clear_f16()
  llvm.func @iree_hexagon_hmx_pack_f16(!llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32)
  llvm.func @iree_hexagon_hmx_pack_transposed_f16(!llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32)
  llvm.func @iree_hexagon_hmx_unpack_acc_f16_to_f32(!llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32)
  llvm.func @iree_hexagon_hmx_unpack_acc_f16_to_f16(!llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32)
  llvm.func @iree_hexagon_hmx_mma_f16(!llvm.ptr, !llvm.ptr)
  llvm.func @iree_hexagon_hmx_acc_read_f16(!llvm.ptr)
}

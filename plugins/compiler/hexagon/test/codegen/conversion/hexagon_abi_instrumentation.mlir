// This file checks every local-executable dispatch instrumentation conversion:
// workgroup, scalar value, memory load, and memory store records. It locks down
// ring-buffer addressing, record sizes/header encodings, and value preservation.
// RUN: iree-opt --verify-diagnostics \
// RUN:   --pass-pipeline='builtin.module(iree-hexagon-convert-to-llvm)' \
// RUN:   %s | FileCheck %s

#hexagon_target = #hal.executable.target<"hexagon", "embedded-elf-hexagon", {
  data_layout = "e-m:e-p:32:32:32-a:0-n16:32-i64:64:64-i32:32:32-i16:16:16-i1:8:8-f32:32:32-f64:64:64-v32:32:32-v64:64:64-v512:512:512-v1024:1024:1024-v2048:2048:2048",
  target_triple = "hexagon-unknown-unknown-elf"
}>

#instrument_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>

module attributes {hal.executable.target = #hexagon_target} {
  // Verifies all four supported instrumentation operations append correctly
  // encoded, 16-byte-aligned records and return their original SSA values.
  // CHECK-LABEL: llvm.func @instrument_supported(
  // CHECK: %[[DISPATCH_ID:.+]] = llvm.mlir.constant(7 : i32) : i32
  // CHECK: %[[KEY_SHIFT:.+]] = llvm.mlir.constant(8 : i32) : i32
  // CHECK: llvm.shl %[[DISPATCH_ID]], %[[KEY_SHIFT]] : i32
  // CHECK: llvm.getelementptr inbounds {{.+}}[{{.+}}] : (!llvm.ptr, i64) -> !llvm.ptr, i8
  // CHECK: llvm.atomicrmw add {{.+}}, {{.+}} monotonic
  // CHECK: llvm.store {{.+}} {alignment = 16 : i64} : !llvm.struct<(i32, i32, i32, i32, i32, i32, i32, i32)>, !llvm.ptr
  // CHECK: llvm.shl {{.+}}, {{.+}} : i64
  // CHECK: llvm.mlir.constant(197634 : i64) : i64
  // CHECK: llvm.atomicrmw add {{.+}}, {{.+}} monotonic
  // CHECK: llvm.store {{.+}} {alignment = 16 : i64} : !llvm.struct<(i64, i64)>, !llvm.ptr
  // CHECK: llvm.call @sink_i32(%[[SCALAR:.+]])
  // CHECK: %[[LOADED:.+]] = llvm.load {{.+}} : !llvm.ptr -> f32
  // CHECK: llvm.mlir.constant(1028 : i64) : i64
  // CHECK: llvm.ptrtoint {{.+}} : !llvm.ptr to i64
  // CHECK: llvm.atomicrmw add {{.+}}, {{.+}} monotonic
  // CHECK: llvm.store {{.+}} {alignment = 16 : i64} : !llvm.struct<(i64, i64)>, !llvm.ptr
  // CHECK: llvm.mlir.constant(1029 : i64) : i64
  // CHECK: llvm.ptrtoint {{.+}} : !llvm.ptr to i64
  // CHECK: llvm.atomicrmw add {{.+}}, {{.+}} monotonic
  // CHECK: llvm.store {{.+}} {alignment = 16 : i64} : !llvm.struct<(i64, i64)>, !llvm.ptr
  // CHECK: llvm.store %[[LOADED]], {{.+}} : f32, !llvm.ptr
  func.func @instrument_supported() {
    %c0 = arith.constant 0 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %dispatch_id = arith.constant 7 : i32
    %scalar = arith.constant 42 : i32
    %buffer = hal.interface.binding.subspan layout(#instrument_layout)
        binding(0) offset(%c0) : memref<67112960xi8>
    %data = hal.interface.binding.subspan layout(#instrument_layout)
        binding(1) offset(%c0) : memref<16xf32>

    // Checks the 32-byte workgroup record and 40-bit workgroup key.
    %key = hal.instrument.workgroup[%buffer : memref<67112960xi8>]
        dispatch(%dispatch_id) : index

    // Checks signed-i32 value type 4, ordinal 3, and preservation of %scalar.
    %observed = hal.instrument.value[
        %buffer : memref<67112960xi8> for %key] 3 : i8 = %scalar : i32
    func.call @sink_i32(%observed) : (i32) -> ()

    // Checks a four-byte memory-load record and preservation of the loaded f32.
    %loaded = memref.load %data[%c2] : memref<16xf32>
    %observed_load = hal.instrument.memory.load[
        %buffer : memref<67112960xi8> for %key]
        %data[%c2], %loaded : memref<16xf32>, f32

    // Checks a four-byte memory-store record and preservation of its f32 value.
    %observed_store = hal.instrument.memory.store[
        %buffer : memref<67112960xi8> for %key]
        %data[%c3], %observed_load : memref<16xf32>, f32
    memref.store %observed_store, %data[%c3] : memref<16xf32>
    return
  }

  // Verifies unsupported vector values are preserved and skipped without
  // writing an instrumentation record.
  // CHECK-LABEL: llvm.func @instrument_unsupported(
  // CHECK-NOT: llvm.atomicrmw
  // CHECK: llvm.call @sink_vector
  func.func @instrument_unsupported() {
    %c0 = arith.constant 0 : index
    %key = arith.constant 0 : index
    %value = arith.constant dense<1.0> : vector<2xf32>
    %buffer = hal.interface.binding.subspan layout(#instrument_layout)
        binding(0) offset(%c0) : memref<67112960xi8>
    // expected-warning@+1 {{skipping hal.instrument.value on unsupported type: 'vector<2xf32>'}}
    %observed = hal.instrument.value[
        %buffer : memref<67112960xi8> for %key] 0 : i8 = %value : vector<2xf32>
    func.call @sink_vector(%observed) : (vector<2xf32>) -> ()
    return
  }

  func.func private @sink_i32(%value: i32) {
    return
  }
  func.func private @sink_vector(%value: vector<2xf32>) {
    return
  }
}

// CHECK-NOT: hal.instrument
// CHECK-NOT: builtin.unrealized_conversion_cast

// This file checks construction of LLVM memref descriptors from IREE HAL
// bindings. It covers binding ordinals, byte offsets, static and dynamic
// shapes/strides, and sub-byte element types under Hexagon's 32-bit layout.
// RUN: iree-opt \
// RUN:   --pass-pipeline='builtin.module(iree-hexagon-convert-to-llvm)' \
// RUN:   %s | FileCheck %s

#hexagon_target = #hal.executable.target<"hexagon", "embedded-elf-hexagon", {
  data_layout = "e-m:e-p:32:32:32-a:0-n16:32-i64:64:64-i32:32:32-i16:16:16-i1:8:8-f32:32:32-f64:64:64-v32:32:32-v64:64:64-v512:512:512-v1024:1024:1024-v2048:2048:2048",
  target_triple = "hexagon-unknown-unknown-elf"
}>

#bindings = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
#bindings_with_constants = #hal.pipeline.layout<constants = 4, bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>

module attributes {hal.executable.target = #hexagon_target} {
  // Verifies binding ordinal 1, a constant 72-byte HAL subspan offset, and a
  // strided memref layout whose element offset is applied after the byte offset.
  // CHECK-LABEL: llvm.func @binding_static_offset(
  // CHECK: %[[STATE:.+]] = llvm.load %arg1
  // CHECK: %[[BINDINGS:.+]] = llvm.extractvalue %[[STATE]][10]
  // CHECK: %[[SLOT:.+]] = llvm.getelementptr %[[BINDINGS]][1] : (!llvm.ptr) -> !llvm.ptr, !llvm.ptr
  // CHECK: %[[BASE:.+]] = llvm.load %[[SLOT]] : !llvm.ptr -> !llvm.ptr
  // CHECK: %[[BYTE_BASE:.+]] = llvm.getelementptr %[[BASE]][72] : (!llvm.ptr) -> !llvm.ptr, i8
  // CHECK: %[[ELEMENT_BASE:.+]] = llvm.getelementptr %[[BYTE_BASE]][18]
  // CHECK: %[[ROW:.+]] = llvm.mul {{.+}}, {{.+}}
  // CHECK: %[[INDEX:.+]] = llvm.add %[[ROW]], {{.+}}
  // CHECK: %[[ELEMENT:.+]] = llvm.getelementptr {{.*}} %[[ELEMENT_BASE]][%[[INDEX]]]
  // CHECK: %[[VALUE:.+]] = llvm.load %[[ELEMENT]]
  func.func @binding_static_offset() {
    %c72 = arith.constant 72 : index
    %c128 = arith.constant 128 : index
    %memref = hal.interface.binding.subspan layout(#bindings) binding(1)
        offset(%c72) : memref<?x2xf32, strided<[2, 1], offset: 18>>{%c128}
    %c1 = arith.constant 1 : index
    %c5 = arith.constant 5 : index
    %value = memref.load %memref[%c5, %c1]
        : memref<?x2xf32, strided<[2, 1], offset: 18>>
    func.call @sink_f32(%value) : (f32) -> ()
    return
  }

  // Verifies a dynamic HAL byte offset is applied to a static-shape memref
  // before the descriptor is used for element addressing.
  // CHECK-LABEL: llvm.func @binding_static_shape_dynamic_offset(
  // CHECK: %[[CONSTANT_STATE:.+]] = llvm.load %arg1
  // CHECK: %[[CONSTANTS:.+]] = llvm.extractvalue %[[CONSTANT_STATE]][9]
  // CHECK: %[[OFFSET32:.+]] = llvm.load %[[CONSTANTS]] : !llvm.ptr -> i32
  // CHECK: %[[OFFSET:.+]] = llvm.zext %[[OFFSET32]] : i32 to i64
  // CHECK: %[[BINDING_PTRS:.+]] = llvm.extractvalue {{.+}}[10]
  // CHECK: %[[SLOT:.+]] = llvm.getelementptr %[[BINDING_PTRS]][1]
  // CHECK: %[[BASE:.+]] = llvm.load %[[SLOT]] : !llvm.ptr -> !llvm.ptr
  // CHECK: %[[BYTE_BASE:.+]] = llvm.getelementptr %[[BASE]][%[[OFFSET]]] : (!llvm.ptr, i64) -> !llvm.ptr, i8
  // CHECK: %[[ELEMENT:.+]] = llvm.getelementptr {{.*}} %[[BYTE_BASE]][3]
  // CHECK: %[[VALUE:.+]] = llvm.load %[[ELEMENT]]
  func.func @binding_static_shape_dynamic_offset() {
    %offset = hal.interface.constant.load layout(#bindings_with_constants)
        ordinal(0) : index
    %memref = hal.interface.binding.subspan layout(#bindings_with_constants)
        binding(1) offset(%offset) : memref<8xf32>
    %c3 = arith.constant 3 : index
    %value = memref.load %memref[%c3] : memref<8xf32>
    func.call @sink_f32(%value) : (f32) -> ()
    return
  }

  // Verifies dynamic dimensions produce a zero descriptor offset and
  // row-major dynamic strides while retaining the dynamic HAL byte offset.
  // CHECK-LABEL: llvm.func @binding_dynamic_shape(
  // CHECK: %[[BINDING_PTRS:.+]] = llvm.extractvalue {{.+}}[10]
  // CHECK: %[[SLOT:.+]] = llvm.getelementptr %[[BINDING_PTRS]][1]
  // CHECK: %[[BASE:.+]] = llvm.load %[[SLOT]] : !llvm.ptr -> !llvm.ptr
  // CHECK: %[[BYTE_BASE:.+]] = llvm.getelementptr %[[BASE]][{{.+}}] : (!llvm.ptr, i64) -> !llvm.ptr, i8
  // CHECK: %[[STRIDE1:.+]] = llvm.mul {{.+}}, {{.+}}
  // CHECK: %[[STRIDE0:.+]] = llvm.mul %[[STRIDE1]], {{.+}}
  // CHECK: %[[INDEX2:.+]] = llvm.mul %[[STRIDE0]], {{.+}}
  // CHECK: %[[INDEX1:.+]] = llvm.mul %[[STRIDE1]], {{.+}}
  // CHECK: %[[PARTIAL:.+]] = llvm.add %[[INDEX2]], %[[INDEX1]]
  // CHECK: %[[INDEX:.+]] = llvm.add %[[PARTIAL]], {{.+}}
  // CHECK: %[[ELEMENT:.+]] = llvm.getelementptr {{.*}} %[[BYTE_BASE]][%[[INDEX]]]
  func.func @binding_dynamic_shape() {
    %offset = hal.interface.constant.load layout(#bindings_with_constants)
        ordinal(0) : index
    %dim0 = hal.interface.constant.load layout(#bindings_with_constants)
        ordinal(1) : index
    %dim1 = hal.interface.constant.load layout(#bindings_with_constants)
        ordinal(2) : index
    %dim2 = hal.interface.constant.load layout(#bindings_with_constants)
        ordinal(3) : index
    %memref = hal.interface.binding.subspan layout(#bindings_with_constants)
        binding(1) offset(%offset)
        : memref<?x?x?xf32, strided<[?, ?, 1], offset: ?>>{%dim0, %dim1, %dim2}
    %c3 = arith.constant 3 : index
    %c5 = arith.constant 5 : index
    %c7 = arith.constant 7 : index
    %value = memref.load %memref[%c7, %c5, %c3]
        : memref<?x?x?xf32, strided<[?, ?, 1], offset: ?>>
    func.call @sink_f32(%value) : (f32) -> ()
    return
  }

  // Verifies a dynamic subspan offset remains byte-granular for i4 elements;
  // the element index is applied only after the byte pointer is constructed.
  // CHECK-LABEL: llvm.func @binding_sub_byte(
  // CHECK: %[[BINDING_PTRS:.+]] = llvm.extractvalue {{.+}}[10]
  // CHECK: %[[SLOT:.+]] = llvm.getelementptr %[[BINDING_PTRS]][1]
  // CHECK: %[[BASE:.+]] = llvm.load %[[SLOT]] : !llvm.ptr -> !llvm.ptr
  // CHECK: %[[BYTE_BASE:.+]] = llvm.getelementptr %[[BASE]][{{.+}}] : (!llvm.ptr, i64) -> !llvm.ptr, i8
  // CHECK: %[[ELEMENT:.+]] = llvm.getelementptr {{.*}} %[[BYTE_BASE]][7]
  // CHECK: %[[VALUE:.+]] = llvm.load %[[ELEMENT]]
  func.func @binding_sub_byte() {
    %offset = hal.interface.constant.load layout(#bindings_with_constants)
        ordinal(0) : index
    %dim0 = hal.interface.constant.load layout(#bindings_with_constants)
        ordinal(1) : index
    %memref = hal.interface.binding.subspan layout(#bindings_with_constants)
        binding(1) offset(%offset)
        : memref<?xi4, strided<[1], offset: ?>>{%dim0}
    %c7 = arith.constant 7 : index
    %value = memref.load %memref[%c7]
        : memref<?xi4, strided<[1], offset: ?>>
    func.call @sink_i4(%value) : (i4) -> ()
    return
  }

  func.func private @sink_f32(%value: f32) {
    return
  }
  func.func private @sink_i4(%value: i4) {
    return
  }
}

// CHECK-NOT: hal.interface
// CHECK-NOT: builtin.unrealized_conversion_cast

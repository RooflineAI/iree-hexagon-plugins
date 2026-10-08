// This file checks the local-executable entry ABI, workgroup and constant
// accessors, and Hexagon's extended runtime-state access. It uses the production
// 32-bit Hexagon data layout so pointer-sensitive ABI changes are observable.
// RUN: iree-opt \
// RUN:   --pass-pipeline='builtin.module(iree-hexagon-convert-to-llvm)' \
// RUN:   %s | FileCheck %s

#hexagon_target = #hal.executable.target<"hexagon", "embedded-elf-hexagon", {
  data_layout = "e-m:e-p:32:32:32-a:0-n16:32-i64:64:64-i32:32:32-i16:16:16-i1:8:8-f32:32:32-f64:64:64-v32:32:32-v64:64:64-v512:512:512-v1024:1024:1024-v2048:2048:2048",
  target_triple = "hexagon-unknown-unknown-elf"
}>

#pipeline_layout = #hal.pipeline.layout<constants = 2, bindings = [
  #hal.pipeline.binding<storage_buffer>
]>

module attributes {hal.executable.target = #hexagon_target} {
  // Verifies that a public empty function becomes the three-argument IREE
  // dispatch ABI and returns status zero.
  // CHECK: module attributes {
  // CHECK-SAME: llvm.data_layout = "e-m:e-p:32:32:32-a:0-n16:32
  // CHECK-SAME: llvm.target_triple = "hexagon-unknown-unknown-elf"
  // CHECK-DAG: llvm.mlir.global internal @__constant_ordinal_foo() {{.+}}hal.executable.constant.key = "foo"{{.+}} : i32
  // CHECK-LABEL: llvm.func @entry(
  // CHECK-SAME: %[[ENV:[A-Za-z0-9_]+]]: !llvm.ptr {llvm.align = 16 : i64, llvm.noalias, llvm.nonnull, llvm.noundef},
  // CHECK-SAME: %[[DISPATCH:[A-Za-z0-9_]+]]: !llvm.ptr {llvm.align = 16 : i64, llvm.noalias, llvm.nonnull, llvm.noundef},
  // CHECK-SAME: %[[WORKGROUP:[A-Za-z0-9_]+]]: !llvm.ptr {llvm.align = 16 : i64, llvm.noalias, llvm.nonnull, llvm.noundef}) -> i32
  // CHECK-SAME: attributes {llvm.emit_c_interface}
  func.func @entry() attributes {llvm.emit_c_interface} {
    // Verifies workgroup ID x/y/z field selection from the workgroup-state
    // argument. With the 32-bit index, i32 fields are used as-is and only the
    // i16 field is extended.
    // CHECK: %[[ID_X_STATE:.+]] = llvm.load %[[WORKGROUP]]
    // CHECK: %[[ID_X_32:.+]] = llvm.extractvalue %[[ID_X_STATE]][0]
    %id_x = hal.interface.workgroup.id[0] : index
    // CHECK: %[[ID_Y_STATE:.+]] = llvm.load %[[WORKGROUP]]
    // CHECK: %[[ID_Y_32:.+]] = llvm.extractvalue %[[ID_Y_STATE]][1]
    %id_y = hal.interface.workgroup.id[1] : index
    // CHECK: %[[ID_Z_STATE:.+]] = llvm.load %[[WORKGROUP]]
    // CHECK: %[[ID_Z_16:.+]] = llvm.extractvalue %[[ID_Z_STATE]][2]
    // CHECK: %[[ID_Z:.+]] = llvm.zext %[[ID_Z_16]] : i16 to i32
    %id_z = hal.interface.workgroup.id[2] : index

    // Verifies workgroup size x/y/z field selection from the dispatch-state
    // argument, including the narrower z field.
    // CHECK: %[[SIZE_X_STATE:.+]] = llvm.load %[[DISPATCH]]
    // CHECK: %[[SIZE_X_32:.+]] = llvm.extractvalue %[[SIZE_X_STATE]][0]
    %size_x = hal.interface.workgroup.size[0] : index
    // CHECK: %[[SIZE_Y_STATE:.+]] = llvm.load %[[DISPATCH]]
    // CHECK: %[[SIZE_Y_32:.+]] = llvm.extractvalue %[[SIZE_Y_STATE]][1]
    %size_y = hal.interface.workgroup.size[1] : index
    // CHECK: %[[SIZE_Z_STATE:.+]] = llvm.load %[[DISPATCH]]
    // CHECK: %[[SIZE_Z_16:.+]] = llvm.extractvalue %[[SIZE_Z_STATE]][2]
    // CHECK: %[[SIZE_Z:.+]] = llvm.zext %[[SIZE_Z_16]] : i16 to i32
    %size_z = hal.interface.workgroup.size[2] : index

    // Verifies workgroup count x/y/z use dispatch-state fields 4/5/6.
    // CHECK: %[[COUNT_X_STATE:.+]] = llvm.load %[[DISPATCH]]
    // CHECK: %[[COUNT_X_32:.+]] = llvm.extractvalue %[[COUNT_X_STATE]][4]
    %count_x = hal.interface.workgroup.count[0] : index
    // CHECK: %[[COUNT_Y_STATE:.+]] = llvm.load %[[DISPATCH]]
    // CHECK: %[[COUNT_Y_32:.+]] = llvm.extractvalue %[[COUNT_Y_STATE]][5]
    %count_y = hal.interface.workgroup.count[1] : index
    // CHECK: %[[COUNT_Z_STATE:.+]] = llvm.load %[[DISPATCH]]
    // CHECK: %[[COUNT_Z_16:.+]] = llvm.extractvalue %[[COUNT_Z_STATE]][6]
    // CHECK: %[[COUNT_Z:.+]] = llvm.zext %[[COUNT_Z_16]] : i16 to i32
    %count_z = hal.interface.workgroup.count[2] : index

    // Verifies push constant ordinal 1 is loaded as i32 from dispatch-state
    // field 9 and extended to the configured 64-bit index representation.
    // CHECK: %[[CONSTANT_STATE:.+]] = llvm.load %[[DISPATCH]]
    // CHECK: %[[CONSTANT_BASE:.+]] = llvm.extractvalue %[[CONSTANT_STATE]][9]
    // CHECK: %[[CONSTANT_PTR:.+]] = llvm.getelementptr %[[CONSTANT_BASE]][1]
    // CHECK: %[[CONSTANT_32:.+]] = llvm.load %[[CONSTANT_PTR]] : !llvm.ptr -> i32
    %constant = hal.interface.constant.load layout(#pipeline_layout) ordinal(1) : index

    // Verifies an executable constant loads its linked ordinal and then indexes
    // environment field 0 before loading the i32 value.
    // CHECK: %[[ORDINAL_ADDR:.+]] = llvm.mlir.addressof @__constant_ordinal_foo
    // CHECK: %[[ORDINAL:.+]] = llvm.load %[[ORDINAL_ADDR]] : !llvm.ptr -> i32
    // CHECK: %[[ENV_STATE:.+]] = llvm.load %[[ENV]]
    // CHECK: %[[CONSTANTS:.+]] = llvm.extractvalue %[[ENV_STATE]][0]
    // CHECK: %[[EXEC_CONSTANT_PTR:.+]] = llvm.getelementptr %[[CONSTANTS]][%[[ORDINAL]]]
    // CHECK: %[[EXEC_CONSTANT:.+]] = llvm.load %[[EXEC_CONSTANT_PTR]] : !llvm.ptr -> i32
    %executable_constant = hal.executable.constant.load "foo" : i32

    // Verifies the Hexagon runtime pointer is the trailing field of the
    // extended dispatch state and is loaded from ABI argument 1.
    // CHECK: %[[RUNTIME_ADDR:.+]] = llvm.getelementptr inbounds %[[DISPATCH]][0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(struct<"iree_hal_executable_dispatch_state_v0_t", {{.*}}>, ptr)>
    // CHECK: %[[RUNTIME:.+]] = llvm.load %[[RUNTIME_ADDR]] : !llvm.ptr -> !llvm.ptr
    %state = iree_hexagon.get_runtime_state : !iree_hexagon.runtime_state

    func.call @consume(%id_x, %id_y, %id_z, %size_x, %size_y, %size_z,
                       %count_x, %count_y, %count_z, %constant,
                       %executable_constant, %state)
        : (index, index, index, index, index, index, index, index, index,
           index, i32, !iree_hexagon.runtime_state) -> ()
    // CHECK: llvm.call @consume(%[[ID_X_32]], %[[ID_Y_32]], %[[ID_Z]], %[[SIZE_X_32]], %[[SIZE_Y_32]], %[[SIZE_Z]], %[[COUNT_X_32]], %[[COUNT_Y_32]], %[[COUNT_Z]], %[[CONSTANT_32]], %[[EXEC_CONSTANT]], %[[RUNTIME]]) : (i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, !llvm.ptr) -> ()
    // CHECK: %[[ZERO:.+]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK: llvm.return %[[ZERO]] : i32
    return
  }

  // Verifies non-public functions use ordinary func-to-LLVM conversion and do
  // not acquire dispatch ABI arguments or an i32 status result.
  // CHECK-LABEL: llvm.func @consume(
  // CHECK-SAME: attributes {sym_visibility = "private"}
  func.func private @consume(
      %id_x: index, %id_y: index, %id_z: index,
      %size_x: index, %size_y: index, %size_z: index,
      %count_x: index, %count_y: index, %count_z: index,
      %constant: index, %executable_constant: i32,
      %state: !iree_hexagon.runtime_state) {
    return
  }
}

// CHECK-NOT: hal.interface
// CHECK-NOT: hal.executable.constant.load
// CHECK-NOT: iree_hexagon.get_runtime_state
// CHECK-NOT: builtin.unrealized_conversion_cast

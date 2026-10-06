// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Adapted from the first Codegen/LLVMGPU/test/link_executables.mlir case
// at IREE revision a45adeaa6115e446c898e6eb21fb6edc0e65ddc4; ordinary address spaces
// and constant blocks exercise Hexagon linking and final ordinal assignment.
// RUN: iree-opt --pass-pipeline='builtin.module(iree-hexagon-link-executables{target="hexagon"})' %s | FileCheck %s
// RUN: iree-opt --pass-pipeline='builtin.module(iree-hal-link-all-executables)' %s | FileCheck %s --check-prefixes=CHECK,ORDINAL

#executable_target_hexagon = #hal.executable.target<"hexagon", "embedded-elf-hexagon">

// CHECK-LABEL: module @link_two_executables
builtin.module @link_two_executables {

// Expect a single executable with both exports and correct ordinals.
// CHECK: hal.executable private @link_two_executables_linked
// CHECK:   hal.executable.variant public @embedded_elf_hexagon
// CHECK:     hal.executable.export public @export0 ordinal(0)
// CHECK:     hal.executable.export public @export1 ordinal(1)

// Expect one LLVM module with all globals and functions.
// Shared external declarations are coalesced; private scratch symbols are renamed.
// CHECK: builtin.module
// CHECK: llvm.mlir.global external @external_data
// CHECK: llvm.mlir.global private @scratch{{.+}} : !llvm.array<2 x array<64 x i32>>
// ORDINAL: llvm.mlir.global internal constant @ordinal_foo(0 : i32)
// CHECK: llvm.func @export0
// CHECK:   llvm.mlir.addressof @external_data : !llvm.ptr
// CHECK:   llvm.mlir.addressof @scratch : !llvm.ptr
//      CHECK: llvm.mlir.global private @scratch_0{{.+}} : !llvm.array<2 x array<128 x i32>>
// ORDINAL: llvm.mlir.global internal constant @ordinal_bar(1 : i32)
// CHECK: llvm.func @export1
// CHECK:   llvm.mlir.addressof @external_data : !llvm.ptr
// CHECK:   llvm.mlir.addressof @scratch_0 : !llvm.ptr

hal.executable private @executable0 {
  hal.executable.variant public @embedded_elf_hexagon target(#executable_target_hexagon) {
    hal.executable.export public @export0 ordinal(0) layout(#hal.pipeline.layout<bindings = [#hal.pipeline.binding<storage_buffer>]>) count(%arg0: !hal.device) -> (index, index, index) {
      %c1 = arith.constant 1 : index
      hal.return %c1, %c1, %c1 : index, index, index
    }
    hal.executable.constant.block(%device: !hal.device) -> i32 as "foo" {
      %c0 = arith.constant 0 : i32
      hal.return %c0 : i32
    }
    builtin.module {
      llvm.mlir.global external @external_data() {addr_space = 0 : i32, alignment = 16 : i64} : !llvm.array<0 x i8>
      llvm.mlir.global private @scratch() {addr_space = 0 : i32, alignment = 4 : i64, sym_visibility = "private"} : !llvm.array<2 x array<64 x i32>>
      llvm.mlir.global internal @ordinal_foo() {hal.executable.constant.key = "foo", sym_visibility = "private"} : i32
      llvm.func @export0(%arg0: !llvm.ptr {llvm.align = 16 : i32, llvm.noalias}) {
        %0 = llvm.mlir.addressof @external_data : !llvm.ptr
        %1 = llvm.mlir.addressof @scratch : !llvm.ptr
        %2 = llvm.load %0 : !llvm.ptr -> i32
        %3 = llvm.load %1 : !llvm.ptr -> i32
        %ordinal_ptr = llvm.mlir.addressof @ordinal_foo : !llvm.ptr
        %ordinal = llvm.load %ordinal_ptr : !llvm.ptr -> i32
        %sum = llvm.add %2, %3 : i32
        %value = llvm.add %sum, %ordinal : i32
        llvm.store %value, %arg0 : i32, !llvm.ptr
        llvm.return
      }
    }
  }
}
hal.executable private @executable1 {
  hal.executable.variant public @embedded_elf_hexagon target(#executable_target_hexagon) {
    hal.executable.export public @export1 ordinal(0) layout(#hal.pipeline.layout<bindings = [#hal.pipeline.binding<storage_buffer>]>) count(%arg0: !hal.device) -> (index, index, index) {
      %c1 = arith.constant 1 : index
      hal.return %c1, %c1, %c1 : index, index, index
    }
    hal.executable.constant.block(%device: !hal.device) -> i32 as "bar" {
      %c0 = arith.constant 0 : i32
      hal.return %c0 : i32
    }
    builtin.module {
      llvm.mlir.global external @external_data() {addr_space = 0 : i32, alignment = 16 : i64} : !llvm.array<0 x i8>
      llvm.mlir.global private @scratch() {addr_space = 0 : i32, alignment = 4 : i64, sym_visibility = "private"} : !llvm.array<2 x array<128 x i32>>
      llvm.mlir.global internal @ordinal_bar() {hal.executable.constant.key = "bar", sym_visibility = "private"} : i32
      llvm.func @export1(%arg0: !llvm.ptr {llvm.align = 16 : i32, llvm.noalias}) {
        %0 = llvm.mlir.addressof @external_data : !llvm.ptr
        %1 = llvm.mlir.addressof @scratch : !llvm.ptr
        %2 = llvm.load %0 : !llvm.ptr -> i32
        %3 = llvm.load %1 : !llvm.ptr -> i32
        %ordinal_ptr = llvm.mlir.addressof @ordinal_bar : !llvm.ptr
        %ordinal = llvm.load %ordinal_ptr : !llvm.ptr -> i32
        %sum = llvm.add %2, %3 : i32
        %value = llvm.add %sum, %ordinal : i32
        llvm.store %value, %arg0 : i32, !llvm.ptr
        llvm.return
      }
    }
  }
}

func.func @lookup() -> index {
  %ordinal = hal.executable.export.ordinal target(@executable1::@embedded_elf_hexagon::@export1) : index
  return %ordinal : index
}
// CHECK: hal.executable.export.ordinal target(@link_two_executables_linked::@embedded_elf_hexagon::@export1)

}

// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Adapted from Codegen/LLVMCPU/test/synchronize_symbol_visibility.mlir
// at IREE revision a45adeaa6115e446c898e6eb21fb6edc0e65ddc4.

// RUN: iree-opt --iree-hexagon-synchronize-symbol-visibility %s | FileCheck %s

// CHECK-LABEL: llvm.func internal @internal_fn() attributes {sym_visibility = "private"}
llvm.func internal @internal_fn() {
  llvm.return
}


// Additional coverage for LLVM globals and externally visible functions.
// CHECK: llvm.mlir.global private @private_global(0 : i32)
// CHECK-SAME: sym_visibility = "private"
llvm.mlir.global private @private_global(0 : i32) {sym_visibility = "public"} : i32
// CHECK: llvm.mlir.global internal @internal_global(0 : i32)
// CHECK-SAME: sym_visibility = "private"
llvm.mlir.global internal @internal_global(0 : i32) : i32
// CHECK: llvm.mlir.global external @external_global() {addr_space = 0 : i32} : i32
llvm.mlir.global external @external_global() {sym_visibility = "private"} : i32
// CHECK: llvm.func @external_fn(){{$}}
llvm.func @external_fn() attributes {sym_visibility = "private"}

// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Adapted from Codegen/LLVMCPU/test/assign_constant_ordinals.mlir
// at IREE revision a45adeaa6115e446c898e6eb21fb6edc0e65ddc4.

// RUN: iree-opt --pass-pipeline="builtin.module(hal.executable(hal.executable.variant(iree-hexagon-assign-constant-ordinals)))" --split-input-file %s --verify-diagnostics | FileCheck %s

hal.executable private @executable {
  hal.executable.variant public @variant target(#hal.executable.target<"hexagon", "embedded-elf-hexagon">) {
    hal.executable.constant.block(%device: !hal.device) -> i32 as "foo" {
      %c0 = arith.constant 0 : i32
      hal.return %c0 : i32
    }
    hal.executable.constant.block(%device: !hal.device) -> i32 as "bar" {
      %c1 = arith.constant 1 : i32
      hal.return %c1 : i32
    }
    builtin.module {
      // CHECK: llvm.mlir.global internal constant @__constant_ordinal_foo_a(0 : i32)
      llvm.mlir.global internal @__constant_ordinal_foo_a() {addr_space = 0 : i32, hal.executable.constant.key = "foo", sym_visibility = "private"} : i32
      // CHECK: llvm.mlir.global internal constant @__constant_ordinal_foo_b(0 : i32)
      llvm.mlir.global internal @__constant_ordinal_foo_b() {addr_space = 0 : i32, hal.executable.constant.key = "foo", sym_visibility = "private"} : i32
      // CHECK: llvm.mlir.global internal constant @__constant_ordinal_bar(1 : i32)
      llvm.mlir.global internal @__constant_ordinal_bar() {addr_space = 0 : i32, hal.executable.constant.key = "bar", sym_visibility = "private"} : i32
    }
  }
}


// A different variant owns its own key order.
hal.executable private @other_executable {
  hal.executable.variant public @variant target(#hal.executable.target<"hexagon", "embedded-elf-hexagon">) {
    hal.executable.constant.block(%device: !hal.device) -> i32 as "bar" {
      %c1 = arith.constant 1 : i32
      hal.return %c1 : i32
    }
    hal.executable.constant.block(%device: !hal.device) -> i32 as "foo" {
      %c0 = arith.constant 0 : i32
      hal.return %c0 : i32
    }
    builtin.module {
      // CHECK: llvm.mlir.global internal constant @__constant_ordinal_foo(1 : i32)
      llvm.mlir.global internal @__constant_ordinal_foo() {hal.executable.constant.key = "foo", sym_visibility = "private"} : i32
      // CHECK: llvm.mlir.global internal constant @__constant_ordinal_bar(0 : i32)
      llvm.mlir.global internal @__constant_ordinal_bar() {hal.executable.constant.key = "bar", sym_visibility = "private"} : i32
    }
  }
}


// -----

hal.executable private @missing_key {
  hal.executable.variant public @variant target(#hal.executable.target<"hexagon", "embedded-elf-hexagon">) {
    hal.executable.constant.block(%device: !hal.device) -> i32 as "provided" {
      %c0 = arith.constant 0 : i32
      hal.return %c0 : i32
    }
    builtin.module {
      // expected-error @+1 {{no constant block providing key}}
      llvm.mlir.global internal @ordinal_missing() {hal.executable.constant.key = "missing", sym_visibility = "private"} : i32
    }
  }
}

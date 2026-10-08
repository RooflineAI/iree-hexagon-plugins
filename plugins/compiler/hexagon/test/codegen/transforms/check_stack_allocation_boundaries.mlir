// Copyright 2026 RooflineAI GmbH
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Additional local coverage for cumulative sizes, alignment and target limits.
// RUN: iree-opt --pass-pipeline="builtin.module(func.func(iree-hexagon-check-ir-before-llvm-conversion))" --split-input-file %s --verify-diagnostics

func.func @at_limit() attributes {
  hal.executable.target = #hal.executable.target<"hexagon", "embedded-elf-hexagon", {max_stack_allocation_size = 16 : i64}>
} {
  %buffer = memref.alloca() : memref<4xf32>
  return
}

// -----

// expected-error @+1 {{exceeded stack allocation limit of 16 bytes for function. Got 20 bytes}}
func.func @cumulative_size() attributes {
  hal.executable.target = #hal.executable.target<"hexagon", "embedded-elf-hexagon", {max_stack_allocation_size = 16 : index}>
} {
  %first = memref.alloca() : memref<2xf32>
  %second = memref.alloca() : memref<3xf32>
  return
}

// -----

// expected-error @+1 {{exceeded stack allocation limit of 16 bytes for function. Got 32 bytes}}
func.func @alignment_rounding() attributes {
  hal.executable.target = #hal.executable.target<"hexagon", "embedded-elf-hexagon", {max_stack_allocation_size = 16 : i64}>
} {
  %buffer = memref.alloca() alignment = 32 : memref<1xi8>
  return
}

// -----

module attributes {
  hal.executable.target = #hal.executable.target<"hexagon", "embedded-elf-hexagon", {max_stack_allocation_size = 16 : i64}>
} {
  // expected-error @+1 {{exceeded stack allocation limit of 16 bytes for function. Got 20 bytes}}
  func.func @inherited_limit() {
    %buffer = memref.alloca() : memref<5xf32>
    return
  }
  func.func @no_allocations() {
    return
  }
  func.func private @declaration()
}

// -----

// The bound of a dynamic size comes from the tightest constraint (8 elements,
// 32 bytes), not from the first one found (64 elements, 256 bytes).
func.func @tightest_dynamic_bound(%size: index) attributes {
  hal.executable.target = #hal.executable.target<"hexagon", "embedded-elf-hexagon", {max_stack_allocation_size = 128 : i64}>
} {
  %inner = affine.min affine_map<(d0) -> (d0, 8)>(%size)
  %outer = affine.min affine_map<(d0) -> (d0, 64)>(%inner)
  %buffer = memref.alloca(%outer) : memref<?xf32>
  return
}

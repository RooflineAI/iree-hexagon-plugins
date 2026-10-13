// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Adapted from Codegen/LLVMCPU/test/verify_vector_size_legality.mlir
// at IREE revision a45adeaa6115e446c898e6eb21fb6edc0e65ddc4.

// RUN: iree-opt --pass-pipeline="builtin.module(func.func(iree-hexagon-verify-vector-size-legality))" --split-input-file %s --verify-diagnostics

// expected-error @+1 {{One or more operations with large vector sizes (32768 bytes) were found:}}
func.func @large_vector_with_native_vector_size(%arg0 : vector<16x64x64xf32>) -> vector<16x64x64xf32> attributes {
  hal.executable.target = #hal.executable.target<"hexagon", "embedded-elf-hexagon", {native_vector_size = 64}>
} {
  // expected-note-re @+1 {{math.exp}}
  %0 = math.exp %arg0 : vector<16x64x64xf32>
  // expected-note-re @+1 {{return}}
  return %0 : vector<16x64x64xf32>
}

// -----

// expected-error @+1 {{One or more operations with large vector sizes (32768 bytes) were found:}}
func.func @large_contract_with_native_vector_size(%lhs : vector<32x64xf32>, %rhs : vector<32x64xf32>, %acc : vector<32x32xf32>) -> vector<32x32xf32> attributes {
  hal.executable.target = #hal.executable.target<"hexagon", "embedded-elf-hexagon", {native_vector_size = 64}>
} {
  // expected-note-re @+1 {{vector.contract}}
  %0 = vector.contract {
      indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>,
                       affine_map<(d0, d1, d2) -> (d1, d2)>,
                       affine_map<(d0, d1, d2) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel", "reduction"],
      kind = #vector.kind<add>}
    %lhs, %rhs, %acc : vector<32x64xf32>, vector<32x64xf32> into vector<32x32xf32>
  return %0 : vector<32x32xf32>
}

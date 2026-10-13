// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Adapted from Codegen/LLVMCPU/test/vector_transpose_lowering.mlir
// at IREE revision a45adeaa6115e446c898e6eb21fb6edc0e65ddc4.

// RUN: iree-opt --pass-pipeline='builtin.module(func.func(iree-hexagon-vector-transpose-lowering))' --split-input-file %s | FileCheck %s

// Verify that the vector transpose lowering patterns trigger as expected. We
// shouldn't check the pattern output in detail as that testing should happen in
// MLIR, where the patterns are implemented.

func.func @i4_transpose(%a: vector<8x16xi4>) -> vector<16x8xi4> {
  %0 = vector.transpose %a, [1, 0] : vector<8x16xi4> to vector<16x8xi4>
  return %0 : vector<16x8xi4>
}

// CHECK-LABEL: func.func @i4_transpose(
//       CHECK:   arith.extsi %{{.*}} : vector<8x16xi4> to vector<8x16xi8>
//       CHECK:   vector.shuffle
//       CHECK:   arith.trunci %{{.*}} : vector<16x8xi8> to vector<16x8xi4>

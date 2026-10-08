// Copyright 2026 RooflineAI GmbH
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Additional local coverage for the split-transfer options and the lowerings
// the ported virtual_vector_lowering.mlir does not exercise.
// RUN: iree-opt --pass-pipeline="builtin.module(func.func(iree-hexagon-virtual-vector-lowering))" %s | FileCheck %s --check-prefixes=CHECK,NONE
// RUN: iree-opt --pass-pipeline="builtin.module(func.func(iree-hexagon-virtual-vector-lowering{split-transfers=shuffle}))" %s | FileCheck %s --check-prefixes=CHECK,NONE
// RUN: iree-opt --pass-pipeline="builtin.module(func.func(iree-hexagon-virtual-vector-lowering{split-transfers=linalg-copy}))" %s | FileCheck %s --check-prefixes=CHECK,COPY

// NONE-LABEL: func.func @partial_transfer(
// NONE-NOT: scf.if
// NONE: vector.transfer_read
// NONE: return
// COPY-LABEL: func.func @partial_transfer(
// COPY: scf.if
// COPY: linalg.fill
// COPY: memref.copy
// COPY: vector.transfer_read
// COPY: return
func.func @partial_transfer(%base: memref<?x8xf32>, %i: index) -> vector<4x8xf32> {
  %c0 = arith.constant 0 : index
  %zero = arith.constant 0.0 : f32
  %r = vector.transfer_read %base[%i, %c0], %zero {in_bounds = [false, true]}
      : memref<?x8xf32>, vector<4x8xf32>
  return %r : vector<4x8xf32>
}

// A real Hexagon target must use the contiguous gather override. Each lane is
// conditional, and an inactive lane keeps the input pass-through value.
// CHECK-LABEL: func.func @masked_aligned_gather(
// CHECK-SAME: %[[BASE:.*]]: memref<4x8xf32>, %[[IDX:.*]]: vector<2xi32>, %[[MASK:.*]]: vector<2xi1>, %[[PASS:.*]]: vector<2xf32>
// CHECK: arith.index_cast %[[IDX]] : vector<2xi32> to vector<2xindex>
// CHECK: %[[M0:.*]] = vector.extract %[[MASK]][0]
// CHECK: %[[R0:.*]] = scf.if %[[M0]]
// CHECK: vector.load %[[BASE]]
// CHECK-SAME: {alignment = 16 : i64}
// CHECK: vector.insert {{.*}}, %[[PASS]] [0]
// CHECK: } else {
// CHECK: scf.yield %[[PASS]]
// CHECK: %[[M1:.*]] = vector.extract %[[MASK]][1]
// CHECK: scf.if %[[M1]]
// CHECK: vector.load %[[BASE]]
// CHECK-SAME: {alignment = 16 : i64}
// CHECK: vector.insert {{.*}}, %[[R0]] [1]
// CHECK: } else {
// CHECK: scf.yield %[[R0]]
// CHECK: return
func.func @masked_aligned_gather(%base: memref<4x8xf32>, %indices: vector<2xi32>,
    %mask: vector<2xi1>, %pass: vector<2xf32>, %i: index, %j: index) -> vector<2xf32>
    attributes {hal.executable.target = #hal.executable.target<"hexagon", "embedded-elf-hexagon", {target_triple = "hexagon-unknown-unknown-elf", cpu_features = "+hvxv79,+hvx-length128b"}>} {
  %r = vector.gather %base[%i, %j] [%indices], %mask, %pass {alignment = 16 : i64}
      : memref<4x8xf32>, vector<2xi32>, vector<2xi1>, vector<2xf32> into vector<2xf32>
  return %r : vector<2xf32>
}

// CHECK-LABEL: func.func @inner_reduction(
// CHECK: vector.reduction <add>
// CHECK: vector.reduction <add>
// CHECK: return
func.func @inner_reduction(%input: vector<2x4xf32>, %init: vector<2xf32>) -> vector<2xf32> {
  %r = vector.multi_reduction <add>, %input, %init [1]
      : vector<2x4xf32> to vector<2xf32>
  return %r : vector<2xf32>
}

// The pass must cancel inverse conversions left by interface-driven lowering.
// CHECK-LABEL: func.func @cancel_inverse_conversions(
// CHECK-SAME: %[[INPUT:.*]]: vector<4xi16>
// CHECK-NOT: util.hoistable_conversion
// CHECK: return %[[INPUT]] : vector<4xi16>
func.func @cancel_inverse_conversions(%input: vector<4xi16>) -> vector<4xi16> {
  %wide = util.hoistable_conversion "widen" inverts("narrow")
      (%v = %input) : (vector<4xi16>) -> vector<4xi32> {
    %converted = arith.extsi %v : vector<4xi16> to vector<4xi32>
    util.return %converted : vector<4xi32>
  }
  %result = util.hoistable_conversion "narrow" inverts("widen")
      (%v = %wide) : (vector<4xi32>) -> vector<4xi16> {
    %converted = arith.trunci %v : vector<4xi32> to vector<4xi16>
    util.return %converted : vector<4xi16>
  }
  return %result : vector<4xi16>
}

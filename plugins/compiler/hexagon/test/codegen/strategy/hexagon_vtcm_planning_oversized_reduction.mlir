// HMX requires VTCM, and VTCM tiling stages every reduction dimension whole.
// When the whole K extent cannot be shown to fit, planning must fail rather
// than plan a K tile the VTCM tiling pass never realizes.
//
// RUN: not iree-opt \
// RUN:   --iree-hexagon-enable-vtcm-tiling \
// RUN:   --iree-hexagon-enable-hmx-matmul \
// RUN:   --pass-pipeline='builtin.module(iree-hexagon-select-lowering-strategy)' \
// RUN:   --split-input-file %s -o /dev/null 2>&1 | FileCheck %s

#target = #hal.executable.target<"hexagon", "embedded-elf-hexagon", {cpu = "hexagonv79", cpu_features = "+hvxv79,+hvx-length128b", max_stack_allocation_size = 16384 : i64, target_triple = "hexagon-unknown-unknown-elf"}>

// K = 1M f16: one LHS row plus one RHS column is 4 MiB, against 2 MiB of VTCM.
func.func @hmx_oversized_k(%lhs: tensor<32x1048576xf16>, %rhs: tensor<1048576x32xf16>) -> tensor<32x32xf32> attributes {hal.executable.target = #target} {
  %zero = arith.constant 0.0 : f32
  %empty = tensor.empty() : tensor<32x32xf32>
  %init = linalg.fill ins(%zero : f32) outs(%empty : tensor<32x32xf32>) -> tensor<32x32xf32>
  %result = linalg.matmul ins(%lhs, %rhs : tensor<32x1048576xf16>, tensor<1048576x32xf16>) outs(%init : tensor<32x32xf32>) -> tensor<32x32xf32>
  return %result : tensor<32x32xf32>
}

// CHECK: error: failed to derive required VTCM tile sizes: the root does not fit in VTCM with its reduction dimensions staged whole

// -----

#target = #hal.executable.target<"hexagon", "embedded-elf-hexagon", {cpu = "hexagonv79", cpu_features = "+hvxv79,+hvx-length128b", max_stack_allocation_size = 16384 : i64, target_triple = "hexagon-unknown-unknown-elf"}>

// A dynamic K is staged whole at its runtime extent, which no static plan can
// bound.
func.func @hmx_dynamic_k(%lhs: tensor<32x?xf16>, %rhs: tensor<?x32xf16>) -> tensor<32x32xf32> attributes {hal.executable.target = #target} {
  %zero = arith.constant 0.0 : f32
  %empty = tensor.empty() : tensor<32x32xf32>
  %init = linalg.fill ins(%zero : f32) outs(%empty : tensor<32x32xf32>) -> tensor<32x32xf32>
  %result = linalg.matmul ins(%lhs, %rhs : tensor<32x?xf16>, tensor<?x32xf16>) outs(%init : tensor<32x32xf32>) -> tensor<32x32xf32>
  return %result : tensor<32x32xf32>
}

// CHECK: error: failed to derive required VTCM tile sizes: dynamic reduction dimensions are staged whole at their runtime extent

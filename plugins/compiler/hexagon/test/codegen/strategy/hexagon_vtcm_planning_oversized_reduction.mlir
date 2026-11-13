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

// A dynamic K is staged whole at its runtime extent, which nothing bounds here.
func.func @hmx_dynamic_k(%lhs: tensor<32x?xf16>, %rhs: tensor<?x32xf16>) -> tensor<32x32xf32> attributes {hal.executable.target = #target} {
  %zero = arith.constant 0.0 : f32
  %empty = tensor.empty() : tensor<32x32xf32>
  %init = linalg.fill ins(%zero : f32) outs(%empty : tensor<32x32xf32>) -> tensor<32x32xf32>
  %result = linalg.matmul ins(%lhs, %rhs : tensor<32x?xf16>, tensor<?x32xf16>) outs(%init : tensor<32x32xf32>) -> tensor<32x32xf32>
  return %result : tensor<32x32xf32>
}

// CHECK: error: failed to derive required VTCM tile sizes: dynamic reduction dimension has no upper bound

// -----

#target = #hal.executable.target<"hexagon", "embedded-elf-hexagon", {cpu = "hexagonv79", cpu_features = "+hvxv79,+hvx-length128b", max_stack_allocation_size = 16384 : i64, target_triple = "hexagon-unknown-unknown-elf"}>

// A dynamic K is sized by its upper bound, here too large to stage whole.
func.func @hmx_dynamic_k_oversized_bound(%lhs: tensor<32x1048576xf16>, %rhs: tensor<1048576x32xf16>, %k_arg: index) -> tensor<32x32xf32> attributes {hal.executable.target = #target} {
  %zero = arith.constant 0.0 : f32
  %k = util.assume.int %k_arg<umin = 1, umax = 1048576> : index
  %lhs_k = tensor.extract_slice %lhs[0, 0] [32, %k] [1, 1] : tensor<32x1048576xf16> to tensor<32x?xf16>
  %rhs_k = tensor.extract_slice %rhs[0, 0] [%k, 32] [1, 1] : tensor<1048576x32xf16> to tensor<?x32xf16>
  %empty = tensor.empty() : tensor<32x32xf32>
  %init = linalg.fill ins(%zero : f32) outs(%empty : tensor<32x32xf32>) -> tensor<32x32xf32>
  %result = linalg.matmul ins(%lhs_k, %rhs_k : tensor<32x?xf16>, tensor<?x32xf16>) outs(%init : tensor<32x32xf32>) -> tensor<32x32xf32>
  return %result : tensor<32x32xf32>
}

// CHECK: error: failed to derive required VTCM tile sizes: the root does not fit in VTCM with its reduction dimensions staged whole

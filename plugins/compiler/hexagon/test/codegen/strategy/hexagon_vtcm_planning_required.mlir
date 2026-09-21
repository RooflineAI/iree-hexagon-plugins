// HMX has a mandatory VTCM contract. Selecting it while VTCM is disabled is a
// planning error.
//
// RUN: not iree-opt \
// RUN:   --iree-hexagon-enable-hmx-matmul \
// RUN:   --iree-hexagon-enable-vtcm-tiling=false \
// RUN:   --mlir-print-ir-after-failure \
// RUN:   --pass-pipeline='builtin.module(iree-hexagon-select-lowering-strategy)' \
// RUN:   %s -o /dev/null 2>&1 | FileCheck %s

#target = #hal.executable.target<"hexagon", "embedded-elf-hexagon", {cpu = "hexagonv79", cpu_features = "+hvxv79,+hvx-length128b", max_stack_allocation_size = 16384 : i64, target_triple = "hexagon-unknown-unknown-elf"}>

func.func @hmx_without_vtcm(%lhs: tensor<32x32xf16>, %rhs: tensor<32x32xf16>) -> tensor<32x32xf16> attributes {hal.executable.target = #target} {
  %zero = arith.constant 0.0 : f16
  %empty = tensor.empty() : tensor<32x32xf16>
  %init = linalg.fill ins(%zero : f16) outs(%empty : tensor<32x32xf16>) -> tensor<32x32xf16>
  %result = linalg.matmul ins(%lhs, %rhs : tensor<32x32xf16>, tensor<32x32xf16>) outs(%init : tensor<32x32xf16>) -> tensor<32x32xf16>
  return %result : tensor<32x32xf16>
}

// CHECK: error: selected Hexagon pipeline requires VTCM tiling
// CHECK: IR Dump After HexagonSelectLoweringStrategyPass Failed
// CHECK: func.func @hmx_without_vtcm(
// CHECK-NOT: translation_info
// CHECK-NOT: lowering_config
// CHECK-NOT: hexagon_vtcm_tiling_config
// CHECK: return

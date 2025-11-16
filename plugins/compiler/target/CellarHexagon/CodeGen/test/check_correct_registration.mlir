// TODO: This is just a placeholder test on placeholder pass only containing a canonicalizer.
// Only meant to be replaced in the future with real Hexagon passes.
// RUN: iree-opt --iree-plugin=hal_target_hexagon --pass-pipeline='builtin.module(iree-hexagon-configuration-pipeline)' %s | FileCheck %s

// CHECK-LABEL: func.func @fold_constants()
// CHECK: %c3_i32 = arith.constant 3 : i32
// CHECK-NEXT: return %c3_i32 : i32

module {
  func.func @fold_constants() -> i32 {
    %c1 = arith.constant 1 : i32
    %c2 = arith.constant 2 : i32
    %sum = arith.addi %c1, %c2 : i32
    return %sum : i32
  }
}

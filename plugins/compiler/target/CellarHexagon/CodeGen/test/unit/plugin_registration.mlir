
// This test is just making sure that the configuration pipeline and therefore the plugin as a whole are properly registered and calling them does not throw any error
// RUN: iree-opt --iree-plugin=hal_target_hexagon --pass-pipeline='builtin.module(hal.executable(hal.executable.variant(iree-hexagon-configuration-pipeline)))' %s

module {
  func.func @fold_constants() -> i32 {
    %c1 = arith.constant 1 : i32
    %c2 = arith.constant 2 : i32
    %sum = arith.addi %c1, %c2 : i32
    return %sum : i32
  }
}

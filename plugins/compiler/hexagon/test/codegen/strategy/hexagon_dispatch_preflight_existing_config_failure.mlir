// Existing per-op lowering configs are rejected before planning. The failed
// pass must preserve the existing config and must not add translation info.
//
// RUN: not iree-opt \
// RUN:   --iree-hexagon-enable-vtcm-tiling=false \
// RUN:   --iree-hexagon-enable-hmx-matmul=false \
// RUN:   --mlir-print-ir-after-failure \
// RUN:   --pass-pipeline='builtin.module(iree-hexagon-select-lowering-strategy)' \
// RUN:   %s -o /dev/null 2>&1 | FileCheck %s

#target = #hal.executable.target<"hexagon", "embedded-elf-hexagon", {cpu = "hexagonv79", cpu_features = "+hvxv79,+hvx-length128b", max_stack_allocation_size = 16384 : i64, target_triple = "hexagon-unknown-unknown-elf"}>
#existing = #iree_cpu.lowering_config<vector_common_parallel = [4]>

func.func @existing_lowering_config(%input: tensor<8xf32>) -> tensor<8xf32> attributes {hal.executable.target = #target} {
  %empty = tensor.empty() : tensor<8xf32>
  %result = linalg.generic {
      indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
      iterator_types = ["parallel"]}
      ins(%input : tensor<8xf32>) outs(%empty : tensor<8xf32>) attrs = {lowering_config = #existing} {
  ^bb0(%value: f32, %out: f32):
    linalg.yield %value : f32
  } -> tensor<8xf32>
  return %result : tensor<8xf32>
}

// CHECK: error: expected an unconfigured operation before Hexagon dispatch planning
// CHECK: IR Dump After HexagonSelectLoweringStrategyPass Failed
// CHECK: func.func @existing_lowering_config(
// CHECK-NOT: translation_info
// CHECK: linalg.generic
// CHECK-SAME: lowering_config = #[[EXISTING:.+]]
// CHECK: return

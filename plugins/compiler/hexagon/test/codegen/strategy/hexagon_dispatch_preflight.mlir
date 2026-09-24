// Focused tests for dispatch-wide planner behavior and target decoding.
//
// RUN: iree-opt \
// RUN:   --iree-hexagon-enable-vtcm-tiling=false \
// RUN:   --iree-hexagon-enable-hmx-matmul=false \
// RUN:   --pass-pipeline='builtin.module(iree-hexagon-select-lowering-strategy)' \
// RUN:   --split-input-file %s | FileCheck %s

#target_128 = #hal.executable.target<"hexagon", "embedded-elf-hexagon", {cpu = "hexagonv79", cpu_features = "+hvxv79,+hvx-length128b", max_stack_allocation_size = 16384 : i64, target_triple = "hexagon-unknown-unknown-elf"}>
#already_selected = #iree_codegen.translation_info<pipeline = #iree_cpu.pipeline<Default>>

// Existing translation info preserves the historical no-op behavior.
func.func @already_configured(%src: tensor<4xf32>) -> tensor<4xf32> attributes {hal.executable.target = #target_128, translation_info = #already_selected} {
  return %src : tensor<4xf32>
}
// CHECK: func.func @already_configured(
// CHECK-SAME: translation_info = #[[ALREADY:.+]]
// CHECK-NOT: lowering_config
// CHECK: return

// -----

#target_128 = #hal.executable.target<"hexagon", "embedded-elf-hexagon", {cpu = "hexagonv79", cpu_features = "+hvxv79,+hvx-length128b", max_stack_allocation_size = 16384 : i64, target_triple = "hexagon-unknown-unknown-elf"}>

// A root excluded from shape analysis is a normal CPUDefault fallback, not an
// undiagnosed strategy-selection failure.
func.func @zero_loop_root() -> tensor<f32> attributes {hal.executable.target = #target_128} {
  %empty = tensor.empty() : tensor<f32>
  %result = linalg.generic {
      indexing_maps = [affine_map<() -> ()>], iterator_types = []}
      outs(%empty : tensor<f32>) {
  ^bb0(%out: f32):
    linalg.yield %out : f32
  } -> tensor<f32>
  return %result : tensor<f32>
}
// CHECK: func.func @zero_loop_root()
// CHECK-SAME: translation_info = #[[ZERO_LOOP_DEFAULT:.+]]
// CHECK: linalg.generic
// CHECK-NOT: lowering_config
// CHECK: return

// -----

#target_128 = #hal.executable.target<"hexagon", "embedded-elf-hexagon", {cpu = "hexagonv79", cpu_features = "+hvxv79,+hvx-length128b", max_stack_allocation_size = 16384 : i64, target_triple = "hexagon-unknown-unknown-elf"}>

// Skipping an ineligible compute operation keeps the analyzed operation
// ordinals compact and consistent with their DispatchShape vector positions.
func.func @skipped_zero_loop_before_root(%input: tensor<4xf32>) -> tensor<4xf32> attributes {hal.executable.target = #target_128} {
  %scalar_empty = tensor.empty() : tensor<f32>
  %scalar = linalg.generic {
      indexing_maps = [affine_map<() -> ()>], iterator_types = []}
      outs(%scalar_empty : tensor<f32>) {
  ^bb0(%out: f32):
    linalg.yield %out : f32
  } -> tensor<f32>
  %empty = tensor.empty() : tensor<4xf32>
  %result = linalg.generic {
      indexing_maps = [affine_map<(d0) -> (d0)>,
                       affine_map<(d0) -> (d0)>],
      iterator_types = ["parallel"]}
      ins(%input : tensor<4xf32>) outs(%empty : tensor<4xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<4xf32>
  return %result : tensor<4xf32>
}
// CHECK: func.func @skipped_zero_loop_before_root(
// CHECK: linalg.generic
// CHECK-NOT: lowering_config
// CHECK: linalg.generic
// CHECK-SAME: lowering_config
// CHECK: return

// -----

#target_128 = #hal.executable.target<"hexagon", "embedded-elf-hexagon", {cpu = "hexagonv79", cpu_features = "+hvxv79,+hvx-length128b", max_stack_allocation_size = 16384 : i64, target_triple = "hexagon-unknown-unknown-elf"}>

// Empty/no-compute dispatches receive only the explicit fallback pipeline.
func.func @empty_dispatch() attributes {hal.executable.target = #target_128} {
  return
}
// CHECK-DAG: #[[DEFAULT:.+]] = #iree_codegen.translation_info<pipeline = #iree_cpu.pipeline<Default>>
// CHECK: func.func @empty_dispatch()
// CHECK-SAME: translation_info = #[[DEFAULT]]
// CHECK-NOT: lowering_config
// CHECK: return

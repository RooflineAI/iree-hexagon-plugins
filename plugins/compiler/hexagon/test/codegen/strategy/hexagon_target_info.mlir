// Focused tests for dispatch-wide planner behavior and target decoding.
//
// RUN: iree-opt \
// RUN:   --iree-hexagon-enable-vtcm-tiling=false \
// RUN:   --iree-hexagon-enable-hmx-matmul=false \
// RUN:   --pass-pipeline='builtin.module(iree-hexagon-select-lowering-strategy)' \
// RUN:   --split-input-file %s | FileCheck %s

#target_128 = #hal.executable.target<"hexagon", "embedded-elf-hexagon", {cpu = "hexagonv79", cpu_features = "+hvxv79,+hvx-length128b", max_stack_allocation_size = 16384 : i64, target_triple = "hexagon-unknown-unknown-elf"}>

// A ragged bound keeps the full native width because the buffer pipeline peels
// the remainder. Distribution and compute are identical and root-owned.
func.func @copy_128b_ragged(%src: memref<4x70xf32>, %dst: memref<4x70xf32>) attributes {hal.executable.target = #target_128} {
  linalg.copy ins(%src : memref<4x70xf32>) outs(%dst : memref<4x70xf32>)
  return
}
// CHECK-DAG: #[[COPY128:.+]] = #iree_cpu.lowering_config<distribution = [1, 32], vector_common_parallel = [1, 32]>
// CHECK-DAG: #[[BUFFER_PIPELINE:.+]] = #iree_codegen.translation_info<pipeline = #iree_cpu.pipeline<BufferOpsTileAndVectorize>>
// CHECK: func.func @copy_128b_ragged(
// CHECK-SAME: translation_info = #[[BUFFER_PIPELINE]]
// CHECK: linalg.copy
// CHECK-SAME: lowering_config = #[[COPY128]]

// -----

#target_64 = #hal.executable.target<"hexagon", "embedded-elf-hexagon", {cpu = "hexagonv79", cpu_features = "+hvxv79,+hvx-length64b", max_stack_allocation_size = 16384 : i64, target_triple = "hexagon-unknown-unknown-elf"}>

func.func @copy_64b(%src: memref<4x70xf32>, %dst: memref<4x70xf32>) attributes {hal.executable.target = #target_64} {
  linalg.copy ins(%src : memref<4x70xf32>) outs(%dst : memref<4x70xf32>)
  return
}
// CHECK-DAG: #[[COPY64:.+]] = #iree_cpu.lowering_config<distribution = [1, 16], vector_common_parallel = [1, 16]>
// CHECK: func.func @copy_64b(
// CHECK: linalg.copy
// CHECK-SAME: lowering_config = #[[COPY64]]

// -----

#target_64 = #hal.executable.target<"hexagon", "embedded-elf-hexagon", {cpu = "hexagonv79", cpu_features = "+hvxv79,+hvx-length64b", max_stack_allocation_size = 16384 : i64, target_triple = "hexagon-unknown-unknown-elf"}>

// Dynamic bounds retain the native target width.
func.func @copy_64b_dynamic(%src: memref<?x?xf32>, %dst: memref<?x?xf32>) attributes {hal.executable.target = #target_64} {
  linalg.copy ins(%src : memref<?x?xf32>) outs(%dst : memref<?x?xf32>)
  return
}
// CHECK-DAG: #[[DYNAMIC:.+]] = #iree_cpu.lowering_config<distribution = [1, 16], vector_common_parallel = [1, 16]>
// CHECK: func.func @copy_64b_dynamic(
// CHECK-NOT: memref.dim
// CHECK: linalg.copy
// CHECK-SAME: lowering_config = #[[DYNAMIC]]

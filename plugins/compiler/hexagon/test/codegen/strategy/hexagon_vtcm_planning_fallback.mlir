// CPUDouble supports VTCM but does not require it. If the footprint helper
// cannot derive a tile, planning continues without VTCM and preserves the
// strategy's ordinary cache/compute decisions.
//
// RUN: iree-opt --iree-load-plugin=hexagon=$ROOF_HEXAGON_COMPILER_PLUGIN \
// RUN:   --iree-hexagon-enable-vtcm-tiling \
// RUN:   --pass-pipeline='builtin.module(iree-hexagon-select-lowering-strategy)' \
// RUN:   %s | FileCheck %s

#target = #hal.executable.target<"hexagon", "embedded-elf-hexagon", {cpu = "hexagonv79", cpu_features = "+hvxv79,+hvx-length128b", max_stack_allocation_size = 16384 : i64, target_triple = "hexagon-unknown-unknown-elf"}>

func.func @unsupported_vtcm_footprint(%input: tensor<8x8xindex>) -> tensor<8x8xindex> attributes {hal.executable.target = #target} {
  %empty = tensor.empty() : tensor<8x8xindex>
  %result = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]}
      ins(%input : tensor<8x8xindex>) outs(%empty : tensor<8x8xindex>) {
  ^bb0(%value: index, %out: index):
    linalg.yield %value : index
  } -> tensor<8x8xindex>
  return %result : tensor<8x8xindex>
}

// CHECK-DAG: #[[ROOT:.+]] = #iree_cpu.lowering_config<cache_parallel = [8, 0], distribution = [0, 0], vector_common_parallel = [1, 1]>
// CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = #iree_cpu.pipeline<DoubleTilingExpert>, {enable_loop_peeling}>
// CHECK-DAG: #[[BUFFER_ROOT:.+]] = #iree_cpu.lowering_config<cache_parallel = [8, 0], distribution = [0, 0], vector_common_parallel = [1, 8]>
// CHECK-DAG: #[[BUFFER_TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = #iree_cpu.pipeline<DoubleTilingExpert>>
// CHECK-NOT: hexagon_vtcm_tiling_config
// CHECK: func.func @unsupported_vtcm_footprint(
// CHECK-SAME: translation_info = #[[TRANSLATION]]
// CHECK: linalg.generic
// CHECK-SAME: lowering_config = #[[ROOT]]

// -----

// Buffer-semantics Linalg is also a valid CPUDouble input. It is not eligible
// for VTCM staging, but that must not make the optional pipeline fail.
func.func @buffer_semantics_generic(%input: memref<8x8xf32>, %output: memref<8x8xf32>) attributes {hal.executable.target = #target} {
  linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]}
      ins(%input : memref<8x8xf32>) outs(%output : memref<8x8xf32>) {
  ^bb0(%value: f32, %out: f32):
    %sum = arith.addf %value, %out : f32
    linalg.yield %sum : f32
  }
  return
}

// CHECK-NOT: hexagon_vtcm_tiling_config
// CHECK: func.func @buffer_semantics_generic(
// CHECK-SAME: translation_info = #[[BUFFER_TRANSLATION]]
// CHECK: linalg.generic
// CHECK-SAME: lowering_config = #[[BUFFER_ROOT]]

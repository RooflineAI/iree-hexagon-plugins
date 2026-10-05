// MultiTilingExpert supports VTCM but does not require it. If the footprint helper
// cannot derive a tile, planning continues without VTCM and preserves the
// strategy's ordinary cache/compute decisions.
//
// RUN: iree-opt \
// RUN:   --iree-hexagon-enable-vtcm-tiling \
// RUN:   --iree-hexagon-enable-hmx-matmul=false \
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
// CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = #iree_hexagon.pipeline<MultiTilingExpert>, {enable_loop_peeling}>
// CHECK-DAG: #[[BUFFER_ROOT:.+]] = #iree_cpu.lowering_config<cache_parallel = [8, 0], distribution = [0, 0], vector_common_parallel = [1, 8]>
// CHECK-DAG: #[[BUFFER_TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = #iree_hexagon.pipeline<MultiTilingExpert>>
// CHECK-DAG: #[[FULL_REDUCTION_VTCM:.+]] = #iree_hexagon.vtcm_tiling_config<tile_sizes = [256]>
// CHECK-DAG: #[[ROW_VTCM:.+]] = #iree_hexagon.vtcm_tiling_config<tile_sizes = [{{[0-9]+}}, 65536]>
// CHECK-NOT: hexagon_vtcm_tiling_config
// CHECK: func.func @unsupported_vtcm_footprint(
// CHECK-SAME: translation_info = #[[TRANSLATION]]
// CHECK: linalg.generic
// CHECK-SAME: lowering_config = #[[ROOT]]

// -----

// Buffer-semantics Linalg is also a valid MultiTilingExpert input. It is not eligible
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

// -----

// A full reduction to a 0-d tensor has no parallel loop. Its operands fit
// whole, so it is still planned for VTCM; HexagonVTCMTilingPass stages it
// through a single-iteration forall (the loop form of a unit parallel dim).
func.func @full_reduction_to_scalar(%input: tensor<256xf32>) -> tensor<f32> attributes {hal.executable.target = #target} {
  %cst = arith.constant 0.0 : f32
  %empty = tensor.empty() : tensor<f32>
  %init = linalg.fill ins(%cst : f32) outs(%empty : tensor<f32>) -> tensor<f32>
  %result = linalg.generic {
      indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> ()>],
      iterator_types = ["reduction"]}
      ins(%input : tensor<256xf32>) outs(%init : tensor<f32>) {
  ^bb0(%value: f32, %out: f32):
    %sum = arith.addf %value, %out : f32
    linalg.yield %sum : f32
  } -> tensor<f32>
  return %result : tensor<f32>
}

// CHECK: func.func @full_reduction_to_scalar(
// CHECK: linalg.generic
// CHECK-SAME: hexagon_vtcm_tiling_config = #[[FULL_REDUCTION_VTCM]]
// CHECK: return

// -----

// VTCM tiling stages reduction dimensions whole. A 4 MiB f32 row does not fit
// in the 2 MiB VTCM even with a unit parallel tile, so the optional VTCM plan
// is declined. (The footprint search alone would shrink the reduction to
// 65536, a tile the VTCM tiling pass never realizes.)
func.func @oversized_row_reduction(%input: tensor<4x1048576xf32>) -> tensor<4xf32> attributes {hal.executable.target = #target} {
  %cst = arith.constant 0.0 : f32
  %empty = tensor.empty() : tensor<4xf32>
  %init = linalg.fill ins(%cst : f32) outs(%empty : tensor<4xf32>) -> tensor<4xf32>
  %result = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>],
      iterator_types = ["parallel", "reduction"]}
      ins(%input : tensor<4x1048576xf32>) outs(%init : tensor<4xf32>) {
  ^bb0(%value: f32, %out: f32):
    %sum = arith.addf %value, %out : f32
    linalg.yield %sum : f32
  } -> tensor<4xf32>
  return %result : tensor<4xf32>
}

// CHECK: func.func @oversized_row_reduction(
// CHECK-NOT: hexagon_vtcm_tiling_config
// CHECK: return

// -----

// The same reduction fits once only the parallel dimension is tiled: the plan
// keeps the 65536-wide reduction whole.
func.func @row_reduction_fits_with_parallel_tiles(%input: tensor<64x65536xf32>) -> tensor<64xf32> attributes {hal.executable.target = #target} {
  %cst = arith.constant 0.0 : f32
  %empty = tensor.empty() : tensor<64xf32>
  %init = linalg.fill ins(%cst : f32) outs(%empty : tensor<64xf32>) -> tensor<64xf32>
  %result = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>],
      iterator_types = ["parallel", "reduction"]}
      ins(%input : tensor<64x65536xf32>) outs(%init : tensor<64xf32>) {
  ^bb0(%value: f32, %out: f32):
    %sum = arith.addf %value, %out : f32
    linalg.yield %sum : f32
  } -> tensor<64xf32>
  return %result : tensor<64xf32>
}

// CHECK: func.func @row_reduction_fits_with_parallel_tiles(
// CHECK: linalg.generic
// CHECK-SAME: hexagon_vtcm_tiling_config = #[[ROW_VTCM]]
// CHECK: return

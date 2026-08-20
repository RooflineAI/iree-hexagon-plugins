// RUN: iree-opt --iree-load-plugin=hexagon=$HEXAGON_COMPILER_PLUGIN \
// RUN:   --pass-pipeline='builtin.module(func.func(iree-hexagon-vtcm-tiling))' \
// RUN:   --split-input-file %s | FileCheck %s

#map = affine_map<(d0, d1) -> (d0, d1)>

// CHECK-LABEL: func.func @configured(
// CHECK: scf.forall
// CHECK: tensor.extract_slice
// CHECK: iree_hexagon.stage_to_vtcm %{{.*}} : tensor<4x4xf32>
// CHECK: tensor.extract_slice
// CHECK: iree_hexagon.stage_to_vtcm %{{.*}} : tensor<4x4xf32>
// CHECK: linalg.generic {{.*}} ins(%{{.*}} : tensor<4x4xf32>) outs(%{{.*}} : tensor<4x4xf32>)
// CHECK: scf.forall.in_parallel
// CHECK: tensor.parallel_insert_slice
// CHECK-NOT: hexagon_vtcm_tiling_config

func.func @configured(%arg0: tensor<8x8xf32>, %arg1: tensor<8x8xf32>) -> tensor<8x8xf32> {
  %0 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%arg0 : tensor<8x8xf32>) outs(%arg1 : tensor<8x8xf32>) attrs = {hexagon_vtcm_tiling_config = #iree_hexagon.vtcm_tiling_config<tile_sizes = [4, 4]>} {
  ^bb0(%in: f32, %out: f32):
    %1 = arith.addf %in, %out : f32
    linalg.yield %1 : f32
  } -> tensor<8x8xf32>
  return %0 : tensor<8x8xf32>
}

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>

// Empty-backed output tiles should be staged with an allocation marker rather
// than a copy marker.
// CHECK-LABEL: func.func @configured_empty_output(
// CHECK: scf.forall
// CHECK: tensor.extract_slice
// CHECK: iree_hexagon.stage_to_vtcm %{{.*}} : tensor<4x4xf32>
// CHECK: iree_hexagon.vtcm_empty() : tensor<4x4xf32>
// CHECK: linalg.generic {{.*}} ins(%{{.*}} : tensor<4x4xf32>) outs(%{{.*}} : tensor<4x4xf32>)

func.func @configured_empty_output(%arg0: tensor<8x8xf32>) -> tensor<8x8xf32> {
  %empty = tensor.empty() : tensor<8x8xf32>
  %0 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%arg0 : tensor<8x8xf32>) outs(%empty : tensor<8x8xf32>) attrs = {hexagon_vtcm_tiling_config = #iree_hexagon.vtcm_tiling_config<tile_sizes = [4, 4]>} {
  ^bb0(%in: f32, %out: f32):
    %1 = arith.addf %in, %out : f32
    linalg.yield %1 : f32
  } -> tensor<8x8xf32>
  return %0 : tensor<8x8xf32>
}

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>

// CHECK-LABEL: func.func @unconfigured(
// CHECK-NOT: scf.for
// CHECK: linalg.generic

func.func @unconfigured(%arg0: tensor<8x8xf32>, %arg1: tensor<8x8xf32>) -> tensor<8x8xf32> {
  %0 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%arg0 : tensor<8x8xf32>) outs(%arg1 : tensor<8x8xf32>) {
  ^bb0(%in: f32, %out: f32):
    %1 = arith.addf %in, %out : f32
    linalg.yield %1 : f32
  } -> tensor<8x8xf32>
  return %0 : tensor<8x8xf32>
}

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>

// Full-tile configs should have the forall canonicalized away,
// but keep the copy operations to the VTCM
// CHECK-LABEL: func.func @configured_full_tile(
// CHECK-NOT: scf.forall
// CHECK: iree_hexagon.stage_to_vtcm
// CHECK: iree_hexagon.stage_to_vtcm
// CHECK: iree_hexagon.vtcm_empty() : tensor<8x8xf32>
// CHECK: linalg.generic
// CHECK-NOT: hexagon_vtcm_tiling_config

func.func @configured_full_tile(%arg0: tensor<8x8xf32>, %arg1: tensor<8x8xf32>) -> tensor<8x8xf32> {
  %empty = tensor.empty() : tensor<8x8xf32>
  %0 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%arg0, %arg1 : tensor<8x8xf32>, tensor<8x8xf32>) outs(%empty : tensor<8x8xf32>) attrs = {hexagon_vtcm_tiling_config = #iree_hexagon.vtcm_tiling_config<tile_sizes = [8, 8]>} {
  ^bb0(%lhs: f32, %rhs: f32, %out: f32):
    %1 = arith.addf %lhs, %rhs : f32
    linalg.yield %1 : f32
  } -> tensor<8x8xf32>
  return %0 : tensor<8x8xf32>
}

// -----

// Matmul with a reduction dimension: the reduction is zeroed out at this
// dispatch-tiling level so the result is a single scf.forall over M and N.
// All operands are staged (none fit as a whole tensor) so each per-tile slice
// is individually copied to VTCM inside the forall body.

// CHECK-LABEL: func.func @configured_matmul(
// CHECK: scf.forall
// LHS slice (M-tiled, full K) → VTCM copy
// CHECK: tensor.extract_slice
// CHECK: iree_hexagon.stage_to_vtcm %{{.*}}
// RHS slice (full K, N-tiled) → VTCM copy
// CHECK: tensor.extract_slice
// CHECK: iree_hexagon.stage_to_vtcm %{{.*}}
// OUT slice → VTCM copy
// CHECK: tensor.extract_slice
// CHECK: iree_hexagon.stage_to_vtcm %{{.*}}
// CHECK: linalg.matmul
// CHECK: scf.forall.in_parallel
// CHECK: tensor.parallel_insert_slice
// CHECK-NOT: scf.for
// CHECK-NOT: hexagon_vtcm_tiling_config

func.func @configured_matmul(%lhs: tensor<8x8xf32>, %rhs: tensor<8x8xf32>, %out: tensor<8x8xf32>) -> tensor<8x8xf32> {
  %0 = linalg.matmul {hexagon_vtcm_tiling_config = #iree_hexagon.vtcm_tiling_config<tile_sizes = [4, 4, 4]>} ins(%lhs, %rhs : tensor<8x8xf32>, tensor<8x8xf32>) outs(%out : tensor<8x8xf32>) -> tensor<8x8xf32>
  return %0 : tensor<8x8xf32>
}

// -----

// TODO: This behavior needs to be revisited in the future, since it is not ideal for the copy to be inside the body of the forall, given that it does not depend on it. It should be hoisted instead.

// Mixed case: the M dimension is already fully covered, so the pass leaves the
// LHS whole-tensor operand in place and only stages the RHS and output slices.
// RHS [4x8] with tile_sizes = [4, 4, 4]: compose through (K,N) gives [4,4]
//   which does not equal shape [4x8] → per-tile staging.
// OUT [4x8] with tile_sizes = [4, 4, 4]: compose through (M,N) gives [4,4]
//   which does not equal shape [4x8] → per-tile staging.

// CHECK: func.func @configured_matmul_mixed(%[[ARG0:[a-zA-Z0-9_]+]]
// CHECK: scf.forall
// LHS fits entirely, there should be no extract_slice for it (canonicalized), but the copy to vtcm should appear regardless
// CHECK-NOT tensor.extract_slice
// CHECK: iree_hexagon.stage_to_vtcm %[[ARG0]]
// CHECK: tensor.extract_slice
// CHECK: iree_hexagon.stage_to_vtcm %{{.*}}
// CHECK: tensor.extract_slice
// CHECK: iree_hexagon.stage_to_vtcm %{{.*}}
// CHECK: linalg.matmul ins(%{{.*}}, %{{.*}} : tensor<4x4xf32>, tensor<4x4xf32>)
// CHECK: scf.forall.in_parallel
// CHECK: tensor.parallel_insert_slice

func.func @configured_matmul_mixed(%lhs: tensor<4x4xf32>, %rhs: tensor<4x8xf32>, %out: tensor<4x8xf32>) -> tensor<4x8xf32> {
  %0 = linalg.matmul {hexagon_vtcm_tiling_config = #iree_hexagon.vtcm_tiling_config<tile_sizes = [4, 4, 4]>} ins(%lhs, %rhs : tensor<4x4xf32>, tensor<4x8xf32>) outs(%out : tensor<4x8xf32>) -> tensor<4x8xf32>
  return %0 : tensor<4x8xf32>
}


// -----

// Softmax function extracted from an attention layer.
// This test makes sure that copies are properly spawned and ordered 
// and that all operations are included into the tile-wide tiling

// No fill or generic before the forall: the producer chain moved inside.
// CHECK-NOT: linalg.fill
// CHECK-NOT: linalg.generic
// CHECK: scf.forall
// Inside: per-tile input extracted and staged to VTCM for the producer chain.
// CHECK: tensor.extract_slice
// Inside: fill_neg_inf and global_max stay inside the tiled region.
// CHECK: iree_hexagon.stage_to_vtcm %{{.*}}
// CHECK: linalg.fill
// CHECK: linalg.generic
// Then the exp+sum init and reduction also remain inside the loop.
// CHECK: linalg.fill
// CHECK: linalg.generic
// The mask slice is copied right before normalize.
// CHECK: tensor.extract_slice
// CHECK: iree_hexagon.stage_to_vtcm %{{.*}}
// The empty-backed output tile is staged as an empty marker instead.
// CHECK: iree_hexagon.vtcm_empty() : tensor<4x64x1024xf32>
// CHECK: linalg.generic
// This should now be the end of the forall region
// CHECK: scf.forall.in_parallel
// And finally the result is stored
// CHECK: tensor.parallel_insert_slice

func.func @main$async_dispatch_2_softmax_4x1024x1024xf32_generic() attributes {translation_info = #iree_codegen.translation_info<pipeline = #iree_cpu.pipeline<DoubleTilingExpert>>} {
  %cst = arith.constant 0xFFC00000 : f32
  %cst_0 = arith.constant 0.000000e+00 : f32
  %c0 = arith.constant 0 : index
  %c16777216 = arith.constant 16777216 : index
  %c16781312 = arith.constant 16781312 : index
  %0 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(0) alignment(64) offset(%c0) flags("ReadOnly|Indirect") : !iree_tensor_ext.dispatch.tensor<readonly:tensor<4x1024x1024xf32>>
  %1 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(0) alignment(64) offset(%c16777216) flags("ReadOnly|Indirect") : !iree_tensor_ext.dispatch.tensor<readonly:tensor<4x1024xi8>>
  %2 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(1) alignment(64) offset(%c16781312) flags(Indirect) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<4x1024x1024xf32>>
  %3 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0, 0], sizes = [4, 1024, 1024], strides = [1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<4x1024x1024xf32>> -> tensor<4x1024x1024xf32>
  %4 = iree_tensor_ext.dispatch.tensor.load %1, offsets = [0, 0], sizes = [4, 1024], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<4x1024xi8>> -> tensor<4x1024xi8>
  %5 = tensor.empty() : tensor<4x1024x1024xf32>
  %6 = tensor.empty() : tensor<4x1024xf32>
  %7 = linalg.fill {lowering_config = #iree_cpu.lowering_config<vector_common_parallel = [1, 32]>} ins(%cst : f32) outs(%6 : tensor<4x1024xf32>) -> tensor<4x1024xf32>
  %8 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%3 : tensor<4x1024x1024xf32>) outs(%7 : tensor<4x1024xf32>) attrs =  {lowering_config = #iree_cpu.lowering_config<vector_common_parallel = [1, 32, 0], vector_reduction = [0, 0, 8]>} {
  ^bb0(%in: f32, %out: f32):
    %12 = arith.maxnumf %in, %out : f32
    linalg.yield %12 : f32
  } -> tensor<4x1024xf32>
  %9 = linalg.fill {lowering_config = #iree_cpu.lowering_config<vector_common_parallel = [1, 32]>} ins(%cst_0 : f32) outs(%6 : tensor<4x1024xf32>) -> tensor<4x1024xf32>
  %10 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%3, %8 : tensor<4x1024x1024xf32>, tensor<4x1024xf32>) outs(%9 : tensor<4x1024xf32>) attrs =  {hexagon_vtcm_tiling_config = #iree_hexagon.vtcm_tiling_config<tile_sizes = [4, 64, 1024]>, lowering_config = #iree_cpu.lowering_config<distribution = [1, 32, 0], vector_common_parallel = [1, 32, 0], vector_reduction = [0, 0, 8]>} {
  ^bb0(%in: f32, %in_1: f32, %out: f32):
    %12 = arith.subf %in, %in_1 : f32
    %13 = math.exp %12 : f32
    %14 = arith.addf %13, %out : f32
    linalg.yield %14 : f32
  } -> tensor<4x1024xf32>
  %11 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%4, %3, %8, %10 : tensor<4x1024xi8>, tensor<4x1024x1024xf32>, tensor<4x1024xf32>, tensor<4x1024xf32>) outs(%5 : tensor<4x1024x1024xf32>) attrs =  {lowering_config = #iree_cpu.lowering_config<vector_common_parallel = [1, 1, 32]>} {
  ^bb0(%in: i8, %in_1: f32, %in_2: f32, %in_3: f32, %out: f32):
    %12 = arith.subf %in_1, %in_2 : f32
    %13 = math.exp %12 : f32
    %14 = arith.divf %13, %in_3 : f32
    %15 = arith.trunci %in : i8 to i1
    %16 = arith.select %15, %cst_0, %14 : f32
    linalg.yield %16 : f32
  } -> tensor<4x1024x1024xf32>
  iree_tensor_ext.dispatch.tensor.store %11, %2, offsets = [0, 0, 0], sizes = [4, 1024, 1024], strides = [1, 1, 1] : tensor<4x1024x1024xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<4x1024x1024xf32>>
  return
}

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>

// Partial/remainder tile: The input is staged with stage_to_vtcm on a
// dynamic tensor; the empty-backed output must use vtcm_empty carrying
// explicit runtime-size operands — not a bare vtcm_empty()
//
// CHECK-LABEL: func.func @configured_partial_tile(
// CHECK: scf.forall
// CHECK: iree_hexagon.stage_to_vtcm %{{.*}} : tensor<?x?xf32>
// CHECK: iree_hexagon.vtcm_empty({{.+}}) : tensor<?x?xf32>
// CHECK: linalg.generic
// CHECK: scf.forall.in_parallel
// CHECK: tensor.parallel_insert_slice

func.func @configured_partial_tile(%arg0: tensor<6x6xf32>) -> tensor<6x6xf32> {
  %empty = tensor.empty() : tensor<6x6xf32>
  %0 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%arg0 : tensor<6x6xf32>) outs(%empty : tensor<6x6xf32>) attrs = {hexagon_vtcm_tiling_config = #iree_hexagon.vtcm_tiling_config<tile_sizes = [4, 4]>} {
  ^bb0(%in: f32, %out: f32):
    %1 = arith.addf %in, %out : f32
    linalg.yield %1 : f32
  } -> tensor<6x6xf32>
  return %0 : tensor<6x6xf32>
}

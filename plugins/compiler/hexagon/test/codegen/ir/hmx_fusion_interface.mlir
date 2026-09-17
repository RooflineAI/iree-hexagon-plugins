// RUN: iree-opt --iree-load-plugin=hexagon=$ROOF_HEXAGON_COMPILER_PLUGIN \
// RUN:   --pass-pipeline='builtin.module(func.func(iree-llvmcpu-tile-and-fuse-producer-consumer{tiling-level=vector_common_parallel}))' \
// RUN:   --split-input-file %s | FileCheck %s

// These tests exercise the HMX producer- and consumer-fusion mappings directly.

#full_tile_config = #iree_cpu.lowering_config<
  distribution = [0, 0, 0, 0, 0],
  vector_common_parallel = [1, 1, 0, 0, 0]>

// CHECK-LABEL: func.func @fuse_complete_physical_tile(
// CHECK: %[[RESULT:.+]] = scf.forall
// CHECK:   %[[MATMUL:.+]] = iree_hexagon.hmx.tensor_matmul
// CHECK:   %[[CONSUMER:.+]] = linalg.generic
// CHECK-SAME: ins(%[[MATMUL]] : tensor<1x1x16x32x2xf16>)
// CHECK:   tensor.parallel_insert_slice %[[CONSUMER]]
// CHECK: return %[[RESULT]]
func.func @fuse_complete_physical_tile(
    %lhs: tensor<2x1x16x32x2xf16>,
    %rhs: tensor<1x1x16x32x2xf16>,
    %acc: tensor<2x1x16x32x2xf16>,
    %consumer_init: tensor<2x1x16x32x2xf16>)
    -> tensor<2x1x16x32x2xf16> {
  %product = iree_hexagon.hmx.tensor_matmul
      ins(%lhs, %rhs : tensor<2x1x16x32x2xf16>,
                       tensor<1x1x16x32x2xf16>)
      outs(%acc : tensor<2x1x16x32x2xf16>)
      -> tensor<2x1x16x32x2xf16>
  %consumer = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2, d3, d4) ->
                                  (d0, d1, d2, d3, d4)>,
                       affine_map<(d0, d1, d2, d3, d4) ->
                                  (d0, d1, d2, d3, d4)>],
      iterator_types = ["parallel", "parallel", "parallel", "parallel",
                        "parallel"],
      lowering_config = #full_tile_config}
      ins(%product : tensor<2x1x16x32x2xf16>)
      outs(%consumer_init : tensor<2x1x16x32x2xf16>) {
    ^bb0(%in: f16, %out: f16):
      %sum = arith.addf %in, %in : f16
      linalg.yield %sum : f16
  } -> tensor<2x1x16x32x2xf16>
  return %consumer : tensor<2x1x16x32x2xf16>
}

// -----

#partial_tile_config = #iree_cpu.lowering_config<
  distribution = [0, 0, 0, 0, 0],
  vector_common_parallel = [1, 1, 8, 32, 2]>

// A partial trailing dimension is not a complete physical HMX tile, so the
// matmul producer must remain outside the tiled consumer.
//
// CHECK-LABEL: func.func @reject_partial_physical_tile(
// CHECK: %[[MATMUL:.+]] = iree_hexagon.hmx.tensor_matmul
// CHECK: %[[RESULT:.+]] = scf.forall
// CHECK:   %[[SLICE:.+]] = tensor.extract_slice %[[MATMUL]]
// CHECK-SAME: [1, 1, 8, 32, 2]
// CHECK:   %[[CONSUMER:.+]] = linalg.generic
// CHECK-SAME: ins(%[[SLICE]] : tensor<1x1x8x32x2xf16>)
// CHECK:   tensor.parallel_insert_slice %[[CONSUMER]]
// CHECK: return %[[RESULT]]
func.func @reject_partial_physical_tile(
    %lhs: tensor<2x1x16x32x2xf16>,
    %rhs: tensor<1x1x16x32x2xf16>,
    %acc: tensor<2x1x16x32x2xf16>,
    %consumer_init: tensor<2x1x16x32x2xf16>)
    -> tensor<2x1x16x32x2xf16> {
  %product = iree_hexagon.hmx.tensor_matmul
      ins(%lhs, %rhs : tensor<2x1x16x32x2xf16>,
                       tensor<1x1x16x32x2xf16>)
      outs(%acc : tensor<2x1x16x32x2xf16>)
      -> tensor<2x1x16x32x2xf16>
  %consumer = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2, d3, d4) ->
                                  (d0, d1, d2, d3, d4)>,
                       affine_map<(d0, d1, d2, d3, d4) ->
                                  (d0, d1, d2, d3, d4)>],
      iterator_types = ["parallel", "parallel", "parallel", "parallel",
                        "parallel"],
      lowering_config = #partial_tile_config}
      ins(%product : tensor<2x1x16x32x2xf16>)
      outs(%consumer_init : tensor<2x1x16x32x2xf16>) {
    ^bb0(%in: f16, %out: f16):
      %sum = arith.addf %in, %in : f16
      linalg.yield %sum : f16
  } -> tensor<2x1x16x32x2xf16>
  return %consumer : tensor<2x1x16x32x2xf16>
}

// -----

#logical_tile_config = #iree_cpu.lowering_config<
  distribution = [0, 0],
  vector_common_parallel = [32, 32]>

// CHECK-LABEL: func.func @fuse_unpack_into_logical_consumer(
// CHECK: %[[RESULT:.+]] = scf.forall
// CHECK:   %[[SOURCE_TILE:.+]] = tensor.extract_slice %{{.+}}[{{.+}}, {{.+}}, 0, 0, 0] [1, 1, 16, 32, 2] [1, 1, 1, 1, 1]
// CHECK:   %[[DEST_TILE:.+]] = tensor.extract_slice %{{.+}}[{{.+}}, {{.+}}] [32, 32] [1, 1]
// CHECK:   %[[UNPACKED:.+]] = iree_hexagon.hmx.tensor_unpack
// CHECK-SAME: ins(%[[SOURCE_TILE]] : tensor<16x32x2xf16>)
// CHECK-SAME: outs(%[[DEST_TILE]] : tensor<32x32xf32>)
// CHECK:   %[[CONSUMER:.+]] = linalg.generic
// CHECK-SAME: ins(%[[UNPACKED]] : tensor<32x32xf32>)
// CHECK:   tensor.parallel_insert_slice %[[CONSUMER]]
// CHECK: return %[[RESULT]]
func.func @fuse_unpack_into_logical_consumer(
    %source: tensor<2x3x16x32x2xf16>,
    %unpack_init: tensor<64x96xf32>,
    %consumer_init: tensor<64x96xf32>) -> tensor<64x96xf32> {
  %unpacked = iree_hexagon.hmx.tensor_unpack
      ins(%source : tensor<2x3x16x32x2xf16>)
      outs(%unpack_init : tensor<64x96xf32>)
      {dim = 0 : i64}
      -> tensor<64x96xf32>
  %consumer = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                       affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"],
      lowering_config = #logical_tile_config}
      ins(%unpacked : tensor<64x96xf32>)
      outs(%consumer_init : tensor<64x96xf32>) {
    ^bb0(%in: f32, %out: f32):
      %sum = arith.addf %in, %in : f32
      linalg.yield %sum : f32
  } -> tensor<64x96xf32>
  return %consumer : tensor<64x96xf32>
}

// -----

#packed_tile_config = #iree_cpu.lowering_config<
  distribution = [0, 0, 0, 0, 0],
  vector_common_parallel = [1, 1, 0, 0, 0]>

// CHECK-LABEL: func.func @fuse_unpack_with_packed_producer(
// CHECK: %[[RESULT:.+]] = scf.forall
// CHECK:   %[[PRODUCER:.+]] = linalg.generic
// CHECK:   %[[DEST_TILE:.+]] = tensor.extract_slice %{{.+}}[{{.+}}, {{.+}}] [32, 32] [1, 1]
// CHECK:   %[[UNPACKED:.+]] = iree_hexagon.hmx.tensor_unpack
// CHECK-SAME: ins(%[[PRODUCER]] : tensor<1x1x16x32x2xf16>)
// CHECK-SAME: outs(%[[DEST_TILE]] : tensor<32x32xf32>)
// CHECK:   tensor.parallel_insert_slice %[[UNPACKED]]
// CHECK: return %[[RESULT]]
func.func @fuse_unpack_with_packed_producer(
    %source: tensor<2x3x16x32x2xf16>,
    %producer_init: tensor<2x3x16x32x2xf16>,
    %dest: tensor<64x96xf32>) -> tensor<64x96xf32> {
  %producer = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2, d3, d4) ->
                                  (d0, d1, d2, d3, d4)>,
                       affine_map<(d0, d1, d2, d3, d4) ->
                                  (d0, d1, d2, d3, d4)>],
      iterator_types = ["parallel", "parallel", "parallel", "parallel",
                        "parallel"],
      lowering_config = #packed_tile_config}
      ins(%source : tensor<2x3x16x32x2xf16>)
      outs(%producer_init : tensor<2x3x16x32x2xf16>) {
    ^bb0(%in: f16, %out: f16):
      %sum = arith.addf %in, %in : f16
      linalg.yield %sum : f16
  } -> tensor<2x3x16x32x2xf16>
  %unpacked = iree_hexagon.hmx.tensor_unpack
      ins(%producer : tensor<2x3x16x32x2xf16>)
      outs(%dest : tensor<64x96xf32>)
      {dim = 0 : i64}
      -> tensor<64x96xf32>
  return %unpacked : tensor<64x96xf32>
}

// -----

#partial_packed_tile_config = #iree_cpu.lowering_config<
  distribution = [0, 0, 0, 0, 0],
  vector_common_parallel = [0, 0, 8, 0, 0]>

// A slice of the trailing physical layout is not a complete HMX tile, so the
// unpack operation must remain outside the producer's tiled loop.
//
// CHECK-LABEL: func.func @reject_partial_unpack_producer_tile(
// CHECK: %[[PRODUCER:.+]] = scf.forall
// CHECK:   %[[PARTIAL:.+]] = linalg.generic
// CHECK:   tensor.parallel_insert_slice %[[PARTIAL]]
// CHECK: %[[UNPACKED:.+]] = iree_hexagon.hmx.tensor_unpack
// CHECK-SAME: ins(%[[PRODUCER]] : tensor<1x1x16x32x2xf16>)
// CHECK: return %[[UNPACKED]]
func.func @reject_partial_unpack_producer_tile(
    %source: tensor<1x1x16x32x2xf16>,
    %producer_init: tensor<1x1x16x32x2xf16>,
    %dest: tensor<32x32xf32>) -> tensor<32x32xf32> {
  %producer = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2, d3, d4) ->
                                  (d0, d1, d2, d3, d4)>,
                       affine_map<(d0, d1, d2, d3, d4) ->
                                  (d0, d1, d2, d3, d4)>],
      iterator_types = ["parallel", "parallel", "parallel", "parallel",
                        "parallel"],
      lowering_config = #partial_packed_tile_config}
      ins(%source : tensor<1x1x16x32x2xf16>)
      outs(%producer_init : tensor<1x1x16x32x2xf16>) {
    ^bb0(%in: f16, %out: f16):
      %sum = arith.addf %in, %in : f16
      linalg.yield %sum : f16
  } -> tensor<1x1x16x32x2xf16>
  %unpacked = iree_hexagon.hmx.tensor_unpack
      ins(%producer : tensor<1x1x16x32x2xf16>)
      outs(%dest : tensor<32x32xf32>)
      {dim = 0 : i64}
      -> tensor<32x32xf32>
  return %unpacked : tensor<32x32xf32>
}

// -----

#ragged_logical_tile_config = #iree_cpu.lowering_config<
  distribution = [0, 0], vector_common_parallel = [32, 32]>

// The final logical consumer tile is smaller than 32x32, but it is the exact
// clipped remainder of one aligned HMX tile and may therefore fuse.
// CHECK-LABEL: func.func @fuse_clipped_ragged_unpack_tile(
// CHECK:       scf.forall
// CHECK:         iree_hexagon.hmx.tensor_unpack
// CHECK-SAME:    outs(%{{.+}} : tensor<?x?xf32>)
func.func @fuse_clipped_ragged_unpack_tile(
    %source: tensor<2x2x16x32x2xf16>,
    %unpack_init: tensor<35x37xf32>,
    %consumer_init: tensor<35x37xf32>) -> tensor<35x37xf32> {
  %unpacked = iree_hexagon.hmx.tensor_unpack
      ins(%source : tensor<2x2x16x32x2xf16>)
      outs(%unpack_init : tensor<35x37xf32>)
      {dim = 0 : i64} -> tensor<35x37xf32>
  %consumer = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                       affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"],
      lowering_config = #ragged_logical_tile_config}
      ins(%unpacked : tensor<35x37xf32>)
      outs(%consumer_init : tensor<35x37xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
  } -> tensor<35x37xf32>
  return %consumer : tensor<35x37xf32>
}

// -----

#partial_logical_tile_config = #iree_cpu.lowering_config<
  distribution = [0, 0], vector_common_parallel = [16, 32]>

// A deliberately partial interior tile does not correspond to a complete HMX
// tile, so unpack stays outside the consumer loop.
// CHECK-LABEL: func.func @reject_partial_logical_unpack_tile(
// CHECK:       %[[UNPACKED:.+]] = iree_hexagon.hmx.tensor_unpack
// CHECK:       scf.forall
// CHECK:         tensor.extract_slice %[[UNPACKED]]
func.func @reject_partial_logical_unpack_tile(
    %source: tensor<2x2x16x32x2xf16>,
    %unpack_init: tensor<64x64xf32>,
    %consumer_init: tensor<64x64xf32>) -> tensor<64x64xf32> {
  %unpacked = iree_hexagon.hmx.tensor_unpack
      ins(%source : tensor<2x2x16x32x2xf16>)
      outs(%unpack_init : tensor<64x64xf32>)
      {dim = 0 : i64} -> tensor<64x64xf32>
  %consumer = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                       affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"],
      lowering_config = #partial_logical_tile_config}
      ins(%unpacked : tensor<64x64xf32>)
      outs(%consumer_init : tensor<64x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
  } -> tensor<64x64xf32>
  return %consumer : tensor<64x64xf32>
}

// -----

#oversized_logical_tile_config = #iree_cpu.lowering_config<
  distribution = [0, 0], vector_common_parallel = [64, 32]>

// CHECK-LABEL: func.func @reject_oversized_logical_unpack_tile(
// CHECK:       %[[UNPACKED:.+]] = iree_hexagon.hmx.tensor_unpack
// CHECK:       scf.forall
// CHECK:         tensor.extract_slice %[[UNPACKED]]
func.func @reject_oversized_logical_unpack_tile(
    %source: tensor<2x2x16x32x2xf16>,
    %unpack_init: tensor<64x64xf32>,
    %consumer_init: tensor<64x64xf32>) -> tensor<64x64xf32> {
  %unpacked = iree_hexagon.hmx.tensor_unpack
      ins(%source : tensor<2x2x16x32x2xf16>)
      outs(%unpack_init : tensor<64x64xf32>)
      {dim = 0 : i64} -> tensor<64x64xf32>
  %consumer = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                       affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"],
      lowering_config = #oversized_logical_tile_config}
      ins(%unpacked : tensor<64x64xf32>)
      outs(%consumer_init : tensor<64x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
  } -> tensor<64x64xf32>
  return %consumer : tensor<64x64xf32>
}

// -----

#crossing_logical_tile_config = #iree_cpu.lowering_config<
  distribution = [0, 0], vector_common_parallel = [32, 32]>

// The consumer requests [16,48) from unpack, crossing two physical HMX tiles.
// CHECK-LABEL: func.func @reject_crossing_logical_unpack_tile(
// CHECK:       %[[UNPACKED:.+]] = iree_hexagon.hmx.tensor_unpack
// CHECK:       scf.forall
// CHECK:         tensor.extract_slice %[[UNPACKED]]
func.func @reject_crossing_logical_unpack_tile(
    %source: tensor<2x2x16x32x2xf16>,
    %unpack_init: tensor<64x64xf32>,
    %consumer_init: tensor<48x64xf32>) -> tensor<48x64xf32> {
  %unpacked = iree_hexagon.hmx.tensor_unpack
      ins(%source : tensor<2x2x16x32x2xf16>)
      outs(%unpack_init : tensor<64x64xf32>)
      {dim = 0 : i64} -> tensor<64x64xf32>
  %consumer = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0 + 16, d1)>,
                       affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"],
      lowering_config = #crossing_logical_tile_config}
      ins(%unpacked : tensor<64x64xf32>)
      outs(%consumer_init : tensor<48x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
  } -> tensor<48x64xf32>
  return %consumer : tensor<48x64xf32>
}

// -----

#root_tile_config = #iree_cpu.lowering_config<
  distribution = [0, 0], vector_common_parallel = [1, 1]>

// The shipped HMX pipeline anchors on `hmx.tensor_unpack`: the matmul carries
// no lowering config and is pulled in as a producer, and the logical
// destination is ragged (and dynamic along N after workgroup distribution).
// Nothing may be left behind outside the loop. That regresses if the unpack
// derives its result tile extents from its own result instead of from `dest`:
// the resulting `tensor.dim` keeps the untiled full-grid matmul and unpack
// alive inside the new loop until a later canonicalization folds it away.
// CHECK-LABEL: func.func @fuse_matmul_into_ragged_unpack_root(
// CHECK-NOT:   iree_hexagon.hmx.tensor_matmul
// CHECK-NOT:   iree_hexagon.hmx.tensor_unpack
// CHECK:       scf.forall
// CHECK:         %[[TILE:.+]] = iree_hexagon.hmx.tensor_matmul
// CHECK-SAME:      outs(%{{.+}} : tensor<1x1x16x32x2xf16>)
// CHECK:         iree_hexagon.hmx.tensor_unpack
// CHECK-SAME:      outs(%{{.+}} : tensor<?x?xf32>)
func.func @fuse_matmul_into_ragged_unpack_root(
    %lhs: tensor<2x3x16x32x2xf16>,
    %rhs: tensor<3x2x16x32x2xf16>,
    %acc: tensor<2x2x16x32x2xf16>,
    %dest: tensor<35x?xf32>) -> tensor<35x?xf32> {
  %product = iree_hexagon.hmx.tensor_matmul
      ins(%lhs, %rhs : tensor<2x3x16x32x2xf16>, tensor<3x2x16x32x2xf16>)
      outs(%acc : tensor<2x2x16x32x2xf16>)
      -> tensor<2x2x16x32x2xf16>
  %unpacked = iree_hexagon.hmx.tensor_unpack
      ins(%product : tensor<2x2x16x32x2xf16>)
      outs(%dest : tensor<35x?xf32>)
      {dim = 0 : i64, lowering_config = #root_tile_config}
      -> tensor<35x?xf32>
  return %unpacked : tensor<35x?xf32>
}

// -----

#packed_tile_config = #iree_cpu.lowering_config<
  distribution = [0, 0, 0, 0, 0],
  vector_common_parallel = [1, 1, 0, 0, 0]>

// Consumer fusion invokes `getResultTilePosition` on the already-tiled unpack,
// which can no longer see the untiled 35x37 extent it would have to clip
// against. A ragged destination therefore declines the fusion rather than
// clipping the boundary tile a second time; the aligned counterpart is
// @fuse_unpack_with_packed_producer above. Ragged destinations still fuse in
// the producer direction, where the hook runs on the untiled op.
// CHECK-LABEL: func.func @reject_ragged_unpack_consumer_fusion(
// CHECK:       %[[PRODUCER:.+]] = scf.forall
// CHECK:         linalg.generic
// CHECK:       iree_hexagon.hmx.tensor_unpack
// CHECK-SAME:    ins(%[[PRODUCER]] : tensor<2x2x16x32x2xf16>)
// CHECK-SAME:    outs(%{{.+}} : tensor<35x37xf32>)
func.func @reject_ragged_unpack_consumer_fusion(
    %source: tensor<2x2x16x32x2xf16>,
    %producer_init: tensor<2x2x16x32x2xf16>,
    %dest: tensor<35x37xf32>) -> tensor<35x37xf32> {
  %producer = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2, d3, d4) ->
                                  (d0, d1, d2, d3, d4)>,
                       affine_map<(d0, d1, d2, d3, d4) ->
                                  (d0, d1, d2, d3, d4)>],
      iterator_types = ["parallel", "parallel", "parallel", "parallel",
                        "parallel"],
      lowering_config = #packed_tile_config}
      ins(%source : tensor<2x2x16x32x2xf16>)
      outs(%producer_init : tensor<2x2x16x32x2xf16>) {
    ^bb0(%in: f16, %out: f16):
      %sum = arith.addf %in, %in : f16
      linalg.yield %sum : f16
  } -> tensor<2x2x16x32x2xf16>
  %unpacked = iree_hexagon.hmx.tensor_unpack
      ins(%producer : tensor<2x2x16x32x2xf16>)
      outs(%dest : tensor<35x37xf32>)
      {dim = 0 : i64}
      -> tensor<35x37xf32>
  return %unpacked : tensor<35x37xf32>
}

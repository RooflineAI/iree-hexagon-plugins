// RUN: iree-opt \
// RUN:   --pass-pipeline='builtin.module(func.func(iree-llvmcpu-tile{tiling-level=vector_common_parallel}))' \
// RUN:   --split-input-file %s | FileCheck %s

// These tests exercise the regular tiling paths of the HMX tensor operations.

// Tiling coordinates count physical HMX tiles, so [1, 1] corresponds to a
// logical 32x32 output tile.
#tile_config = #iree_cpu.lowering_config<vector_common_parallel = [1, 1]>

// CHECK-LABEL: func.func @tile_unpack_grid(
// CHECK-DAG: %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG: %[[C1:.+]] = arith.constant 1 : index
// CHECK-DAG: %[[C2:.+]] = arith.constant 2 : index
// CHECK-DAG: %[[C3:.+]] = arith.constant 3 : index
// CHECK: scf.for %[[M:.+]] = %[[C0]] to %[[C2]] step %[[C1]]
// CHECK:   scf.for %[[N:.+]] = %[[C0]] to %[[C3]] step %[[C1]]
// CHECK:     %[[SOURCE_TILE:.+]] = tensor.extract_slice %{{.+}}[%[[M]], %[[N]], 0, 0, 0] [1, 1, 16, 32, 2] [1, 1, 1, 1, 1]
// CHECK:     %[[DEST_TILE:.+]] = tensor.extract_slice %{{.+}}[{{.+}}, {{.+}}] [32, 32] [1, 1]
// CHECK:     %[[TILED:.+]] = iree_hexagon.hmx.tensor_unpack
// CHECK-SAME:  ins(%[[SOURCE_TILE]] : tensor<16x32x2xf16>)
// CHECK-SAME:  outs(%[[DEST_TILE]] : tensor<32x32xf32>)
// CHECK:     tensor.insert_slice %[[TILED]]
func.func @tile_unpack_grid(
    %source: tensor<2x3x16x32x2xf16>,
    %dest: tensor<64x96xf32>) -> tensor<64x96xf32> {
  %result = iree_hexagon.hmx.tensor_unpack
      ins(%source : tensor<2x3x16x32x2xf16>)
      outs(%dest : tensor<64x96xf32>)
      {dim = 0 : i64, lowering_config = #tile_config}
      -> tensor<64x96xf32>
  return %result : tensor<64x96xf32>
}

// -----

#tile_config = #iree_cpu.lowering_config<vector_common_parallel = [1, 1]>

// The physical source grid remains 2x2, but logical destination tiles are
// clipped to the remaining rows and columns. This prevents the final tiles from
// constructing out-of-bounds 32x32 slices of a 35x37 destination.

// CHECK-LABEL: func.func @tile_ragged_unpack_grid(
// CHECK:       scf.for
// CHECK:         scf.for
// CHECK-DAG:       %[[M_SIZE:.+]] = affine.min
// CHECK-DAG:       %[[N_SIZE:.+]] = affine.min
// CHECK:           %[[DEST_TILE:.+]] = tensor.extract_slice %{{.+}}[{{.+}}, {{.+}}] [%[[M_SIZE]], %[[N_SIZE]]] [1, 1] : tensor<35x37xf32> to tensor<?x?xf32>
// CHECK:           %[[TILED:.+]] = iree_hexagon.hmx.tensor_unpack
// CHECK-SAME:        outs(%[[DEST_TILE]] : tensor<?x?xf32>)
// CHECK-DAG:       %[[INSERT_M_SIZE:.+]] = affine.min
// CHECK-DAG:       %[[INSERT_N_SIZE:.+]] = affine.min
// CHECK:           tensor.insert_slice %[[TILED]] into %{{.+}}[{{.+}}, {{.+}}] [%[[INSERT_M_SIZE]], %[[INSERT_N_SIZE]]] [1, 1]
func.func @tile_ragged_unpack_grid(
    %source: tensor<2x2x16x32x2xf16>,
    %dest: tensor<35x37xf32>) -> tensor<35x37xf32> {
  %result = iree_hexagon.hmx.tensor_unpack
      ins(%source : tensor<2x2x16x32x2xf16>)
      outs(%dest : tensor<35x37xf32>)
      {dim = 0 : i64, lowering_config = #tile_config}
      -> tensor<35x37xf32>
  return %result : tensor<35x37xf32>
}

// -----

#tile_config = #iree_cpu.lowering_config<vector_common_parallel = [1, 1]>

// CHECK-LABEL: func.func @tile_matmul_grid(
// CHECK: scf.for
// CHECK:   scf.for
// CHECK:     %[[LHS_TILE:.+]] = tensor.extract_slice %{{.+}}[{{.+}}, 0, 0, 0, 0] [1, 4, 16, 32, 2] [1, 1, 1, 1, 1]
// CHECK:     %[[RHS_TILE:.+]] = tensor.extract_slice %{{.+}}[0, {{.+}}, 0, 0, 0] [4, 1, 16, 32, 2] [1, 1, 1, 1, 1]
// CHECK:     %[[ACC_TILE:.+]] = tensor.extract_slice %{{.+}}[{{.+}}, {{.+}}, 0, 0, 0] [1, 1, 16, 32, 2] [1, 1, 1, 1, 1]
// CHECK:     %[[TILED:.+]] = iree_hexagon.hmx.tensor_matmul
// CHECK-SAME:  ins(%[[LHS_TILE]], %[[RHS_TILE]]
// CHECK-SAME:  outs(%[[ACC_TILE]]
// CHECK:     tensor.insert_slice %[[TILED]]
func.func @tile_matmul_grid(
    %lhs: tensor<2x4x16x32x2xf16>,
    %rhs: tensor<4x3x16x32x2xf16>,
    %acc: tensor<2x3x16x32x2xf16>) -> tensor<2x3x16x32x2xf16> {
  %result = iree_hexagon.hmx.tensor_matmul
      ins(%lhs, %rhs : tensor<2x4x16x32x2xf16>,
                       tensor<4x3x16x32x2xf16>)
      outs(%acc : tensor<2x3x16x32x2xf16>)
      {lowering_config = #tile_config}
      -> tensor<2x3x16x32x2xf16>
  return %result : tensor<2x3x16x32x2xf16>
}

// -----

#tile_config = #iree_cpu.lowering_config<vector_common_parallel = [1, 1]>

// After workgroup distribution the logical destination of a ragged matmul is
// dynamic along N (e.g. `tensor<1537x?xf32>`), so both boundary clips have to
// be computed at runtime from the destination operand.

// CHECK-LABEL: func.func @tile_dynamic_unpack_dest(
// CHECK-SAME:    %[[SOURCE:[^:]+]]: tensor<2x2x16x32x2xf16>
// CHECK-SAME:    %[[DEST:[^:]+]]: tensor<35x?xf32>
// CHECK:       scf.for
// CHECK:         scf.for
// CHECK-DAG:       %[[N_EXTENT:.+]] = tensor.dim %[[DEST]]
// CHECK-DAG:       %[[M_SIZE:.+]] = affine.min
// CHECK-DAG:       %[[N_SIZE:.+]] = affine.min
// CHECK:           %[[DEST_TILE:.+]] = tensor.extract_slice %{{.+}}[{{.+}}, {{.+}}] [%{{.+}}, %{{.+}}] [1, 1] : tensor<35x?xf32> to tensor<?x?xf32>
// CHECK:           iree_hexagon.hmx.tensor_unpack
// CHECK-SAME:        outs(%[[DEST_TILE]] : tensor<?x?xf32>)
func.func @tile_dynamic_unpack_dest(
    %source: tensor<2x2x16x32x2xf16>,
    %dest: tensor<35x?xf32>) -> tensor<35x?xf32> {
  %result = iree_hexagon.hmx.tensor_unpack
      ins(%source : tensor<2x2x16x32x2xf16>)
      outs(%dest : tensor<35x?xf32>)
      {dim = 0 : i64, lowering_config = #tile_config}
      -> tensor<35x?xf32>
  return %result : tensor<35x?xf32>
}

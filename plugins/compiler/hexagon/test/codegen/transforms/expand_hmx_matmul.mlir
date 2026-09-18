// RUN: iree-opt --iree-load-plugin=hexagon=$ROOF_HEXAGON_COMPILER_PLUGIN \
// RUN:   --pass-pipeline='builtin.module(func.func(iree-hexagon-expand-hmx-matmul))' \
// RUN:   --split-input-file %s | FileCheck %s

// CHECK-LABEL: func.func @expand_multi_k(
// CHECK-DAG:   %[[SCRATCH:.+]] = hexagonmem.alloc() {alignment = 2048 : i64} : memref<16x32x2xf16, 1>
// CHECK-DAG:   %[[CONFIG:.+]] = hexagonmem.alloc() {alignment = 2048 : i64} : memref<2048xi8, 1 : i32>
// CHECK:       iree_hexagon.hmx.acc.setup_read %[[CONFIG]] : memref<2048xi8, 1 : i32>
// CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG:   %[[C512:.+]] = arith.constant 512 : index
// CHECK-DAG:   %[[C32:.+]] = arith.constant 32 : index
// CHECK:       %[[ZERO:.+]] = iree_hexagon.hmx.acc.zero : !iree_hexagon.hmx.acc<32x32xf32>
// CHECK:       %[[ACC:.+]] = scf.for %[[K:.+]] = %[[C0]] to %[[C512]] step %[[C32]] iter_args(%[[ITER:.+]] = %[[ZERO]]) -> (!iree_hexagon.hmx.acc<32x32xf32>) {
// CHECK:         %[[KTILE:.+]] = arith.divui %[[K]], %{{.+}} : index
// CHECK:         %[[LHS_TILE:.+]] = memref.subview %{{.+}}[0, %[[KTILE]], 0, 0, 0] [1, 1, 16, 32, 2] [1, 1, 1, 1, 1]
// CHECK:         %[[RHS_TILE:.+]] = memref.subview %{{.+}}[%[[KTILE]], 0, 0, 0, 0] [1, 1, 16, 32, 2] [1, 1, 1, 1, 1]
// CHECK:         %[[NEXT:.+]] = iree_hexagon.hmx.mma %[[LHS_TILE]], %[[RHS_TILE]], %[[ITER]]
// CHECK:         scf.yield %[[NEXT]]
// CHECK:       }
// CHECK:       iree_hexagon.hmx.acc.read %[[ACC]], %[[SCRATCH]]
// CHECK:       hexagonmem.dealloc %[[CONFIG]] : memref<2048xi8, 1 : i32>
// CHECK:       return
// CHECK-NOT:   iree_hexagon.hmx.matmul
func.func @expand_multi_k() {
  %lhs = hexagonmem.alloc() {alignment = 2048 : i64} : memref<1x16x16x32x2xf16, 1>
  %rhs = hexagonmem.alloc() {alignment = 2048 : i64} : memref<16x1x16x32x2xf16, 1>
  %acc = hexagonmem.alloc() {alignment = 2048 : i64} : memref<16x32x2xf16, 1>
  iree_hexagon.hmx.matmul
      ins(%lhs, %rhs : memref<1x16x16x32x2xf16, 1>, memref<16x1x16x32x2xf16, 1>)
      outs(%acc : memref<16x32x2xf16, 1>)
  return
}

// -----

// CHECK-LABEL: func.func @expand_single_k(
// CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG:   %[[C32:.+]] = arith.constant 32 : index
// CHECK:       %[[ZERO:.+]] = iree_hexagon.hmx.acc.zero : !iree_hexagon.hmx.acc<32x32xf32>
// CHECK:       %[[ACC:.+]] = scf.for %[[K:.+]] = %[[C0]] to %[[C32]] step %{{.+}} iter_args(%[[ITER:.+]] = %[[ZERO]]) -> (!iree_hexagon.hmx.acc<32x32xf32>) {
// CHECK:         iree_hexagon.hmx.mma
// CHECK:         scf.yield
// CHECK:       }
// CHECK:       iree_hexagon.hmx.acc.read %[[ACC]]
func.func @expand_single_k() {
  %lhs = hexagonmem.alloc() {alignment = 2048 : i64} : memref<1x1x16x32x2xf16, 1>
  %rhs = hexagonmem.alloc() {alignment = 2048 : i64} : memref<1x1x16x32x2xf16, 1>
  %acc = hexagonmem.alloc() {alignment = 2048 : i64} : memref<16x32x2xf16, 1>
  iree_hexagon.hmx.matmul
      ins(%lhs, %rhs : memref<1x1x16x32x2xf16, 1>, memref<1x1x16x32x2xf16, 1>)
      outs(%acc : memref<16x32x2xf16, 1>)
  return
}

// -----

// Bufferization preserves the singleton rank-5 output container and may create
// multiple rank-reduced consumer views. Expansion must use its own write view
// without reinstating general M/N grid loops or requiring a unique consumer.

// CHECK-LABEL: func.func @expand_bufferized_singleton_grid(
// CHECK:       %[[CONSUMER_TILE_0:.+]] = memref.subview %[[ACC_GRID:.+]][0, 0, 0, 0, 0] [1, 1, 16, 32, 2]
// CHECK:       %[[CONSUMER_TILE_1:.+]] = memref.subview %[[ACC_GRID]][0, 0, 0, 0, 0] [1, 1, 16, 32, 2]
// CHECK:       %[[WRITE_TILE:.+]] = memref.subview %[[ACC_GRID]][0, 0, 0, 0, 0] [1, 1, 16, 32, 2]
// CHECK:       %[[ACC:.+]] = scf.for
// CHECK:       iree_hexagon.hmx.acc.read %[[ACC]], %[[WRITE_TILE]]
// CHECK:       iree_hexagon.hmx.unpack ins(%[[CONSUMER_TILE_0]]
// CHECK:       iree_hexagon.hmx.unpack ins(%[[CONSUMER_TILE_1]]
// CHECK-NOT:   iree_hexagon.hmx.matmul
func.func @expand_bufferized_singleton_grid(
    %dest_0: memref<32x32xf32, 1>,
    %dest_1: memref<32x32xf32, 1>) {
  %lhs = hexagonmem.alloc() {alignment = 2048 : i64} : memref<1x4x16x32x2xf16, 1>
  %rhs = hexagonmem.alloc() {alignment = 2048 : i64} : memref<4x1x16x32x2xf16, 1>
  %acc_grid = hexagonmem.alloc() {alignment = 2048 : i64} : memref<1x1x16x32x2xf16, 1>
  %acc_tile_0 = memref.subview %acc_grid[0, 0, 0, 0, 0]
      [1, 1, 16, 32, 2] [1, 1, 1, 1, 1]
      : memref<1x1x16x32x2xf16, 1>
      to memref<16x32x2xf16, strided<[64, 2, 1], offset: 0>, 1>
  %acc_tile_1 = memref.subview %acc_grid[0, 0, 0, 0, 0]
      [1, 1, 16, 32, 2] [1, 1, 1, 1, 1]
      : memref<1x1x16x32x2xf16, 1>
      to memref<16x32x2xf16, strided<[64, 2, 1], offset: 0>, 1>
  iree_hexagon.hmx.matmul
      ins(%lhs, %rhs : memref<1x4x16x32x2xf16, 1>,
                       memref<4x1x16x32x2xf16, 1>)
      outs(%acc_grid : memref<1x1x16x32x2xf16, 1>)
  iree_hexagon.hmx.unpack
      ins(%acc_tile_0 : memref<16x32x2xf16, strided<[64, 2, 1], offset: 0>, 1>)
      outs(%dest_0 : memref<32x32xf32, 1>) {dim = 0 : i64}
  iree_hexagon.hmx.unpack
      ins(%acc_tile_1 : memref<16x32x2xf16, strided<[64, 2, 1], offset: 0>, 1>)
      outs(%dest_1 : memref<32x32xf32, 1>) {dim = 0 : i64}
  return
}

// -----

// A singleton grid is sufficient by contract; expansion must not depend on a
// downstream unpack having already materialized the rank-reducing subview.

// CHECK-LABEL: func.func @expand_singleton_grid_without_consumer(
// CHECK:       %[[WRITE_TILE:.+]] = memref.subview %{{.+}}[0, 0, 0, 0, 0] [1, 1, 16, 32, 2]
// CHECK:       iree_hexagon.hmx.acc.read %{{.+}}, %[[WRITE_TILE]]
// CHECK-NOT:   iree_hexagon.hmx.matmul
func.func @expand_singleton_grid_without_consumer() {
  %lhs = hexagonmem.alloc() {alignment = 2048 : i64} : memref<1x4x16x32x2xf16, 1>
  %rhs = hexagonmem.alloc() {alignment = 2048 : i64} : memref<4x1x16x32x2xf16, 1>
  %acc_grid = hexagonmem.alloc() {alignment = 2048 : i64} : memref<1x1x16x32x2xf16, 1>
  iree_hexagon.hmx.matmul
      ins(%lhs, %rhs : memref<1x4x16x32x2xf16, 1>,
                       memref<4x1x16x32x2xf16, 1>)
      outs(%acc_grid : memref<1x1x16x32x2xf16, 1>)
  return
}

// -----

// A downstream rank-reducing view may be nested in another region. Expansion
// creates a separate write view in the matmul block and leaves this consumer
// in place, avoiding cross-block operation-order queries.

// CHECK-LABEL: func.func @expand_with_nested_consumer(
// CHECK-DAG:   %[[ACC_GRID:.+]] = hexagonmem.alloc() {{.*}} : memref<1x1x16x32x2xf16, 1>
// CHECK:       %[[WRITE_TILE:.+]] = memref.subview %[[ACC_GRID]][0, 0, 0, 0, 0] [1, 1, 16, 32, 2]
// CHECK:       iree_hexagon.hmx.acc.read %{{.+}}, %[[WRITE_TILE]]
// CHECK:       scf.if
// CHECK:         %[[CONSUMER_TILE:.+]] = memref.subview %[[ACC_GRID]][0, 0, 0, 0, 0] [1, 1, 16, 32, 2]
// CHECK:         iree_hexagon.hmx.unpack ins(%[[CONSUMER_TILE]]
func.func @expand_with_nested_consumer(
    %condition: i1,
    %dest: memref<32x32xf32, 1>) {
  %lhs = hexagonmem.alloc() {alignment = 2048 : i64} : memref<1x4x16x32x2xf16, 1>
  %rhs = hexagonmem.alloc() {alignment = 2048 : i64} : memref<4x1x16x32x2xf16, 1>
  %acc_grid = hexagonmem.alloc() {alignment = 2048 : i64} : memref<1x1x16x32x2xf16, 1>
  iree_hexagon.hmx.matmul
      ins(%lhs, %rhs : memref<1x4x16x32x2xf16, 1>,
                       memref<4x1x16x32x2xf16, 1>)
      outs(%acc_grid : memref<1x1x16x32x2xf16, 1>)
  scf.if %condition {
    %acc_tile = memref.subview %acc_grid[0, 0, 0, 0, 0]
        [1, 1, 16, 32, 2] [1, 1, 1, 1, 1]
        : memref<1x1x16x32x2xf16, 1>
        to memref<16x32x2xf16, strided<[64, 2, 1], offset: 0>, 1>
    iree_hexagon.hmx.unpack
        ins(%acc_tile : memref<16x32x2xf16, strided<[64, 2, 1], offset: 0>, 1>)
        outs(%dest : memref<32x32xf32, 1>) {dim = 0 : i64}
  }
  return
}

// -----

// CHECK-LABEL: func.func @expand_tiled_subviews(
// CHECK-DAG:   hexagonmem.alloc() {alignment = 2048 : i64} : memref<16x16x16x32x2xf16, 1>
// CHECK-NOT:   hexagonmem.alloc() {alignment = 2048 : i64} : memref<16x32x2xf16, strided
// CHECK-NOT:   hexagonmem.alloc() {alignment = 2048 : i64} : memref<1x16x16x32x2xf16, strided
// CHECK-NOT:   hexagonmem.alloc() {alignment = 2048 : i64} : memref<16x1x16x32x2xf16, strided
// CHECK:       iree_hexagon.hmx.acc.read
// CHECK-NOT:   iree_hexagon.hmx.matmul
func.func @expand_tiled_subviews() {
  // An under-aligned parent allocation is rebuilt; its subviews are preserved.
  %lhs_grid = hexagonmem.alloc() {alignment = 64 : i64} : memref<16x16x16x32x2xf16, 1>
  %rhs_grid = hexagonmem.alloc() {alignment = 2048 : i64} : memref<16x16x16x32x2xf16, 1>
  %acc_grid = hexagonmem.alloc() {alignment = 2048 : i64} : memref<16x16x16x32x2xf16, 1>
  %c0 = arith.constant 0 : index
  %lhs = memref.subview %lhs_grid[%c0, 0, 0, 0, 0]
      [1, 16, 16, 32, 2] [1, 1, 1, 1, 1]
      : memref<16x16x16x32x2xf16, 1>
      to memref<1x16x16x32x2xf16, strided<[16384, 1024, 64, 2, 1], offset: ?>, 1>
  %rhs = memref.subview %rhs_grid[0, %c0, 0, 0, 0]
      [16, 1, 16, 32, 2] [1, 1, 1, 1, 1]
      : memref<16x16x16x32x2xf16, 1>
      to memref<16x1x16x32x2xf16, strided<[16384, 1024, 64, 2, 1], offset: ?>, 1>
  %acc = memref.subview %acc_grid[%c0, %c0, 0, 0, 0]
      [1, 1, 16, 32, 2] [1, 1, 1, 1, 1]
      : memref<16x16x16x32x2xf16, 1>
      to memref<16x32x2xf16, strided<[64, 2, 1], offset: ?>, 1>
  iree_hexagon.hmx.matmul
      ins(%lhs, %rhs
          : memref<1x16x16x32x2xf16, strided<[16384, 1024, 64, 2, 1], offset: ?>, 1>,
            memref<16x1x16x32x2xf16, strided<[16384, 1024, 64, 2, 1], offset: ?>, 1>)
      outs(%acc : memref<16x32x2xf16, strided<[64, 2, 1], offset: ?>, 1>)
  return
}

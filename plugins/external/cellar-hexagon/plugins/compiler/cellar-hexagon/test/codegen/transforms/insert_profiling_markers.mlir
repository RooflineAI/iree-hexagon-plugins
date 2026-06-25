// RUN: iree-opt --iree-load-plugin=cellar_hexagon=$CELLAR_HEXAGON_COMPILER_PLUGIN \
// RUN:   --pass-pipeline='builtin.module(func.func(iree-hexagon-insert-profiling-markers))' \
// RUN:   %s | FileCheck %s

func.func @insert_markers(%src: memref<4xf32>, %dst: memref<4xf32, 1>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %alloc = hexagonmem.alloc() : memref<4xf32, 1>
  hexagonmem.copy %src, %alloc : memref<4xf32> to memref<4xf32, 1>
  scf.for %i = %c0 to %c4 step %c1 {
    %0 = arith.addi %i, %i : index
  }
  return
}

// CHECK-LABEL: func.func @insert_markers(
// CHECK: iree_hexagon.profiling.begin {extra_info = "hexagonmem.alloc", zone_type = 7 : i32}
// CHECK-NEXT: hexagonmem.alloc
// CHECK-NEXT: iree_hexagon.profiling.end
// CHECK: iree_hexagon.profiling.begin {extra_info = "hexagonmem.copy", zone_type = 7 : i32}
// CHECK-NEXT: hexagonmem.copy
// CHECK-NEXT: iree_hexagon.profiling.end
// CHECK: iree_hexagon.profiling.begin {extra_info = "compute.inner_loop", zone_type = 7 : i32}
// CHECK-NEXT: scf.for
// CHECK: }
// CHECK-NEXT: iree_hexagon.profiling.end

func.func @nested_loop_markers(%src: memref<4xf32>, %dst: memref<4xf32, 1>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  scf.for %i = %c0 to %c4 step %c1 {
    hexagonmem.copy %src, %dst : memref<4xf32> to memref<4xf32, 1>
    scf.for %j = %c0 to %c4 step %c1 {
      scf.for %k = %c0 to %c4 step %c1 {
        %0 = arith.addi %i, %k : index
      }
    }
  }
  return
}

// CHECK-LABEL: func.func @nested_loop_markers(
// CHECK-NOT: iree_hexagon.profiling.begin {extra_info = "compute.inner_loop"
// CHECK: scf.for
// CHECK: iree_hexagon.profiling.begin {extra_info = "hexagonmem.copy", zone_type = 7 : i32}
// CHECK-NEXT: hexagonmem.copy
// CHECK-NEXT: iree_hexagon.profiling.end
// CHECK-NEXT: iree_hexagon.profiling.begin {extra_info = "compute.inner_loop", zone_type = 7 : i32}
// CHECK-NEXT: scf.for
// CHECK-NOT: iree_hexagon.profiling.begin
// CHECK: scf.for
// CHECK-NOT: iree_hexagon.profiling.begin
// CHECK: arith.addi
// CHECK: }
// CHECK-NEXT: }
// CHECK-NEXT: iree_hexagon.profiling.end

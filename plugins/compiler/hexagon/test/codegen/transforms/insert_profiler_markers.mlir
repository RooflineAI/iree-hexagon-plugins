// RUN: iree-opt --iree-load-plugin=hexagon=$HEXAGON_COMPILER_PLUGIN \
// RUN:   --pass-pipeline='builtin.module(func.func(iree-hexagon-insert-profiler-markers))' \
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
  hexagonmem.dealloc %alloc : memref<4xf32, 1>
  return
}

// CHECK-LABEL: func.func @insert_markers(
// CHECK: %[[CTX:.*]] = iree_hexagon.get_runtime_state : !iree_hexagon.runtime_state
// CHECK: %[[ALLOC_REC:.*]] = iree_hexagon.profiler.begin %[[CTX]] {extra_info = "hexagonmem.alloc", zone_type = #iree_hexagon.profiler_zone<memory_management>} : !iree_hexagon.runtime_state -> !iree_hexagon.profiler_record
// CHECK-NEXT: hexagonmem.alloc
// CHECK-NEXT: iree_hexagon.profiler.end %[[ALLOC_REC]] : !iree_hexagon.profiler_record
// CHECK: %[[COPY_REC:.*]] = iree_hexagon.profiler.begin %[[CTX]] {extra_info = "hexagonmem.copy", zone_type = #iree_hexagon.profiler_zone<memory_management>} : !iree_hexagon.runtime_state -> !iree_hexagon.profiler_record
// CHECK-NEXT: hexagonmem.copy
// CHECK-NEXT: iree_hexagon.profiler.end %[[COPY_REC]] : !iree_hexagon.profiler_record
// CHECK: %[[LOOP_REC:.*]] = iree_hexagon.profiler.begin %[[CTX]] {extra_info = "compute.inner_loop", zone_type = #iree_hexagon.profiler_zone<marker>} : !iree_hexagon.runtime_state -> !iree_hexagon.profiler_record
// CHECK-NEXT: scf.for
// CHECK: }
// CHECK-NEXT: iree_hexagon.profiler.end %[[LOOP_REC]] : !iree_hexagon.profiler_record
// CHECK: %[[DEALLOC_REC:.*]] = iree_hexagon.profiler.begin %[[CTX]] {extra_info = "hexagonmem.dealloc", zone_type = #iree_hexagon.profiler_zone<memory_management>} : !iree_hexagon.runtime_state -> !iree_hexagon.profiler_record
// CHECK-NEXT: hexagonmem.dealloc
// CHECK-NEXT: iree_hexagon.profiler.end %[[DEALLOC_REC]] : !iree_hexagon.profiler_record

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
// CHECK: %[[CTX:.*]] = iree_hexagon.get_runtime_state : !iree_hexagon.runtime_state
// CHECK-NOT: extra_info = "compute.inner_loop"
// CHECK: scf.for
// CHECK: %[[COPY_REC:.*]] = iree_hexagon.profiler.begin %[[CTX]] {extra_info = "hexagonmem.copy", zone_type = #iree_hexagon.profiler_zone<memory_management>} : !iree_hexagon.runtime_state -> !iree_hexagon.profiler_record
// CHECK-NEXT: hexagonmem.copy
// CHECK-NEXT: iree_hexagon.profiler.end %[[COPY_REC]] : !iree_hexagon.profiler_record
// CHECK-NEXT: %[[LOOP_REC:.*]] = iree_hexagon.profiler.begin %[[CTX]] {extra_info = "compute.inner_loop", zone_type = #iree_hexagon.profiler_zone<marker>} : !iree_hexagon.runtime_state -> !iree_hexagon.profiler_record
// CHECK-NEXT: scf.for
// CHECK-NOT: iree_hexagon.profiler.begin
// CHECK: scf.for
// CHECK-NOT: iree_hexagon.profiler.begin
// CHECK: arith.addi
// CHECK: }
// CHECK-NEXT: }
// CHECK-NEXT: iree_hexagon.profiler.end %[[LOOP_REC]] : !iree_hexagon.profiler_record

func.func @memory_management_markers(%src: memref<4xf32>, %dst: memref<4xf32>) {
  %alloc = memref.alloc() : memref<4xf32>
  memref.copy %src, %alloc : memref<4xf32> to memref<4xf32>
  memref.dealloc %alloc : memref<4xf32>
  return
}

// CHECK-LABEL: func.func @memory_management_markers(
// CHECK: %[[CTX:.*]] = iree_hexagon.get_runtime_state : !iree_hexagon.runtime_state

// CHECK: %[[ALLOC_REC:.*]] = iree_hexagon.profiler.begin %[[CTX]] {extra_info = "kernel_allocation", zone_type = #iree_hexagon.profiler_zone<memory_management>} : !iree_hexagon.runtime_state -> !iree_hexagon.profiler_record
// CHECK-NEXT: memref.alloc
// CHECK-NEXT: iree_hexagon.profiler.end %[[ALLOC_REC]] : !iree_hexagon.profiler_record

// CHECK: %[[COPY_REC:.*]] = iree_hexagon.profiler.begin %[[CTX]] {extra_info = "memref_copy", zone_type = #iree_hexagon.profiler_zone<memory_management>} : !iree_hexagon.runtime_state -> !iree_hexagon.profiler_record
// CHECK-NEXT: memref.copy
// CHECK-NEXT: iree_hexagon.profiler.end %[[COPY_REC]] : !iree_hexagon.profiler_record

// CHECK: %[[FREE_REC:.*]] = iree_hexagon.profiler.begin %[[CTX]] {extra_info = "kernel_free", zone_type = #iree_hexagon.profiler_zone<memory_management>} : !iree_hexagon.runtime_state -> !iree_hexagon.profiler_record
// CHECK-NEXT: memref.dealloc
// CHECK-NEXT: iree_hexagon.profiler.end %[[FREE_REC]] : !iree_hexagon.profiler_record

// memref.alloca allocates on the stack and does not reach a runtime helper, so
// it must not be wrapped.
func.func @alloca_not_wrapped() {
  %alloca = memref.alloca() : memref<4xf32>
  return
}

// CHECK-LABEL: func.func @alloca_not_wrapped(
// CHECK-NOT: iree_hexagon.profiler.begin

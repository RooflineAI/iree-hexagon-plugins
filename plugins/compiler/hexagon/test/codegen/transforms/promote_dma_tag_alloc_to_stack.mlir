// RUN: iree-opt --iree-load-plugin=hexagon=$HEXAGON_COMPILER_PLUGIN \
// RUN:   --pass-pipeline='builtin.module(func.func(iree-hexagon-promote-dma-tag-alloc-to-stack))' \
// RUN:   %s | FileCheck %s

// CHECK-LABEL: func.func @promote_dma_tag_alloc(
// CHECK: %[[TAG:.*]] = memref.alloca() : memref<1xi32>
// CHECK: memref.dma_start %arg0[%{{.*}}], %arg1[%{{.*}}], %{{.*}}, %[[TAG]][%{{.*}}] : memref<4xf32, 1>, memref<4xf32>, memref<1xi32>
// CHECK: memref.dma_wait %[[TAG]][%{{.*}}], %{{.*}} : memref<1xi32>
// CHECK-NOT: memref.dealloc %[[TAG]]

func.func @promote_dma_tag_alloc(%src: memref<4xf32, 1>, %dst: memref<4xf32>) {
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %tag = memref.alloc() : memref<1xi32>
  memref.dma_start %src[%c0], %dst[%c0], %c4, %tag[%c0] : memref<4xf32, 1>, memref<4xf32>, memref<1xi32>
  memref.dma_wait %tag[%c0], %c4 : memref<1xi32>
  memref.dealloc %tag : memref<1xi32>
  return
}

// CHECK-LABEL: func.func @keep_regular_heap_alloc()
// CHECK: %[[HEAP:.*]] = memref.alloc() : memref<1xi32>
// CHECK: memref.dealloc %[[HEAP]] : memref<1xi32>

func.func @keep_regular_heap_alloc() {
  %heap = memref.alloc() : memref<1xi32>
  memref.dealloc %heap : memref<1xi32>
  return
}

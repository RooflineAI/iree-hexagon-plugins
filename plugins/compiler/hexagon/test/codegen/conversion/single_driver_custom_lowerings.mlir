// Verify that the final conversion driver composes the HexagonMem, DMA, and
// HexKL patterns together with the standard LLVM conversion. In particular,
// VTCM memrefs must be converted directly to default-address-space pointers; no
// collapse pass or materialization-cast cleanup is expected after this pass.

// RUN: iree-opt --pass-pipeline='builtin.module(iree-hexagon-convert-to-llvm)' \
// RUN:   %s | FileCheck %s \
// RUN:     --implicit-check-not=builtin.unrealized_conversion_cast \
// RUN:     --implicit-check-not='!llvm.ptr<1>' \
// RUN:     --implicit-check-not=hexagonmem. \
// RUN:     --implicit-check-not=hexkl. \
// RUN:     --implicit-check-not=memref.dma_

module {
  func.func private @memory_and_dma() {
    %c0 = arith.constant 0 : index
    %c4 = arith.constant 4 : index
    %src = memref.alloc() : memref<4xf32>
    %dst = hexagonmem.alloc() {alignment = 128 : i64} : memref<4xf32, 1>
    %tag = memref.alloca() : memref<1xi32>
    memref.dma_start %src[%c0], %dst[%c0], %c4, %tag[%c0]
        : memref<4xf32>, memref<4xf32, 1>, memref<1xi32>
    memref.dma_wait %tag[%c0], %c4 : memref<1xi32>
    hexagonmem.dealloc %dst : memref<4xf32, 1>
    memref.dealloc %src : memref<4xf32>
    return
  }

  func.func private @hexkl_matmul(%lhs: memref<2x4xf16>,
                                  %rhs: memref<4x3xf16>,
                                  %out: memref<2x3xf32>) {
    hexkl.matmul ins(%lhs, %rhs : memref<2x4xf16>, memref<4x3xf16>)
                 outs(%out : memref<2x3xf32>)
    return
  }
}

// CHECK-DAG: llvm.func @hexagon_runtime_alloc_1d(i32, i64, i1) -> !llvm.ptr attributes {hexagon.native_runtime_link}
// CHECK-DAG: llvm.func @hexagon_runtime_free_1d(!llvm.ptr) attributes {hexagon.native_runtime_link}
// CHECK-DAG: llvm.func @hexagon_runtime_dma_start(!llvm.ptr, i32, !llvm.ptr, i32, i32, i32, i32, !llvm.ptr) -> i32 attributes {hexagon.native_runtime_link}
// CHECK-DAG: llvm.func @hexagon_runtime_dma_wait(i32) attributes {hexagon.native_runtime_link}
// CHECK-DAG: llvm.func @hexkl_matmul_f16f16_f32(i64, i64, i64, !llvm.ptr, !llvm.ptr, !llvm.ptr) attributes {hexagon.native_runtime_link}

// CHECK-LABEL: llvm.func @memory_and_dma() attributes {sym_visibility = "private"}
// CHECK: llvm.call @hexagon_runtime_alloc_1d
// CHECK: llvm.call @hexagon_runtime_dma_start
// CHECK: llvm.call @hexagon_runtime_dma_wait
// CHECK: llvm.call @hexagon_runtime_free_1d

// CHECK-LABEL: llvm.func @hexkl_matmul(
// CHECK: llvm.call @hexkl_matmul_f16f16_f32

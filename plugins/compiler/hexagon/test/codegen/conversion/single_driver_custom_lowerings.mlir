// Verify that the final conversion driver composes the HexagonMem, DMA, and
// HexKL patterns together with the standard LLVM conversion. In particular,
// VTCM memrefs must be converted directly to default-address-space pointers; no
// collapse pass or materialization-cast cleanup is expected after this pass.
// Every VTCM allocation, including HexKL's matmul scratch, is taken from the
// dispatch's VTCM partition through the runtime state, and frees are dropped.
// The HexKL scratch is allocated at the dispatch entry.

// RUN: iree-opt --pass-pipeline='builtin.module(iree-hexagon-convert-to-llvm)' \
// RUN:   %s | FileCheck %s \
// RUN:     --implicit-check-not=builtin.unrealized_conversion_cast \
// RUN:     --implicit-check-not='!llvm.ptr<1>' \
// RUN:     --implicit-check-not=hexagonmem. \
// RUN:     --implicit-check-not=hexkl. \
// RUN:     --implicit-check-not=memref.dma_ \
// RUN:     --implicit-check-not=malloc \
// RUN:     --implicit-check-not=free \
// RUN:     --implicit-check-not=hexagon_runtime_alloc_1d \
// RUN:     --implicit-check-not='@hexkl_matmul_f16f16_f32('

module {
  func.func @memory_and_dma() {
    %c0 = arith.constant 0 : index
    %c4 = arith.constant 4 : index
    %src = memref.alloc() : memref<4xf32>
    %dst = hexagonmem.alloc() <alignment = 128> : memref<4xf32, 1>
    %tag = memref.alloca() : memref<1xi32>
    memref.dma_start %src[%c0], %dst[%c0], %c4, %tag[%c0]
        : memref<4xf32>, memref<4xf32, 1>, memref<1xi32>
    memref.dma_wait %tag[%c0], %c4 : memref<1xi32>
    hexagonmem.dealloc %dst : memref<4xf32, 1>
    memref.dealloc %src : memref<4xf32>
    return
  }

  func.func @hexkl_matmul() {
    %lhs = memref.alloca() : memref<2x4xf16>
    %rhs = memref.alloca() : memref<4x3xf16>
    %out = memref.alloca() : memref<2x3xf32>
    hexkl.matmul ins(%lhs, %rhs : memref<2x4xf16>, memref<4x3xf16>)
                 outs(%out : memref<2x3xf32>)
    return
  }
}

// CHECK-DAG: llvm.func @hexagon_runtime_vtcm_alloc(!llvm.ptr, i64, i64) -> !llvm.ptr attributes {hal.import.static}
// CHECK-DAG: llvm.func @hexagon_runtime_dma_start(!llvm.ptr, i32, !llvm.ptr, i32, i32, i32, i32, !llvm.ptr) -> i32 attributes {hal.import.static}
// CHECK-DAG: llvm.func @hexagon_runtime_dma_wait(i32) attributes {hal.import.static}
// CHECK-DAG: llvm.func @hexkl_matmul_f16f16_f32_scratch_bytes(i64, i64, i64) -> i32 attributes {hal.import.static}
// CHECK-DAG: llvm.func @hexkl_matmul_f16f16_f32_with_scratch(i64, i64, i64, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> i32 attributes {hal.import.static}

// CHECK-LABEL: llvm.func @memory_and_dma(
// CHECK-SAME:    %{{[A-Za-z0-9_]+}}: !llvm.ptr {{{[^}]*}}}, %[[DISPATCH:[A-Za-z0-9_]+]]: !llvm.ptr
// CHECK:         %[[STATE_ADDR:.+]] = llvm.getelementptr inbounds %[[DISPATCH]][0, 1]
// CHECK:         %[[STATE:.+]] = llvm.load %[[STATE_ADDR]] : !llvm.ptr -> !llvm.ptr
// CHECK:         %[[ALIGN_128:.+]] = llvm.mlir.constant(128 : i64) : i64
// CHECK:         llvm.call @hexagon_runtime_vtcm_alloc(%[[STATE]], %{{.+}}, %[[ALIGN_128]]) : (!llvm.ptr, i64, i64) -> !llvm.ptr
// CHECK:         llvm.call @hexagon_runtime_vtcm_alloc(%[[STATE]], %{{.+}}, %{{.+}}) : (!llvm.ptr, i64, i64) -> !llvm.ptr
// CHECK:         llvm.call @hexagon_runtime_dma_start
// CHECK:         llvm.call @hexagon_runtime_dma_wait
// CHECK-NOT:     llvm.call
// CHECK:         llvm.return

// CHECK-LABEL: llvm.func @hexkl_matmul(
// CHECK:         %[[STATE:.+]] = llvm.load %{{.+}} : !llvm.ptr -> !llvm.ptr
// CHECK-DAG:     %[[ROWS:.+]] = llvm.mlir.constant(2 : i64) : i64
// CHECK-DAG:     %[[COLS:.+]] = llvm.mlir.constant(3 : i64) : i64
// CHECK-DAG:     %[[INNER:.+]] = llvm.mlir.constant(4 : i64) : i64
// CHECK:         %[[BYTES:.+]] = llvm.call @hexkl_matmul_f16f16_f32_scratch_bytes(%[[ROWS]], %[[COLS]], %[[INNER]]) : (i64, i64, i64) -> i32
// CHECK-DAG:     %[[BYTES_IDX:.+]] = llvm.zext %[[BYTES]] : i32 to i64
// CHECK-DAG:     %[[ALIGN_2K:.+]] = llvm.mlir.constant(2048 : i64) : i64
// CHECK:         %[[SCRATCH:.+]] = llvm.call @hexagon_runtime_vtcm_alloc(%[[STATE]], %[[BYTES_IDX]], %[[ALIGN_2K]])
// CHECK:         llvm.alloca
// CHECK:         llvm.call @hexkl_matmul_f16f16_f32_with_scratch(%{{.+}}, %{{.+}}, %{{.+}}, %{{.+}}, %{{.+}}, %{{.+}}, %[[SCRATCH]])

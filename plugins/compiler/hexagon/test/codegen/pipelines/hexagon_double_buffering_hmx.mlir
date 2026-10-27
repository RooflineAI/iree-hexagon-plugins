// Double buffering in the HmxMatmulExpert pipeline. With the VTCM budget
// halved for the second buffers, the matmul is tiled in M and N: the B tile
// and the output tile of the inner (N) loop are double-buffered. The A tile of
// the outer (M) loop is double-buffered too, unless outer tile loops are
// disabled; then it stays a synchronous copy. The HMX ops take the buffer
// slots as strided views and still pass the HMX runtime ABI checks on the way
// to LLVM.

// RUN: iree-opt \
// RUN:   --iree-hexagon-enable-vtcm-tiling --iree-hexagon-enable-hmx-matmul \
// RUN:   --iree-hexagon-enable-double-buffering \
// RUN:   --iree-hexagon-double-buffer-outer-tile-loops=false \
// RUN:   --pass-pipeline='builtin.module(hal.executable(hal.executable.variant(builtin.module(iree-hexagon-select-lowering-strategy, func.func(iree-hexagon-lower-executable-target)))))' \
// RUN:   --mlir-print-ir-after=iree-hexagon-pipeline-vtcm-transfers \
// RUN:   %s -o /dev/null 2>&1 | FileCheck %s

// RUN: iree-opt \
// RUN:   --iree-hexagon-enable-vtcm-tiling --iree-hexagon-enable-hmx-matmul \
// RUN:   --iree-hexagon-enable-double-buffering \
// RUN:   --iree-hexagon-double-buffer-outer-tile-loops \
// RUN:   --pass-pipeline='builtin.module(hal.executable(hal.executable.variant(iree-hexagon-configuration-pipeline, iree-hexagon-translation-pipeline)))' \
// RUN:   %s | FileCheck %s --check-prefix=LLVM

// CHECK-LABEL: IR Dump After HexagonPipelineVTCMTransfersPass
//   CHECK-DAG:   memref.alloc() : memref<2x{{[0-9]+}}x{{[0-9]+}}xf32, 1>
//   CHECK-DAG:   memref.alloc() : memref<2x{{[0-9]+}}x{{[0-9]+}}xf16, 1>
//       CHECK:   scf.for
//       CHECK:     memref.copy
//       CHECK:     iree_hexagon.hmx.pack
//       CHECK:     memref.dma_start
//       CHECK:     scf.for
//       CHECK:       scf.if
//       CHECK:         memref.dma_start
//       CHECK:       }
//       CHECK:       memref.dma_wait
//       CHECK:       iree_hexagon.hmx.pack ins(%{{.+}} : memref<{{.+}}, strided<{{.+}}, offset: ?>, 1>)
//       CHECK:       iree_hexagon.hmx.matmul
//       CHECK:       iree_hexagon.hmx.unpack {{.+}} outs(%{{.+}} : memref<{{.+}}, strided<{{.+}}, offset: ?>, 1>)
//       CHECK:       memref.dma_start
//       CHECK:       scf.if
//       CHECK:         memref.dma_wait
//       CHECK:       }
//       CHECK:     } {llvm.loop_annotation = #{{.+}}}
//       CHECK:     memref.dma_wait
//       CHECK:   }
//   CHECK-NOT:   llvm.loop_annotation

// Both tile loops are rotated and keep their annotation down to the latch.
// LLVM-LABEL: llvm.func @mm(
//       LLVM:   llvm.br {{.+}} loop_annotation = #[[ANNOTATION:.+]] {
//       LLVM:   llvm.br {{.+}} loop_annotation = #[[ANNOTATION]] {

#executable_target_embedded_elf_hexagon = #hal.executable.target<"hexagon", "embedded-elf-hexagon", {cpu = "hexagonv79", cpu_features = "+hvxv79,+hvx-length128b", data_layout = "e-m:e-p:32:32:32-a:0-n16:32-i64:64:64-i32:32:32-i16:16:16-i1:8:8-f32:32:32-f64:64:64-v32:32:32-v64:64:64-v512:512:512-v1024:1024:1024-v2048:2048:2048", hexagon.version = "79", max_stack_allocation_size = 16384 : i64, native_vector_size = 128 : i64, target_triple = "hexagon-unknown-unknown-elf"}>
#pipeline_layout = #hal.pipeline.layout<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>
hal.executable private @test {
  hal.executable.variant public @embedded_elf_hexagon target(#executable_target_embedded_elf_hexagon) {
    hal.executable.export public @mm ordinal(0) layout(#pipeline_layout) count(%arg0: !hal.device) -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice()
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @mm() {
        %cst = arith.constant 0.0 : f32
        %c0 = arith.constant 0 : index
        %a = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags("ReadOnly|Indirect") : !iree_tensor_ext.dispatch.tensor<readonly:tensor<512x1024xf16>>
        %b = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) flags("ReadOnly|Indirect") : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1024x1024xf16>>
        %o = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0) flags(Indirect) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<512x1024xf32>>
        %at = iree_tensor_ext.dispatch.tensor.load %a, offsets = [0, 0], sizes = [512, 1024], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<512x1024xf16>> -> tensor<512x1024xf16>
        %bt = iree_tensor_ext.dispatch.tensor.load %b, offsets = [0, 0], sizes = [1024, 1024], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1024x1024xf16>> -> tensor<1024x1024xf16>
        %e = tensor.empty() : tensor<512x1024xf32>
        %f = linalg.fill ins(%cst : f32) outs(%e : tensor<512x1024xf32>) -> tensor<512x1024xf32>
        %r = linalg.matmul ins(%at, %bt : tensor<512x1024xf16>, tensor<1024x1024xf16>) outs(%f : tensor<512x1024xf32>) -> tensor<512x1024xf32>
        iree_tensor_ext.dispatch.tensor.store %r, %o, offsets = [0, 0], sizes = [512, 1024], strides = [1, 1] : tensor<512x1024xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<512x1024xf32>>
        return
      }
    }
  }
}

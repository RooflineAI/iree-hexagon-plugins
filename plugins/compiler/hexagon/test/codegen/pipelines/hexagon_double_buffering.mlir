// Double buffering in the MultiTilingExpert pipeline, with the default
// threading inside the VTCM tile. The VTCM tile loop is marked on tensors,
// its transfers become DMAs into two buffer slots, the loop is rotated, and
// the thread loop of the tile (the compute) exists exactly once.

// RUN: iree-opt \
// RUN:   --iree-hexagon-enable-vtcm-tiling --iree-hexagon-enable-double-buffering \
// RUN:   --pass-pipeline='builtin.module(hal.executable(hal.executable.variant(builtin.module(func.func(iree-hexagon-lower-executable-target)))))' \
// RUN:   --mlir-print-ir-after=iree-hexagon-coalesce-vtcm-loops \
// RUN:   --mlir-print-ir-after=iree-hexagon-pipeline-vtcm-transfers \
// RUN:   --mlir-print-ir-after=iree-hexagon-form-async-threads \
// RUN:   %s -o /dev/null 2>&1 | FileCheck %s

// CHECK-LABEL: IR Dump After HexagonCoalesceVTCMLoopsPass
//       CHECK:   scf.for
//       CHECK:     iree_hexagon.stage_to_vtcm
//       CHECK:   } {iree_hexagon.vtcm_tile_loop}

// CHECK-LABEL: IR Dump After HexagonPipelineVTCMTransfersPass
//   CHECK-DAG:   memref.alloca() : memref<2xi32>
//   CHECK-DAG:   memref.alloc() : memref<2x128x512xf32, 1>
//       CHECK:   memref.dma_start
//       CHECK:   memref.dma_start
//       CHECK:   scf.for
//       CHECK:     scf.if
//       CHECK:       memref.dma_start
//       CHECK:       memref.dma_start
//       CHECK:     }
//       CHECK:     memref.dma_wait
//       CHECK:     memref.dma_wait
//       CHECK:     scf.forall
//       CHECK:     } {mapping = [#iree_codegen.local_mapping<0>]}
//       CHECK:     memref.dma_start
//       CHECK:     scf.if
//       CHECK:       memref.dma_wait
//       CHECK:     }
//       CHECK:   } {llvm.loop_annotation = #{{.+}}}
//       CHECK:   memref.dma_wait
//   CHECK-NOT:   memref.copy
//   CHECK-NOT:   scf.forall

// CHECK-LABEL: IR Dump After HexagonFormAsyncThreadsPass
//       CHECK:   async.runtime.create_group
//   CHECK-NOT:   async.runtime.create_group
//       CHECK:   } {llvm.loop_annotation = #{{.+}}}

#executable_target_embedded_elf_hexagon = #hal.executable.target<"hexagon", "embedded-elf-hexagon", {cpu = "hexagonv79", cpu_features = "+hvxv79,+hvx-length128b", data_layout = "e-m:e-p:32:32:32-a:0-n16:32-i64:64:64-i32:32:32-i16:16:16-i1:8:8-f32:32:32-f64:64:64-v32:32:32-v64:64:64-v512:512:512-v1024:1024:1024-v2048:2048:2048", hexagon.version = "79", max_stack_allocation_size = 16384 : i64, native_vector_size = 128 : i64, target_triple = "hexagon-unknown-unknown-elf"}>
#pipeline_layout = #hal.pipeline.layout<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>
#translation = #iree_codegen.translation_info<pipeline = #iree_hexagon.pipeline<MultiTilingExpert>>
#config = #iree_hexagon.lowering_config<distribution = [0, 0], vector_common_parallel = [1, 32], vtcm = [128, 512]>
hal.executable private @test_vtcm {
  hal.executable.variant public @embedded_elf_hexagon target(#executable_target_embedded_elf_hexagon) {
    builtin.module {
      func.func @add() attributes {translation_info = #translation} {
        %c0 = arith.constant 0 : index
        %a = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags("ReadOnly|Indirect") : !iree_tensor_ext.dispatch.tensor<readonly:tensor<2048x512xf32>>
        %b = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) flags("ReadOnly|Indirect") : !iree_tensor_ext.dispatch.tensor<readonly:tensor<2048x512xf32>>
        %out = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0) flags(Indirect) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<2048x512xf32>>
        %at = iree_tensor_ext.dispatch.tensor.load %a, offsets = [0, 0], sizes = [2048, 512], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<2048x512xf32>> -> tensor<2048x512xf32>
        %bt = iree_tensor_ext.dispatch.tensor.load %b, offsets = [0, 0], sizes = [2048, 512], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<2048x512xf32>> -> tensor<2048x512xf32>
        %init = tensor.empty() : tensor<2048x512xf32>
        %result = linalg.generic {indexing_maps = [affine_map<(i, j) -> (i, j)>, affine_map<(i, j) -> (i, j)>, affine_map<(i, j) -> (i, j)>], iterator_types = ["parallel", "parallel"]} ins(%at, %bt : tensor<2048x512xf32>, tensor<2048x512xf32>) outs(%init : tensor<2048x512xf32>) attrs = {lowering_config = #config} {
        ^bb0(%x: f32, %y: f32, %o: f32):
          %s = arith.addf %x, %y : f32
          linalg.yield %s : f32
        } -> tensor<2048x512xf32>
        iree_tensor_ext.dispatch.tensor.store %result, %out, offsets = [0, 0], sizes = [2048, 512], strides = [1, 1] : tensor<2048x512xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<2048x512xf32>>
        return
      }
    }
  }
}

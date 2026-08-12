// This test is making sure that the translation pipeline with the VTCM flag enabled
// (assuming the configuration pipeline has already run) executes correctly and
// generates memrefs in the VTCM address space

// RUN: iree-opt --iree-load-plugin=cellar_hexagon=$CELLAR_HEXAGON_COMPILER_PLUGIN \
// RUN:   --iree-hexagon-enable-vtcm-tiling \
// RUN:   --pass-pipeline='builtin.module(hal.executable(hal.executable.variant(builtin.module(func.func(iree-hexagon-lower-executable-target)))))' \
// RUN:   --mlir-print-ir-after=iree-hexagon-vtcm-tiling \
// RUN:   --mlir-print-ir-after=iree-hexagon-lower-vtcm-staging \
// RUN:   --mlir-print-ir-after=iree-codegen-iree-comprehensive-bufferize \
// RUN:   --mlir-print-ir-after=iree-codegen-canonicalize \
// RUN:   %s -o /dev/null 2>&1 |  FileCheck %s

#executable_target_embedded_elf_hexagon = #hal.executable.target<"hexagon", "embedded-elf-hexagon", {cpu = "hexagonv79", cpu_features = "+hvxv79,+hvx-length128b", data_layout = "e-m:e-p:32:32:32-a:0-n16:32-i64:64:64-i32:32:32-i16:16:16-i1:8:8-f32:32:32-f64:64:64-v32:32:32-v64:64:64-v512:512:512-v1024:1024:1024-v2048:2048:2048", hexagon.version = "79", iree.encoding.resolver = #iree_hexagon.hexagon_encoding_resolver<>, link_embedded = false, max_stack_allocation_size = 16384 : i64, native_vector_size = 32 : i64, target_triple = "hexagon-unknown-unknown-elf"}>
#pipeline_layout = #hal.pipeline.layout<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>
#translation = #iree_codegen.translation_info<pipeline = #iree_cpu.pipeline<DoubleTilingExpert>>
#config = #iree_cpu.lowering_config<distribution = [4, 4], vector_common_parallel = [2, 2]>
#vtcm = #iree_hexagon.vtcm_tiling_config<tile_sizes = [4, 4]>

// Check that we generate our new VTCM staging marker
// CHECK-LABEL: IR Dump After HexagonVTCMTilingPass
// CHECK: func.func @vtcm_then_cpu_dispatch
// CHECK: iree_hexagon.stage_to_vtcm

// Check that we generate bufferization.alloc_tensor from our new op
// CHECK-LABEL: IR Dump After HexagonLowerVTCMStagingPass
// CHECK: func.func @vtcm_then_cpu_dispatch
// CHECK: bufferization.alloc_tensor() copy
// CHECK-SAME: memory_space = 1 : i64

// Check that we copy directly to DDR from VTCM (without intermediate buffers) the output from one of the loops
// CHECK-LABEL: IR Dump After IREEComprehensiveBufferizePass
// In order to check the output copy, we need to go into the first loop (copying happens there)
// CHECK: memref.copy {{.*}}, {{.*}} : memref<{{.*}}, 1> to memref<{{.*}}, #hal.descriptor_type<storage_buffer>>
// There should now be a redundant copy (DDR -> DDR) that will get optimized away:
// CHECK: memref.copy {{.*}}, {{.*}} : memref<{{.*}}, #hal.descriptor_type<storage_buffer>> to memref<{{.*}}, #hal.descriptor_type<storage_buffer>>

// Finally, check that the redundant copy is gone after all optimizations
// CHECK-LABEL: IR Dump After IREECodegenCanonicalizerPass
// CHECK-NOT: memref.copy {{.*}}, {{.*}} : memref<{{.*}}, #hal.descriptor_type<storage_buffer>> to memref<{{.*}}, #hal.descriptor_type<storage_buffer>>

hal.executable private @test_vtcm {
  hal.executable.variant public @embedded_elf_hexagon target(#executable_target_embedded_elf_hexagon) {
    builtin.module {
      func.func @vtcm_then_cpu_dispatch() attributes {translation_info = #translation} {
        %c0 = arith.constant 0 : index
        %cst = arith.constant 1.0 : f32
        %lhs = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags("ReadOnly|Indirect") : !iree_tensor_ext.dispatch.tensor<readonly:tensor<8x8xf32>>
        %out = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) flags(Indirect) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<8x8xf32>>
        %lhs_t = iree_tensor_ext.dispatch.tensor.load %lhs, offsets = [0, 0], sizes = [8, 8], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<8x8xf32>> -> tensor<8x8xf32>
        %init = tensor.empty() : tensor<8x8xf32>
        %filled = linalg.fill ins(%cst : f32) outs(%init : tensor<8x8xf32>) -> tensor<8x8xf32>
        %result = linalg.generic {indexing_maps = [affine_map<(i, j) -> (i, j)>, affine_map<(i, j) -> (i, j)>], iterator_types = ["parallel", "parallel"]} ins(%lhs_t : tensor<8x8xf32>) outs(%filled : tensor<8x8xf32>) attrs = {hexagon_vtcm_tiling_config = #vtcm, lowering_config = #config} {
        ^bb0(%in: f32, %out0: f32):
          %sum = arith.addf %in, %out0 : f32
          linalg.yield %sum : f32
        } -> tensor<8x8xf32>
        iree_tensor_ext.dispatch.tensor.store %result, %out, offsets = [0, 0], sizes = [8, 8], strides = [1, 1] : tensor<8x8xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<8x8xf32>>
        return
      }
    }
  }
}

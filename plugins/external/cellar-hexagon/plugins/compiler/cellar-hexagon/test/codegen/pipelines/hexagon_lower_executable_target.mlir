// Verifies that the custom iree-hexagon-lower-executable-target maps each
// LLVMCPU pipeline selection to its Hexagon equivalent. Each chunk below uses a
// different translation_info pipeline and checks that a pipeline-specific pass
// runs on the expected dispatch function.

// RUN: iree-opt --iree-load-plugin=cellar_hexagon=$CELLAR_HEXAGON_COMPILER_PLUGIN \
// RUN:   --pass-pipeline='builtin.module(hal.executable(hal.executable.variant(builtin.module(func.func(iree-hexagon-lower-executable-target)))))' \
// RUN:   --mlir-print-ir-after-all \
// RUN:   --split-input-file %s -o /dev/null 2>&1 | FileCheck %s

// CHECK-LABEL: IR Dump After LLVMCPUTileAndFuseProducerConsumerPass
// CHECK: func.func @default_dispatch

#translation_default = #iree_codegen.translation_info<pipeline = CPUDefault>
#executable_target_embedded_elf_hexagon = #hal.executable.target<"hexagon", "embedded-elf-hexagon", {cpu = "hexagonv79", cpu_features = "+hvxv79,+hvx-length128b", data_layout = "e-m:e-p:32:32:32-a:0-n16:32-i64:64:64-i32:32:32-i16:16:16-i1:8:8-f32:32:32-f64:64:64-v32:32:32-v64:64:64-v512:512:512-v1024:1024:1024-v2048:2048:2048", hexagon.version = "79", iree.encoding.resolver = #iree_hexagon.hexagon_encoding_resolver<>, link_embedded = false, max_stack_allocation_size = 16384 : i64, native_vector_size = 32 : i64, target_triple = "hexagon-unknown-unknown-elf"}>
#pipeline_layout = #hal.pipeline.layout<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>

hal.executable private @test_default {
  hal.executable.variant public @embedded_elf_hexagon target(#executable_target_embedded_elf_hexagon) {
    builtin.module {
      func.func @default_dispatch() attributes {translation_info = #translation_default} {
        %cst = arith.constant 0.0 : f32
        %c0 = arith.constant 0 : index
        %c64 = arith.constant 64 : index
        %lhs = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags("ReadOnly|Indirect") : !iree_tensor_ext.dispatch.tensor<readonly:tensor<4x4xf32>>
        %out = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) flags(Indirect) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<4x4xf32>>

        %lhs_t = iree_tensor_ext.dispatch.tensor.load %lhs, offsets = [0, 0], sizes = [4, 4], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<4x4xf32>> -> tensor<4x4xf32>
        %empty = tensor.empty() : tensor<4x4xf32>
        %filled = linalg.fill ins(%cst : f32) outs(%empty : tensor<4x4xf32>) -> tensor<4x4xf32>
        %add = linalg.generic {indexing_maps = [affine_map<(i, j) -> (i, j)>, affine_map<(i, j) -> (i, j)>], iterator_types = ["parallel", "parallel"]} ins(%lhs_t : tensor<4x4xf32>) outs(%filled : tensor<4x4xf32>) {
        ^bb0(%in: f32, %out0: f32):
          %sum = arith.addf %in, %out0 : f32
          linalg.yield %sum : f32
        } -> tensor<4x4xf32>

        iree_tensor_ext.dispatch.tensor.store %add, %out, offsets = [0, 0], sizes = [4, 4], strides = [1, 1] : tensor<4x4xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<4x4xf32>>
        return
      }
    }
  }
}

// -----

// CHECK-LABEL: IR Dump After LLVMCPUVirtualVectorLoweringPass
// CHECK: func.func @buffer_dispatch

#executable_target_embedded_elf_hexagon = #hal.executable.target<"hexagon", "embedded-elf-hexagon", {cpu = "hexagonv79", cpu_features = "+hvxv79,+hvx-length128b", data_layout = "e-m:e-p:32:32:32-a:0-n16:32-i64:64:64-i32:32:32-i16:16:16-i1:8:8-f32:32:32-f64:64:64-v32:32:32-v64:64:64-v512:512:512-v1024:1024:1024-v2048:2048:2048", hexagon.version = "79", iree.encoding.resolver = #iree_hexagon.hexagon_encoding_resolver<>, link_embedded = false, max_stack_allocation_size = 16384 : i64, native_vector_size = 32 : i64, target_triple = "hexagon-unknown-unknown-elf"}>
#pipeline_layout = #hal.pipeline.layout<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>
#translation_buffer = #iree_codegen.translation_info<pipeline = CPUBufferOpsTileAndVectorize>

hal.executable private @test_buffer {
  hal.executable.variant public @embedded_elf_hexagon target(#executable_target_embedded_elf_hexagon) {
    builtin.module {
      func.func @buffer_dispatch() attributes {translation_info = #translation_buffer} {
        %c0 = arith.constant 0 : index
        %c64 = arith.constant 64 : index
        %lhs = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags("ReadOnly|Indirect") : !iree_tensor_ext.dispatch.tensor<readonly:tensor<4x4xf32>>
        %out = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) flags(Indirect) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<4x4xf32>>

        %lhs_t = iree_tensor_ext.dispatch.tensor.load %lhs, offsets = [0, 0], sizes = [4, 4], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<4x4xf32>> -> tensor<4x4xf32>
        %init = tensor.empty() : tensor<4x4xf32>
        %result = linalg.generic {indexing_maps = [affine_map<(i, j) -> (i, j)>, affine_map<(i, j) -> (i, j)>], iterator_types = ["parallel", "parallel"]} ins(%lhs_t : tensor<4x4xf32>) outs(%init : tensor<4x4xf32>) {
        ^bb0(%in: f32, %out0: f32):
          linalg.yield %in : f32
        } -> tensor<4x4xf32>

        iree_tensor_ext.dispatch.tensor.store %result, %out, offsets = [0, 0], sizes = [4, 4], strides = [1, 1] : tensor<4x4xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<4x4xf32>>
        return
      }
    }
  }
}

// -----

// CHECK-LABEL: IR Dump After LLVMCPUSplitReductionPass
// CHECK: func.func @double_tiling_dispatch

#executable_target_embedded_elf_hexagon = #hal.executable.target<"hexagon", "embedded-elf-hexagon", {cpu = "hexagonv79", cpu_features = "+hvxv79,+hvx-length128b", data_layout = "e-m:e-p:32:32:32-a:0-n16:32-i64:64:64-i32:32:32-i16:16:16-i1:8:8-f32:32:32-f64:64:64-v32:32:32-v64:64:64-v512:512:512-v1024:1024:1024-v2048:2048:2048", hexagon.version = "79", iree.encoding.resolver = #iree_hexagon.hexagon_encoding_resolver<>, link_embedded = false, max_stack_allocation_size = 16384 : i64, native_vector_size = 32 : i64, target_triple = "hexagon-unknown-unknown-elf"}>
#pipeline_layout = #hal.pipeline.layout<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>
#translation_double = #iree_codegen.translation_info<pipeline = CPUDoubleTilingExpert>
#config_double = #iree_cpu.lowering_config<distribution = [1, 1, 0], vector_common_parallel = [1, 1, 0], vector_reduction = [0, 0, 1]>

hal.executable private @test_double_tiling {
  hal.executable.variant public @embedded_elf_hexagon target(#executable_target_embedded_elf_hexagon) {
    builtin.module {
      func.func @double_tiling_dispatch() attributes {translation_info = #translation_double} {
        %c0 = arith.constant 0 : index
        %lhs = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags("ReadOnly|Indirect") : !iree_tensor_ext.dispatch.tensor<readonly:tensor<2x2xf32>>
        %rhs = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags("ReadOnly|Indirect") : !iree_tensor_ext.dispatch.tensor<readonly:tensor<2x2xf32>>
        %out = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) flags(Indirect) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<2x2xf32>>

        %lhs_t = iree_tensor_ext.dispatch.tensor.load %lhs, offsets = [0, 0], sizes = [2, 2], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<2x2xf32>> -> tensor<2x2xf32>
        %rhs_t = iree_tensor_ext.dispatch.tensor.load %rhs, offsets = [0, 0], sizes = [2, 2], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<2x2xf32>> -> tensor<2x2xf32>

        %init = tensor.empty() : tensor<2x2xf32>
        %result = linalg.matmul {lowering_config = #config_double} ins(%lhs_t, %rhs_t : tensor<2x2xf32>, tensor<2x2xf32>) outs(%init : tensor<2x2xf32>) -> tensor<2x2xf32>

        iree_tensor_ext.dispatch.tensor.store %result, %out, offsets = [0, 0], sizes = [2, 2], strides = [1, 1] : tensor<2x2xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<2x2xf32>>
        return
      }
    }
  }
}

// -----

// CHECK-LABEL: IR Dump After DecomposeConvolutionToLowerDimOpsPass
// CHECK: func.func @conv_dispatch

#executable_target_embedded_elf_hexagon = #hal.executable.target<"hexagon", "embedded-elf-hexagon", {cpu = "hexagonv79", cpu_features = "+hvxv79,+hvx-length128b", data_layout = "e-m:e-p:32:32:32-a:0-n16:32-i64:64:64-i32:32:32-i16:16:16-i1:8:8-f32:32:32-f64:64:64-v32:32:32-v64:64:64-v512:512:512-v1024:1024:1024-v2048:2048:2048", hexagon.version = "79", iree.encoding.resolver = #iree_hexagon.hexagon_encoding_resolver<>, link_embedded = false, max_stack_allocation_size = 16384 : i64, native_vector_size = 32 : i64, target_triple = "hexagon-unknown-unknown-elf"}>
#pipeline_layout = #hal.pipeline.layout<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>
#translation_conv = #iree_codegen.translation_info<pipeline = CPUConvTileAndDecomposeExpert>

hal.executable private @test_conv {
  hal.executable.variant public @embedded_elf_hexagon target(#executable_target_embedded_elf_hexagon) {
    builtin.module {
      func.func @conv_dispatch() attributes {translation_info = #translation_conv} {
        %c0 = arith.constant 0 : index
        %lhs = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags("ReadOnly|Indirect") : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x1x3x3xf32>>
        %rhs = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags("ReadOnly|Indirect") : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x1x3x3xf32>>
        %out = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) flags(Indirect) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1x1x1x3xf32>>

        %lhs_t = iree_tensor_ext.dispatch.tensor.load %lhs, offsets = [0, 0, 0, 0], sizes = [1, 1, 3, 3], strides = [1, 1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x1x3x3xf32>> -> tensor<1x1x3x3xf32>
        %rhs_t = iree_tensor_ext.dispatch.tensor.load %rhs, offsets = [0, 0, 0, 0], sizes = [1, 1, 3, 3], strides = [1, 1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x1x3x3xf32>> -> tensor<1x1x3x3xf32>

        %init = tensor.empty() : tensor<1x1x1x3xf32>
        %result = linalg.conv_2d_nhwc_hwcf
            ins(%lhs_t, %rhs_t : tensor<1x1x3x3xf32>, tensor<1x1x3x3xf32>)
           outs(%init : tensor<1x1x1x3xf32>) -> tensor<1x1x1x3xf32>

        iree_tensor_ext.dispatch.tensor.store %result, %out, offsets = [0, 0, 0, 0], sizes = [1, 1, 1, 3], strides = [1, 1, 1, 1] : tensor<1x1x1x3xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1x1x1x3xf32>>
        return
      }
    }
  }
}

// -----

// CHECK-LABEL: IR Dump After LLVMCPUVectorTransposeLoweringPass
// CHECK: func.func @data_tiling_dispatch

#executable_target_embedded_elf_hexagon = #hal.executable.target<"hexagon", "embedded-elf-hexagon", {cpu = "hexagonv79", cpu_features = "+hvxv79,+hvx-length128b", data_layout = "e-m:e-p:32:32:32-a:0-n16:32-i64:64:64-i32:32:32-i16:16:16-i1:8:8-f32:32:32-f64:64:64-v32:32:32-v64:64:64-v512:512:512-v1024:1024:1024-v2048:2048:2048", hexagon.version = "79", iree.encoding.resolver = #iree_hexagon.hexagon_encoding_resolver<>, link_embedded = false, max_stack_allocation_size = 16384 : i64, native_vector_size = 32 : i64, target_triple = "hexagon-unknown-unknown-elf"}>
#pipeline_layout = #hal.pipeline.layout<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>
#translation_data = #iree_codegen.translation_info<pipeline = CPUDataTiling>

hal.executable private @test_data_tiling {
  hal.executable.variant public @embedded_elf_hexagon target(#executable_target_embedded_elf_hexagon) {
    builtin.module {
      func.func @data_tiling_dispatch() attributes {translation_info = #translation_data} {
        %c0 = arith.constant 0 : index
        %lhs = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags("ReadOnly|Indirect") : !iree_tensor_ext.dispatch.tensor<readonly:tensor<2x2xf32>>
        %out = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) flags(Indirect) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<2x2xf32>>

        %lhs_t = iree_tensor_ext.dispatch.tensor.load %lhs, offsets = [0, 0], sizes = [2, 2], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<2x2xf32>> -> tensor<2x2xf32>
        %result = tensor.empty() : tensor<2x2xf32>
        %copy = tensor.insert_slice %lhs_t into %result[0, 0] [2, 2] [1, 1] : tensor<2x2xf32> into tensor<2x2xf32>

        iree_tensor_ext.dispatch.tensor.store %copy, %out, offsets = [0, 0], sizes = [2, 2], strides = [1, 1] : tensor<2x2xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<2x2xf32>>
        return
      }
    }
  }
}

// -----

// CHECK-LABEL: IR Dump After LLVMCPUMmt4dVectorLoweringPass
// CHECK: func.func @mmt4d_dispatch

#executable_target_embedded_elf_hexagon = #hal.executable.target<"hexagon", "embedded-elf-hexagon", {cpu = "hexagonv79", cpu_features = "+hvxv79,+hvx-length128b", data_layout = "e-m:e-p:32:32:32-a:0-n16:32-i64:64:64-i32:32:32-i16:16:16-i1:8:8-f32:32:32-f64:64:64-v32:32:32-v64:64:64-v512:512:512-v1024:1024:1024-v2048:2048:2048", hexagon.version = "79", iree.encoding.resolver = #iree_hexagon.hexagon_encoding_resolver<>, link_embedded = false, max_stack_allocation_size = 16384 : i64, native_vector_size = 32 : i64, target_triple = "hexagon-unknown-unknown-elf"}>
#pipeline_layout = #hal.pipeline.layout<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>
#config2 = #iree_cpu.lowering_config<vector_common_parallel = [1, 1, 0, 16], vector_inner_parallel = [0, 0, 1, 0]>
#config3 = #iree_cpu.lowering_config<distribution = [1, 1, 0, 0, 0, 0], vector_common_parallel = [1, 1, 0, 16, 16, 0], vector_reduction = [0, 0, 1, 0, 0, 1]>
#translation = #iree_codegen.translation_info<pipeline = Mmt4dTilingExpert>

hal.executable private @test {
  hal.executable.variant public @embedded_elf_hexagon target(#executable_target_embedded_elf_hexagon) {
    builtin.module {
      func.func @mmt4d_dispatch() attributes {translation_info = #translation} {
        %cst = arith.constant 0.000000e+00 : f32
        %c0 = arith.constant 0 : index
        %c256 = arith.constant 256 : index
        %lhs = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags("ReadOnly|Indirect") : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x4x16x1xf32>>
        %rhs = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c256) flags("ReadOnly|Indirect") : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x4x16x1xf32>>
        %out = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) flags(Indirect) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1x1x16x16xf32>>

        %lhs_t = iree_tensor_ext.dispatch.tensor.load %lhs, offsets = [0, 0, 0, 0], sizes = [1, 4, 16, 1], strides = [1, 1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x4x16x1xf32>> -> tensor<1x4x16x1xf32>
        %rhs_t = iree_tensor_ext.dispatch.tensor.load %rhs, offsets = [0, 0, 0, 0], sizes = [1, 4, 16, 1], strides = [1, 1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x4x16x1xf32>> -> tensor<1x4x16x1xf32>

        %init = tensor.empty() : tensor<1x1x16x16xf32>
        %filled = linalg.fill {lowering_config = #config2} ins(%cst : f32) outs(%init : tensor<1x1x16x16xf32>) -> tensor<1x1x16x16xf32>
        %result = linalg.mmt4d {lowering_config = #config3} ins(%lhs_t, %rhs_t : tensor<1x4x16x1xf32>, tensor<1x4x16x1xf32>) outs(%filled : tensor<1x1x16x16xf32>) -> tensor<1x1x16x16xf32>

        iree_tensor_ext.dispatch.tensor.store %result, %out, offsets = [0, 0, 0, 0], sizes = [1, 1, 16, 16], strides = [1, 1, 1, 1] : tensor<1x1x16x16xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1x1x16x16xf32>>
        return
      }
    }
  }
}

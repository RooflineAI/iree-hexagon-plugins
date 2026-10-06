// Verifies that each Hexagon translation_info pipeline builds and runs its
// pipeline-specific passes on the expected dispatch function.

// RUN: iree-opt \
// RUN:   --pass-pipeline='builtin.module(hal.executable(hal.executable.variant(builtin.module(func.func(iree-hexagon-lower-executable-target)))))' \
// RUN:   --mlir-print-ir-after-all \
// RUN:   --split-input-file %s -o /dev/null 2>&1 | FileCheck %s

// CHECK-LABEL: IR Dump After HexagonTileAndFuseProducerConsumerPass
// CHECK: func.func @default_dispatch

#translation = #iree_codegen.translation_info<pipeline = #iree_hexagon.pipeline<Default>>
#executable_target_embedded_elf_hexagon = #hal.executable.target<"hexagon", "embedded-elf-hexagon", {cpu = "hexagonv79", cpu_features = "+hvxv79,+hvx-length128b", data_layout = "e-m:e-p:32:32:32-a:0-n16:32-i64:64:64-i32:32:32-i16:16:16-i1:8:8-f32:32:32-f64:64:64-v32:32:32-v64:64:64-v512:512:512-v1024:1024:1024-v2048:2048:2048", hexagon.version = "79", iree.encoding.resolver = #iree_hexagon.hexagon_encoding_resolver<>, max_stack_allocation_size = 16384 : i64, native_vector_size = 32 : i64, target_triple = "hexagon-unknown-unknown-elf"}>
#pipeline_layout = #hal.pipeline.layout<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>

hal.executable private @test_default {
  hal.executable.variant public @embedded_elf_hexagon target(#executable_target_embedded_elf_hexagon) {
    builtin.module {
      func.func @default_dispatch() attributes {translation_info = #translation} {
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

// CHECK-LABEL: IR Dump After HexagonPeelPass
// CHECK: func.func @buffer_dispatch
// CHECK-LABEL: IR Dump After HexagonVerifyVectorSizeLegalityPass
// CHECK: func.func @buffer_dispatch
// CHECK-LABEL: IR Dump After HexagonVirtualVectorLoweringPass
// CHECK: func.func @buffer_dispatch
// CHECK-LABEL: IR Dump After HexagonVectorTransposeLoweringPass
// CHECK: func.func @buffer_dispatch
// CHECK-LABEL: IR Dump After HexagonVectorShapeCastLoweringPass
// CHECK: func.func @buffer_dispatch

#executable_target_embedded_elf_hexagon = #hal.executable.target<"hexagon", "embedded-elf-hexagon", {cpu = "hexagonv79", cpu_features = "+hvxv79,+hvx-length128b", data_layout = "e-m:e-p:32:32:32-a:0-n16:32-i64:64:64-i32:32:32-i16:16:16-i1:8:8-f32:32:32-f64:64:64-v32:32:32-v64:64:64-v512:512:512-v1024:1024:1024-v2048:2048:2048", hexagon.version = "79", iree.encoding.resolver = #iree_hexagon.hexagon_encoding_resolver<>, max_stack_allocation_size = 16384 : i64, native_vector_size = 32 : i64, target_triple = "hexagon-unknown-unknown-elf"}>
#pipeline_layout = #hal.pipeline.layout<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>
#translation_buffer = #iree_codegen.translation_info<pipeline = #iree_hexagon.pipeline<BufferOpsTileAndVectorize>, {enable_loop_peeling}>

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

// CHECK-LABEL: IR Dump After HexagonSplitReductionPass
// CHECK: func.func @double_tiling_dispatch
// CHECK-LABEL: IR Dump After HexagonPeelPass
// CHECK: func.func @double_tiling_dispatch
// CHECK-LABEL: IR Dump After HexagonTileToVectorSizePass
// CHECK: func.func @double_tiling_dispatch

#executable_target_embedded_elf_hexagon = #hal.executable.target<"hexagon", "embedded-elf-hexagon", {cpu = "hexagonv79", cpu_features = "+hvxv79,+hvx-length128b", data_layout = "e-m:e-p:32:32:32-a:0-n16:32-i64:64:64-i32:32:32-i16:16:16-i1:8:8-f32:32:32-f64:64:64-v32:32:32-v64:64:64-v512:512:512-v1024:1024:1024-v2048:2048:2048", hexagon.version = "79", iree.encoding.resolver = #iree_hexagon.hexagon_encoding_resolver<>, max_stack_allocation_size = 16384 : i64, native_vector_size = 32 : i64, target_triple = "hexagon-unknown-unknown-elf"}>
#pipeline_layout = #hal.pipeline.layout<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>
#translation_double = #iree_codegen.translation_info<pipeline = #iree_hexagon.pipeline<MultiTilingExpert>, {enable_loop_peeling}>
#config_double = #iree_hexagon.lowering_config<distribution = [1, 1, 0], vector_common_parallel = [1, 1, 0], vector_reduction = [0, 0, 1]>

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
// CHECK-LABEL: IR Dump After HexagonPeelPass
// CHECK: func.func @conv_dispatch

#executable_target_embedded_elf_hexagon = #hal.executable.target<"hexagon", "embedded-elf-hexagon", {cpu = "hexagonv79", cpu_features = "+hvxv79,+hvx-length128b", data_layout = "e-m:e-p:32:32:32-a:0-n16:32-i64:64:64-i32:32:32-i16:16:16-i1:8:8-f32:32:32-f64:64:64-v32:32:32-v64:64:64-v512:512:512-v1024:1024:1024-v2048:2048:2048", hexagon.version = "79", iree.encoding.resolver = #iree_hexagon.hexagon_encoding_resolver<>, max_stack_allocation_size = 16384 : i64, native_vector_size = 32 : i64, target_triple = "hexagon-unknown-unknown-elf"}>
#pipeline_layout = #hal.pipeline.layout<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>
#translation_conv = #iree_codegen.translation_info<pipeline = #iree_hexagon.pipeline<ConvTileAndDecomposeExpert>, {enable_loop_peeling}>

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

// CHECK-LABEL: IR Dump After HexagonVectorTransposeLoweringPass
// CHECK: func.func @data_tiling_dispatch

#executable_target_embedded_elf_hexagon = #hal.executable.target<"hexagon", "embedded-elf-hexagon", {cpu = "hexagonv79", cpu_features = "+hvxv79,+hvx-length128b", data_layout = "e-m:e-p:32:32:32-a:0-n16:32-i64:64:64-i32:32:32-i16:16:16-i1:8:8-f32:32:32-f64:64:64-v32:32:32-v64:64:64-v512:512:512-v1024:1024:1024-v2048:2048:2048", hexagon.version = "79", iree.encoding.resolver = #iree_hexagon.hexagon_encoding_resolver<>, max_stack_allocation_size = 16384 : i64, native_vector_size = 32 : i64, target_triple = "hexagon-unknown-unknown-elf"}>
#pipeline_layout = #hal.pipeline.layout<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>
#translation_data = #iree_codegen.translation_info<pipeline = #iree_hexagon.pipeline<DataTiling>>

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

// CHECK-LABEL: IR Dump After HexagonConvertMatmulToHmxPass
// CHECK: func.func @matmul_dispatch

#executable_target_embedded_elf_hexagon = #hal.executable.target<"hexagon", "embedded-elf-hexagon", {cpu = "hexagonv79", cpu_features = "+hvxv79,+hvx-length128b", data_layout = "e-m:e-p:32:32:32-a:0-n16:32-i64:64:64-i32:32:32-i16:16:16-i1:8:8-f32:32:32-f64:64:64-v32:32:32-v64:64:64-v512:512:512-v1024:1024:1024-v2048:2048:2048", hexagon.version = "79", iree.encoding.resolver = #iree_hexagon.hexagon_encoding_resolver<>, max_stack_allocation_size = 16384 : i64, native_vector_size = 32 : i64, target_triple = "hexagon-unknown-unknown-elf"}>
#pipeline_layout = #hal.pipeline.layout<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>
#config_fill = #iree_hexagon.lowering_config<vector_common_parallel = [16, 16]>
#config_matmul = #iree_hexagon.lowering_config<distribution = [1, 1, 0], vector_common_parallel = [16, 16, 0], vector_reduction = [0, 0, 1]>
#translation = #iree_codegen.translation_info<pipeline = #iree_hexagon.pipeline<HmxMatmulExpert>>

hal.executable private @test {
  hal.executable.variant public @embedded_elf_hexagon target(#executable_target_embedded_elf_hexagon) {
    builtin.module {
      func.func @matmul_dispatch() attributes {translation_info = #translation} {
        %cst = arith.constant 0.000000e+00 : f32
        %c0 = arith.constant 0 : index
        %c256 = arith.constant 256 : index
        %lhs = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags("ReadOnly|Indirect") : !iree_tensor_ext.dispatch.tensor<readonly:tensor<16x4xf32>>
        %rhs = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c256) flags("ReadOnly|Indirect") : !iree_tensor_ext.dispatch.tensor<readonly:tensor<4x16xf32>>
        %out = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) flags(Indirect) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<16x16xf32>>

        %lhs_t = iree_tensor_ext.dispatch.tensor.load %lhs, offsets = [0, 0], sizes = [16, 4], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<16x4xf32>> -> tensor<16x4xf32>
        %rhs_t = iree_tensor_ext.dispatch.tensor.load %rhs, offsets = [0, 0], sizes = [4, 16], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<4x16xf32>> -> tensor<4x16xf32>

        %init = tensor.empty() : tensor<16x16xf32>
        %filled = linalg.fill {lowering_config = #config_fill} ins(%cst : f32) outs(%init : tensor<16x16xf32>) -> tensor<16x16xf32>
        %result = linalg.matmul {lowering_config = #config_matmul} ins(%lhs_t, %rhs_t : tensor<16x4xf32>, tensor<4x16xf32>) outs(%filled : tensor<16x16xf32>) -> tensor<16x16xf32>

        iree_tensor_ext.dispatch.tensor.store %result, %out, offsets = [0, 0], sizes = [16, 16], strides = [1, 1] : tensor<16x16xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<16x16xf32>>
        return
      }
    }
  }
}

// -----

// CHECK-LABEL: IR Dump After DecomposeAttentionPass
// CHECK: func.func @linalg_ext_dispatch
// CHECK-NOT: iree_linalg_ext.online_attention
// CHECK: linalg.generic

#translation = #iree_codegen.translation_info<pipeline = #iree_hexagon.pipeline<LinalgExtTileAndVectorize>>
#executable_target_embedded_elf_hexagon = #hal.executable.target<"hexagon", "embedded-elf-hexagon", {cpu = "hexagonv79", cpu_features = "+hvxv79,+hvx-length128b", data_layout = "e-m:e-p:32:32:32-a:0-n16:32-i64:64:64-i32:32:32-i16:16:16-i1:8:8-f32:32:32-f64:64:64-v32:32:32-v64:64:64-v512:512:512-v1024:1024:1024-v2048:2048:2048", hexagon.version = "79", iree.encoding.resolver = #iree_hexagon.hexagon_encoding_resolver<>, max_stack_allocation_size = 16384 : i64, native_vector_size = 32 : i64, target_triple = "hexagon-unknown-unknown-elf"}>
#pipeline_layout = #hal.pipeline.layout<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>
#config = #iree_hexagon.lowering_config<distribution = [1, 1, 0, 0, 1], vector_common_parallel = [1, 1, 0, 0, 1], vector_reduction = [0, 0, 0, 1, 0]>
#map_q = affine_map<(batch, m, k1, k2, n) -> (batch, m, k1)>
#map_k = affine_map<(batch, m, k1, k2, n) -> (batch, k2, k1)>
#map_v = affine_map<(batch, m, k1, k2, n) -> (batch, k2, n)>
#map_scale = affine_map<(batch, m, k1, k2, n) -> ()>
#map_out = affine_map<(batch, m, k1, k2, n) -> (batch, m, n)>
#map_stat = affine_map<(batch, m, k1, k2, n) -> (batch, m)>

hal.executable private @test_linalg_ext {
  hal.executable.variant public @embedded_elf_hexagon target(#executable_target_embedded_elf_hexagon) {
    builtin.module {
      func.func @linalg_ext_dispatch() attributes {translation_info = #translation} {
        %c0 = arith.constant 0 : index
        %scale = arith.constant 0.5 : f16
        %zero = arith.constant 0.0 : f32
        %neg_inf = arith.constant -3.40282347E+38 : f32
        %query = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags("ReadOnly|Indirect") : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x2x4xf16>>
        %key = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) flags("ReadOnly|Indirect") : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x3x4xf16>>
        %value = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0) flags("ReadOnly|Indirect") : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x3x5xf16>>
        %output = hal.interface.binding.subspan layout(#pipeline_layout) binding(3) alignment(64) offset(%c0) flags(Indirect) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1x2x5xf32>>

        %query_t = iree_tensor_ext.dispatch.tensor.load %query, offsets = [0, 0, 0], sizes = [1, 2, 4], strides = [1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x2x4xf16>> -> tensor<1x2x4xf16>
        %key_t = iree_tensor_ext.dispatch.tensor.load %key, offsets = [0, 0, 0], sizes = [1, 3, 4], strides = [1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x3x4xf16>> -> tensor<1x3x4xf16>
        %value_t = iree_tensor_ext.dispatch.tensor.load %value, offsets = [0, 0, 0], sizes = [1, 3, 5], strides = [1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x3x5xf16>> -> tensor<1x3x5xf16>

        %acc = tensor.empty() : tensor<1x2x5xf32>
        %stat = tensor.empty() : tensor<1x2xf32>
        %filled_acc = linalg.fill ins(%zero : f32) outs(%acc : tensor<1x2x5xf32>) -> tensor<1x2x5xf32>
        %filled_max = linalg.fill ins(%neg_inf : f32) outs(%stat : tensor<1x2xf32>) -> tensor<1x2xf32>
        %filled_sum = linalg.fill ins(%zero : f32) outs(%stat : tensor<1x2xf32>) -> tensor<1x2xf32>
        %result:3 = iree_linalg_ext.online_attention {
          indexing_maps = [#map_q, #map_k, #map_v, #map_scale, #map_out, #map_stat, #map_stat],
          lowering_config = #config
        } ins(%query_t, %key_t, %value_t, %scale : tensor<1x2x4xf16>, tensor<1x3x4xf16>, tensor<1x3x5xf16>, f16)
          outs(%filled_acc, %filled_max, %filled_sum : tensor<1x2x5xf32>, tensor<1x2xf32>, tensor<1x2xf32>) {
        ^bb0(%score: f32):
          iree_linalg_ext.yield %score : f32
        } -> tensor<1x2x5xf32>, tensor<1x2xf32>, tensor<1x2xf32>

        iree_tensor_ext.dispatch.tensor.store %result#0, %output, offsets = [0, 0, 0], sizes = [1, 2, 5], strides = [1, 1, 1] : tensor<1x2x5xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1x2x5xf32>>
        return
      }
    }
  }
}

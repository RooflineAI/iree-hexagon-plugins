// Test written to reflect current Hexagon selector policy.
// This file covers fill, transpose-like generic, reduction root, matmul with
// epilogue, conv fallback, fft fallback, and dot fallback.
// 
// This file is not intended as a long-term reference of expected behavior,
// but it keeps track of the current state of the kernelDispatch.cpp file. 
//
// Compared to LLVMCPU today:
//   fill stays on CPUDefault for both, but tile sizes differ;
//   transpose stays on CPUDefault on Hexagon but uses CPUDoubleTilingExpert on LLVMCPU;
//   reduction and matmul epilogues use the same expert pipeline but different tiles;
//   conv, fft, and dot intentionally diverge in pipeline choice.
//
// Study manually with:
// iree-opt --iree-load-plugin=cellar_hexagon=$CELLAR_HEXAGON_COMPILER_PLUGIN \
//   --pass-pipeline='builtin.module(hal.executable(hal.executable.variant(builtin.module(iree-hexagon-select-lowering-strategy))))' \
//   --split-input-file plugins/external/cellar-hexagon/plugins/compiler/cellar-hexagon/test/codegen/strategy/hexagon_select_lowering_strategy_study.mlir
//
// RUN: iree-opt --iree-load-plugin=cellar_hexagon=$CELLAR_HEXAGON_COMPILER_PLUGIN \
// RUN:   --pass-pipeline='builtin.module(hal.executable(hal.executable.variant(builtin.module(iree-hexagon-select-lowering-strategy))))' \
// RUN:   --split-input-file %s | FileCheck %s

// CHECK-DAG: #config = #iree_cpu.lowering_config<vector_common_parallel = [1, 32]>
// CHECK-DAG: #translation = #iree_codegen.translation_info<pipeline = CPUDefault>
// CHECK-LABEL: func.func @fill_root_dispatch()
// CHECK: linalg.fill

#target = #hal.executable.target<"hexagon", "embedded-elf-hexagon", {cpu = "hexagonv79", cpu_features = "+hvxv79,+hvx-length128b", data_layout = "e-m:e-p:32:32:32-a:0-n16:32-i64:64:64-i32:32:32-i16:16:16-i1:8:8-f32:32:32-f64:64:64-v32:32:32-v64:64:64-v512:512:512-v1024:1024:1024-v2048:2048:2048", hexagon.version = "79", iree.encoding.resolver = #iree_hexagon.hexagon_encoding_resolver<>, link_embedded = false, max_stack_allocation_size = 16384 : i64, native_vector_size = 32 : i64, target_triple = "hexagon-unknown-unknown-elf"}>
#layout = #hal.pipeline.layout<bindings = [#hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>

hal.executable private @fill_root {
  hal.executable.variant public @embedded_elf_hexagon target(#target) {
    builtin.module {
      func.func @fill_root_dispatch() {
        %c0 = arith.constant 0 : index
        %cst = arith.constant 0.0 : f32
        %out = hal.interface.binding.subspan layout(#layout) binding(0) alignment(64) offset(%c0) flags(Indirect) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<64x128xf32>>
        %empty = tensor.empty() : tensor<64x128xf32>
        %filled = linalg.fill ins(%cst : f32) outs(%empty : tensor<64x128xf32>) -> tensor<64x128xf32>
        iree_tensor_ext.dispatch.tensor.store %filled, %out, offsets = [0, 0], sizes = [64, 128], strides = [1, 1] : tensor<64x128xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<64x128xf32>>
        return
      }
    }
  }
}

// -----

// CHECK-DAG: #config = #iree_cpu.lowering_config<vector_common_parallel = [8, 8]>
// CHECK-DAG: #translation = #iree_codegen.translation_info<pipeline = CPUDefault>
// CHECK-LABEL: func.func @transpose_root_dispatch()
// CHECK: linalg.generic
// CHECK-SAME: iterator_types = ["parallel", "parallel"]

#target = #hal.executable.target<"hexagon", "embedded-elf-hexagon", {cpu = "hexagonv79", cpu_features = "+hvxv79,+hvx-length128b", data_layout = "e-m:e-p:32:32:32-a:0-n16:32-i64:64:64-i32:32:32-i16:16:16-i1:8:8-f32:32:32-f64:64:64-v32:32:32-v64:64:64-v512:512:512-v1024:1024:1024-v2048:2048:2048", hexagon.version = "79", iree.encoding.resolver = #iree_hexagon.hexagon_encoding_resolver<>, link_embedded = false, max_stack_allocation_size = 16384 : i64, native_vector_size = 32 : i64, target_triple = "hexagon-unknown-unknown-elf"}>
#layout = #hal.pipeline.layout<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>

hal.executable private @transpose_root {
  hal.executable.variant public @embedded_elf_hexagon target(#target) {
    builtin.module {
      func.func @transpose_root_dispatch() {
        %c0 = arith.constant 0 : index
        %src = hal.interface.binding.subspan layout(#layout) binding(0) alignment(64) offset(%c0) flags("ReadOnly|Indirect") : !iree_tensor_ext.dispatch.tensor<readonly:tensor<64x128xf32>>
        %out = hal.interface.binding.subspan layout(#layout) binding(1) alignment(64) offset(%c0) flags(Indirect) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<128x64xf32>>
        %src_t = iree_tensor_ext.dispatch.tensor.load %src, offsets = [0, 0], sizes = [64, 128], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<64x128xf32>> -> tensor<64x128xf32>
        %empty = tensor.empty() : tensor<128x64xf32>
        %transposed = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d1, d0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%src_t : tensor<64x128xf32>) outs(%empty : tensor<128x64xf32>) {
        ^bb0(%in: f32, %out0: f32):
          linalg.yield %in : f32
        } -> tensor<128x64xf32>
        iree_tensor_ext.dispatch.tensor.store %transposed, %out, offsets = [0, 0], sizes = [128, 64], strides = [1, 1] : tensor<128x64xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<128x64xf32>>
        return
      }
    }
  }
}

// -----

// CHECK-DAG: #config = #iree_cpu.lowering_config<vector_common_parallel = [1, 32]>
// CHECK-DAG: #config1 = #iree_cpu.lowering_config<distribution = [1, 16, 0], vector_common_parallel = [1, 32, 0], vector_reduction = [0, 0, 8]>
// CHECK-DAG: #translation = #iree_codegen.translation_info<pipeline = CPUDoubleTilingExpert>
// CHECK-LABEL: func.func @reduction_root_dispatch()
// CHECK: linalg.fill
// CHECK: linalg.generic
// CHECK-SAME: iterator_types = ["parallel", "parallel", "reduction"]
// CHECK: linalg.generic
// CHECK-SAME: iterator_types = ["parallel", "parallel"]

#target = #hal.executable.target<"hexagon", "embedded-elf-hexagon", {cpu = "hexagonv79", cpu_features = "+hvxv79,+hvx-length128b", data_layout = "e-m:e-p:32:32:32-a:0-n16:32-i64:64:64-i32:32:32-i16:16:16-i1:8:8-f32:32:32-f64:64:64-v32:32:32-v64:64:64-v512:512:512-v1024:1024:1024-v2048:2048:2048", hexagon.version = "79", iree.encoding.resolver = #iree_hexagon.hexagon_encoding_resolver<>, link_embedded = false, max_stack_allocation_size = 16384 : i64, native_vector_size = 32 : i64, target_triple = "hexagon-unknown-unknown-elf"}>
#layout = #hal.pipeline.layout<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>

hal.executable private @reduction_root {
  hal.executable.variant public @embedded_elf_hexagon target(#target) {
    builtin.module {
      func.func @reduction_root_dispatch() {
        %c0 = arith.constant 0 : index
        %cst = arith.constant 0.0 : f32
        %scale_cst = arith.constant 0.5 : f32
        %src = hal.interface.binding.subspan layout(#layout) binding(0) alignment(64) offset(%c0) flags("ReadOnly|Indirect") : !iree_tensor_ext.dispatch.tensor<readonly:tensor<4x128x128xf32>>
        %scale = hal.interface.binding.subspan layout(#layout) binding(1) alignment(64) offset(%c0) flags("ReadOnly|Indirect") : !iree_tensor_ext.dispatch.tensor<readonly:tensor<4x128xf32>>
        %out = hal.interface.binding.subspan layout(#layout) binding(2) alignment(64) offset(%c0) flags(Indirect) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<4x128xf32>>
        %src_t = iree_tensor_ext.dispatch.tensor.load %src, offsets = [0, 0, 0], sizes = [4, 128, 128], strides = [1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<4x128x128xf32>> -> tensor<4x128x128xf32>
        %scale_t = iree_tensor_ext.dispatch.tensor.load %scale, offsets = [0, 0], sizes = [4, 128], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<4x128xf32>> -> tensor<4x128xf32>
        %red_empty = tensor.empty() : tensor<4x128xf32>
        %red_init = linalg.fill ins(%cst : f32) outs(%red_empty : tensor<4x128xf32>) -> tensor<4x128xf32>
        %reduced = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%src_t : tensor<4x128x128xf32>) outs(%red_init : tensor<4x128xf32>) {
        ^bb0(%in: f32, %out0: f32):
          %sum = arith.addf %in, %out0 : f32
          linalg.yield %sum : f32
        } -> tensor<4x128xf32>
        %ew_empty = tensor.empty() : tensor<4x128xf32>
        %ew_init = linalg.fill ins(%cst : f32) outs(%ew_empty : tensor<4x128xf32>) -> tensor<4x128xf32>
        %scaled = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%reduced, %scale_t : tensor<4x128xf32>, tensor<4x128xf32>) outs(%ew_init : tensor<4x128xf32>) {
        ^bb0(%in: f32, %scale_in: f32, %out0: f32):
          %scaled0 = arith.mulf %in, %scale_in : f32
          %scaled1 = arith.addf %scaled0, %scale_cst : f32
          linalg.yield %scaled1 : f32
        } -> tensor<4x128xf32>
        iree_tensor_ext.dispatch.tensor.store %scaled, %out, offsets = [0, 0], sizes = [4, 128], strides = [1, 1] : tensor<4x128xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<4x128xf32>>
        return
      }
    }
  }
}

// -----

// CHECK-DAG: #config = #iree_cpu.lowering_config<vector_common_parallel = [1, 32]>
// CHECK-DAG: #config1 = #iree_cpu.lowering_config<distribution = [16, 16, 0], vector_common_parallel = [8, 32, 0], vector_reduction = [0, 0, 8]>
// CHECK-DAG: #translation = #iree_codegen.translation_info<pipeline = CPUDoubleTilingExpert>
// CHECK-LABEL: func.func @matmul_epilogue_dispatch()
// CHECK: linalg.fill
// CHECK: linalg.matmul
// CHECK: linalg.generic
// CHECK-SAME: iterator_types = ["parallel", "parallel"]

#target = #hal.executable.target<"hexagon", "embedded-elf-hexagon", {cpu = "hexagonv79", cpu_features = "+hvxv79,+hvx-length128b", data_layout = "e-m:e-p:32:32:32-a:0-n16:32-i64:64:64-i32:32:32-i16:16:16-i1:8:8-f32:32:32-f64:64:64-v32:32:32-v64:64:64-v512:512:512-v1024:1024:1024-v2048:2048:2048", hexagon.version = "79", iree.encoding.resolver = #iree_hexagon.hexagon_encoding_resolver<>, link_embedded = false, max_stack_allocation_size = 16384 : i64, native_vector_size = 32 : i64, target_triple = "hexagon-unknown-unknown-elf"}>
#layout = #hal.pipeline.layout<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>

hal.executable private @matmul_epilogue_root {
  hal.executable.variant public @embedded_elf_hexagon target(#target) {
    builtin.module {
      func.func @matmul_epilogue_dispatch() {
        %c0 = arith.constant 0 : index
        %cst = arith.constant 0.0 : f32
        %lhs = hal.interface.binding.subspan layout(#layout) binding(0) alignment(64) offset(%c0) flags("ReadOnly|Indirect") : !iree_tensor_ext.dispatch.tensor<readonly:tensor<128x128xf32>>
        %rhs = hal.interface.binding.subspan layout(#layout) binding(1) alignment(64) offset(%c0) flags("ReadOnly|Indirect") : !iree_tensor_ext.dispatch.tensor<readonly:tensor<128x128xf32>>
        %bias = hal.interface.binding.subspan layout(#layout) binding(2) alignment(64) offset(%c0) flags("ReadOnly|Indirect") : !iree_tensor_ext.dispatch.tensor<readonly:tensor<128x128xf32>>
        %out = hal.interface.binding.subspan layout(#layout) binding(3) alignment(64) offset(%c0) flags(Indirect) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<128x128xf32>>
        %lhs_t = iree_tensor_ext.dispatch.tensor.load %lhs, offsets = [0, 0], sizes = [128, 128], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<128x128xf32>> -> tensor<128x128xf32>
        %rhs_t = iree_tensor_ext.dispatch.tensor.load %rhs, offsets = [0, 0], sizes = [128, 128], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<128x128xf32>> -> tensor<128x128xf32>
        %bias_t = iree_tensor_ext.dispatch.tensor.load %bias, offsets = [0, 0], sizes = [128, 128], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<128x128xf32>> -> tensor<128x128xf32>
        %empty = tensor.empty() : tensor<128x128xf32>
        %init = linalg.fill ins(%cst : f32) outs(%empty : tensor<128x128xf32>) -> tensor<128x128xf32>
        %matmul = linalg.matmul ins(%lhs_t, %rhs_t : tensor<128x128xf32>, tensor<128x128xf32>) outs(%init : tensor<128x128xf32>) -> tensor<128x128xf32>
        %epilogue_empty = tensor.empty() : tensor<128x128xf32>
        %epilogue_init = linalg.fill ins(%cst : f32) outs(%epilogue_empty : tensor<128x128xf32>) -> tensor<128x128xf32>
        %result = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%matmul, %bias_t : tensor<128x128xf32>, tensor<128x128xf32>) outs(%epilogue_init : tensor<128x128xf32>) {
        ^bb0(%acc: f32, %bias_in: f32, %out0: f32):
          %sum = arith.addf %acc, %bias_in : f32
          linalg.yield %sum : f32
        } -> tensor<128x128xf32>
        iree_tensor_ext.dispatch.tensor.store %result, %out, offsets = [0, 0], sizes = [128, 128], strides = [1, 1] : tensor<128x128xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<128x128xf32>>
        return
      }
    }
  }
}

// -----

// CHECK-DAG: #config = #iree_cpu.lowering_config<vector_common_parallel = [1, 1, 1, 16]>
// CHECK-DAG: #config1 = #iree_cpu.lowering_config<distribution = [1, 0, 0, 2, 0, 0, 0], vector_common_parallel = [1, 0, 0, 16, 0, 0, 0], vector_reduction = [0, 0, 0, 0, 0, 0, 8]>
// CHECK-DAG: #translation = #iree_codegen.translation_info<pipeline = CPUDoubleTilingExpert>
// CHECK-LABEL: func.func @conv_fallback_dispatch()
// CHECK: linalg.fill
// CHECK: linalg.conv_2d_nhwc_hwcf
// CHECK-SAME: lowering_config =

#target = #hal.executable.target<"hexagon", "embedded-elf-hexagon", {cpu = "hexagonv79", cpu_features = "+hvxv79,+hvx-length128b", data_layout = "e-m:e-p:32:32:32-a:0-n16:32-i64:64:64-i32:32:32-i16:16:16-i1:8:8-f32:32:32-f64:64:64-v32:32:32-v64:64:64-v512:512:512-v1024:1024:1024-v2048:2048:2048", hexagon.version = "79", iree.encoding.resolver = #iree_hexagon.hexagon_encoding_resolver<>, link_embedded = false, max_stack_allocation_size = 16384 : i64, native_vector_size = 32 : i64, target_triple = "hexagon-unknown-unknown-elf"}>
#layout = #hal.pipeline.layout<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>

hal.executable private @conv_fallback_root {
  hal.executable.variant public @embedded_elf_hexagon target(#target) {
    builtin.module {
      func.func @conv_fallback_dispatch() {
        %c0 = arith.constant 0 : index
        %cst = arith.constant 0.0 : f32
        %input = hal.interface.binding.subspan layout(#layout) binding(0) alignment(64) offset(%c0) flags("ReadOnly|Indirect") : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x32x32x8xf32>>
        %filter = hal.interface.binding.subspan layout(#layout) binding(1) alignment(64) offset(%c0) flags("ReadOnly|Indirect") : !iree_tensor_ext.dispatch.tensor<readonly:tensor<3x3x8x16xf32>>
        %out = hal.interface.binding.subspan layout(#layout) binding(2) alignment(64) offset(%c0) flags(Indirect) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1x30x30x16xf32>>
        %input_t = iree_tensor_ext.dispatch.tensor.load %input, offsets = [0, 0, 0, 0], sizes = [1, 32, 32, 8], strides = [1, 1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x32x32x8xf32>> -> tensor<1x32x32x8xf32>
        %filter_t = iree_tensor_ext.dispatch.tensor.load %filter, offsets = [0, 0, 0, 0], sizes = [3, 3, 8, 16], strides = [1, 1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<3x3x8x16xf32>> -> tensor<3x3x8x16xf32>
        %empty = tensor.empty() : tensor<1x30x30x16xf32>
        %init = linalg.fill ins(%cst : f32) outs(%empty : tensor<1x30x30x16xf32>) -> tensor<1x30x30x16xf32>
        %result = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%input_t, %filter_t : tensor<1x32x32x8xf32>, tensor<3x3x8x16xf32>) outs(%init : tensor<1x30x30x16xf32>) -> tensor<1x30x30x16xf32>
        iree_tensor_ext.dispatch.tensor.store %result, %out, offsets = [0, 0, 0, 0], sizes = [1, 30, 30, 16], strides = [1, 1, 1, 1] : tensor<1x30x30x16xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1x30x30x16xf32>>
        return
      }
    }
  }
}

// -----

// CHECK-DAG: #config = #iree_cpu.lowering_config<>
// CHECK-DAG: #translation = #iree_codegen.translation_info<pipeline = CPUDefault>
// CHECK-LABEL: func.func @fft_fallback_dispatch()
// CHECK: iree_linalg_ext.fft
// CHECK-SAME: lowering_config =

#target = #hal.executable.target<"hexagon", "embedded-elf-hexagon", {cpu = "hexagonv79", cpu_features = "+hvxv79,+hvx-length128b", data_layout = "e-m:e-p:32:32:32-a:0-n16:32-i64:64:64-i32:32:32-i16:16:16-i1:8:8-f32:32:32-f64:64:64-v32:32:32-v64:64:64-v512:512:512-v1024:1024:1024-v2048:2048:2048", hexagon.version = "79", iree.encoding.resolver = #iree_hexagon.hexagon_encoding_resolver<>, link_embedded = false, max_stack_allocation_size = 16384 : i64, native_vector_size = 32 : i64, target_triple = "hexagon-unknown-unknown-elf"}>
#layout = #hal.pipeline.layout<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>

hal.executable private @fft_fallback_root {
  hal.executable.variant public @embedded_elf_hexagon target(#target) {
    builtin.module {
      func.func @fft_fallback_dispatch() {
        %c0 = arith.constant 0 : index
        %c2 = arith.constant 2 : index
        %twiddle_real = hal.interface.binding.subspan layout(#layout) binding(0) alignment(64) offset(%c0) flags("ReadOnly|Indirect") : !iree_tensor_ext.dispatch.tensor<readonly:tensor<2xf32>>
        %twiddle_imag = hal.interface.binding.subspan layout(#layout) binding(1) alignment(64) offset(%c0) flags("ReadOnly|Indirect") : !iree_tensor_ext.dispatch.tensor<readonly:tensor<2xf32>>
        %out_real = hal.interface.binding.subspan layout(#layout) binding(2) alignment(64) offset(%c0) flags(Indirect) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<32xf32>>
        %out_imag = hal.interface.binding.subspan layout(#layout) binding(3) alignment(64) offset(%c0) flags(Indirect) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<32xf32>>
        %twiddle_real_t = iree_tensor_ext.dispatch.tensor.load %twiddle_real, offsets = [0], sizes = [2], strides = [1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<2xf32>> -> tensor<2xf32>
        %twiddle_imag_t = iree_tensor_ext.dispatch.tensor.load %twiddle_imag, offsets = [0], sizes = [2], strides = [1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<2xf32>> -> tensor<2xf32>
        %empty_real = tensor.empty() : tensor<32xf32>
        %empty_imag = tensor.empty() : tensor<32xf32>
        %fft_real, %fft_imag = iree_linalg_ext.fft ins(%c2, %twiddle_real_t, %twiddle_imag_t : index, tensor<2xf32>, tensor<2xf32>) outs(%empty_real, %empty_imag : tensor<32xf32>, tensor<32xf32>) : tensor<32xf32>, tensor<32xf32>
        iree_tensor_ext.dispatch.tensor.store %fft_real, %out_real, offsets = [0], sizes = [32], strides = [1] : tensor<32xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<32xf32>>
        iree_tensor_ext.dispatch.tensor.store %fft_imag, %out_imag, offsets = [0], sizes = [32], strides = [1] : tensor<32xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<32xf32>>
        return
      }
    }
  }
}

// -----

// Dot-like contractions are rejected by Hexagon's matmul policy.
// Until Hexagon grows a dedicated reduction strategy for them, they should
// cleanly fall back to the generic CPUDefault path

// CHECK-DAG: #translation = #iree_codegen.translation_info<pipeline = CPUDefault>
// CHECK-LABEL: func.func @dot_dispatch()
// CHECK: linalg.dot

#target = #hal.executable.target<"hexagon", "embedded-elf-hexagon", {cpu = "hexagonv79", cpu_features = "+hvxv79,+hvx-length128b", data_layout = "e-m:e-p:32:32:32-a:0-n16:32-i64:64:64-i32:32:32-i16:16:16-i1:8:8-f32:32:32-f64:64:64-v32:32:32-v64:64:64-v512:512:512-v1024:1024:1024-v2048:2048:2048", hexagon.version = "79", iree.encoding.resolver = #iree_hexagon.hexagon_encoding_resolver<>, link_embedded = false, max_stack_allocation_size = 16384 : i64, native_vector_size = 32 : i64, target_triple = "hexagon-unknown-unknown-elf"}>
#layout = #hal.pipeline.layout<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>

hal.executable private @dot_root {
  hal.executable.variant public @embedded_elf_hexagon target(#target) {
    builtin.module {
      func.func @dot_dispatch() {
        %c0 = arith.constant 0 : index
        %cst = arith.constant 0.0 : f32
        %lhs = hal.interface.binding.subspan layout(#layout) binding(0) alignment(64) offset(%c0) flags("ReadOnly|Indirect") : !iree_tensor_ext.dispatch.tensor<readonly:tensor<128xf32>>
        %rhs = hal.interface.binding.subspan layout(#layout) binding(1) alignment(64) offset(%c0) flags("ReadOnly|Indirect") : !iree_tensor_ext.dispatch.tensor<readonly:tensor<128xf32>>
        %out = hal.interface.binding.subspan layout(#layout) binding(2) alignment(64) offset(%c0) flags(Indirect) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<f32>>
        %lhs_t = iree_tensor_ext.dispatch.tensor.load %lhs, offsets = [0], sizes = [128], strides = [1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<128xf32>> -> tensor<128xf32>
        %rhs_t = iree_tensor_ext.dispatch.tensor.load %rhs, offsets = [0], sizes = [128], strides = [1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<128xf32>> -> tensor<128xf32>
        %empty = tensor.empty() : tensor<f32>
        %init = linalg.fill ins(%cst : f32) outs(%empty : tensor<f32>) -> tensor<f32>
        %result = linalg.dot ins(%lhs_t, %rhs_t : tensor<128xf32>, tensor<128xf32>) outs(%init : tensor<f32>) -> tensor<f32>
        iree_tensor_ext.dispatch.tensor.store %result, %out, offsets = [], sizes = [], strides = [] : tensor<f32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<f32>>
        return
      }
    }
  }
}

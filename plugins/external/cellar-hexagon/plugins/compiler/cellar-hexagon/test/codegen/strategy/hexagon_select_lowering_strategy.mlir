// RUN: iree-opt --iree-load-plugin=cellar_hexagon=$CELLAR_HEXAGON_COMPILER_PLUGIN \
// RUN:   --pass-pipeline='builtin.module(hal.executable(hal.executable.variant(builtin.module(iree-hexagon-select-lowering-strategy))))' \
// RUN:   --split-input-file %s | FileCheck %s

#target = #hal.executable.target<"hexagon", "embedded-elf-hexagon", {cpu = "hexagonv79", cpu_features = "+hvxv79,+hvx-length128b", data_layout = "e-m:e-p:32:32:32-a:0-n16:32-i64:64:64-i32:32:32-i16:16:16-i1:8:8-f32:32:32-f64:64:64-v32:32:32-v64:64:64-v512:512:512-v1024:1024:1024-v2048:2048:2048", hexagon.version = "79", iree.encoding.resolver = #iree_hexagon.hexagon_encoding_resolver<>, link_embedded = false, max_stack_allocation_size = 16384 : i64, native_vector_size = 32 : i64, target_triple = "hexagon-unknown-unknown-elf"}>
#layout = #hal.pipeline.layout<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>

// CHECK: #config = #iree_cpu.lowering_config<vector_common_parallel = [1, 32]>
// CHECK: #config1 = #iree_cpu.lowering_config<distribution = [16, 16, 0], vector_common_parallel = [8, 32, 0], vector_reduction = [0, 0, 8]>
// CHECK: #translation = #iree_codegen.translation_info<pipeline = CPUDoubleTilingExpert>
// CHECK-LABEL: func.func @matmul_dispatch()
// CHECK-SAME: translation_info = #translation
// CHECK: linalg.fill {lowering_config = #config}
// CHECK: linalg.matmul {lowering_config = #config1}

hal.executable private @matmul_root {
  hal.executable.variant public @embedded_elf_hexagon target(#target) {
    builtin.module {
      func.func @matmul_dispatch() {
        %c0 = arith.constant 0 : index
        %cst = arith.constant 0.0 : f32
        %lhs = hal.interface.binding.subspan layout(#layout) binding(0) alignment(64) offset(%c0) flags("ReadOnly|Indirect") : !iree_tensor_ext.dispatch.tensor<readonly:tensor<128x128xf32>>
        %rhs = hal.interface.binding.subspan layout(#layout) binding(1) alignment(64) offset(%c0) flags("ReadOnly|Indirect") : !iree_tensor_ext.dispatch.tensor<readonly:tensor<128x128xf32>>
        %out = hal.interface.binding.subspan layout(#layout) binding(2) alignment(64) offset(%c0) flags(Indirect) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<128x128xf32>>
        %lhs_t = iree_tensor_ext.dispatch.tensor.load %lhs, offsets = [0, 0], sizes = [128, 128], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<128x128xf32>> -> tensor<128x128xf32>
        %rhs_t = iree_tensor_ext.dispatch.tensor.load %rhs, offsets = [0, 0], sizes = [128, 128], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<128x128xf32>> -> tensor<128x128xf32>
        %empty = tensor.empty() : tensor<128x128xf32>
        %init = linalg.fill ins(%cst : f32) outs(%empty : tensor<128x128xf32>) -> tensor<128x128xf32>
        %result = linalg.matmul ins(%lhs_t, %rhs_t : tensor<128x128xf32>, tensor<128x128xf32>) outs(%init : tensor<128x128xf32>) -> tensor<128x128xf32>
        iree_tensor_ext.dispatch.tensor.store %result, %out, offsets = [0, 0], sizes = [128, 128], strides = [1, 1] : tensor<128x128xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<128x128xf32>>
        return
      }
    }
  }
}

// -----

#target = #hal.executable.target<"hexagon", "embedded-elf-hexagon", {cpu = "hexagonv79", cpu_features = "+hvxv79,+hvx-length128b", data_layout = "e-m:e-p:32:32:32-a:0-n16:32-i64:64:64-i32:32:32-i16:16:16-i1:8:8-f32:32:32-f64:64:64-v32:32:32-v64:64:64-v512:512:512-v1024:1024:1024-v2048:2048:2048", hexagon.version = "79", iree.encoding.resolver = #iree_hexagon.hexagon_encoding_resolver<>, link_embedded = false, max_stack_allocation_size = 16384 : i64, native_vector_size = 32 : i64, target_triple = "hexagon-unknown-unknown-elf"}>
#layout = #hal.pipeline.layout<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>

// CHECK: #config = #iree_cpu.lowering_config<vector_common_parallel = [1, 1, 32]>
// CHECK: #config1 = #iree_cpu.lowering_config<distribution = [1, 16, 16, 0], vector_common_parallel = [1, 8, 32, 0], vector_reduction = [0, 0, 0, 8]>
// CHECK: #translation = #iree_codegen.translation_info<pipeline = CPUDoubleTilingExpert>
// CHECK-LABEL: func.func @batch_matmul_dispatch()
// CHECK-SAME: translation_info = #translation
// CHECK: linalg.fill {lowering_config = #config}
// CHECK: linalg.batch_matmul {lowering_config = #config1}

hal.executable private @batch_matmul_root {
  hal.executable.variant public @embedded_elf_hexagon target(#target) {
    builtin.module {
      func.func @batch_matmul_dispatch() {
        %c0 = arith.constant 0 : index
        %cst = arith.constant 0.0 : f32
        %lhs = hal.interface.binding.subspan layout(#layout) binding(0) alignment(64) offset(%c0) flags("ReadOnly|Indirect") : !iree_tensor_ext.dispatch.tensor<readonly:tensor<4x128x128xf32>>
        %rhs = hal.interface.binding.subspan layout(#layout) binding(1) alignment(64) offset(%c0) flags("ReadOnly|Indirect") : !iree_tensor_ext.dispatch.tensor<readonly:tensor<4x128x128xf32>>
        %out = hal.interface.binding.subspan layout(#layout) binding(2) alignment(64) offset(%c0) flags(Indirect) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<4x128x128xf32>>
        %lhs_t = iree_tensor_ext.dispatch.tensor.load %lhs, offsets = [0, 0, 0], sizes = [4, 128, 128], strides = [1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<4x128x128xf32>> -> tensor<4x128x128xf32>
        %rhs_t = iree_tensor_ext.dispatch.tensor.load %rhs, offsets = [0, 0, 0], sizes = [4, 128, 128], strides = [1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<4x128x128xf32>> -> tensor<4x128x128xf32>
        %empty = tensor.empty() : tensor<4x128x128xf32>
        %init = linalg.fill ins(%cst : f32) outs(%empty : tensor<4x128x128xf32>) -> tensor<4x128x128xf32>
        %result = linalg.batch_matmul ins(%lhs_t, %rhs_t : tensor<4x128x128xf32>, tensor<4x128x128xf32>) outs(%init : tensor<4x128x128xf32>) -> tensor<4x128x128xf32>
        iree_tensor_ext.dispatch.tensor.store %result, %out, offsets = [0, 0, 0], sizes = [4, 128, 128], strides = [1, 1, 1] : tensor<4x128x128xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<4x128x128xf32>>
        return
      }
    }
  }
}

// -----

#target = #hal.executable.target<"hexagon", "embedded-elf-hexagon", {cpu = "hexagonv79", cpu_features = "+hvxv79,+hvx-length128b", data_layout = "e-m:e-p:32:32:32-a:0-n16:32-i64:64:64-i32:32:32-i16:16:16-i1:8:8-f32:32:32-f64:64:64-v32:32:32-v64:64:64-v512:512:512-v1024:1024:1024-v2048:2048:2048", hexagon.version = "79", iree.encoding.resolver = #iree_hexagon.hexagon_encoding_resolver<>, link_embedded = false, max_stack_allocation_size = 16384 : i64, native_vector_size = 32 : i64, target_triple = "hexagon-unknown-unknown-elf"}>
#layout = #hal.pipeline.layout<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>

// CHECK: #config = #iree_cpu.lowering_config<vector_common_parallel = [1, 32]>
// CHECK: #config1 = #iree_cpu.lowering_config<distribution = [1, 16, 0], vector_common_parallel = [1, 32, 0], vector_reduction = [0, 0, 8]>
// CHECK: #translation = #iree_codegen.translation_info<pipeline = CPUDoubleTilingExpert>
// CHECK-LABEL: func.func @generic_dispatch()
// CHECK-SAME: translation_info = #translation
// CHECK: linalg.fill {lowering_config = #config}
// CHECK: lowering_config = #config1
// CHECK: lowering_config = #config

hal.executable private @generic_root {
  hal.executable.variant public @embedded_elf_hexagon target(#target) {
    builtin.module {
      func.func @generic_dispatch() {
        %c0 = arith.constant 0 : index
        %cst = arith.constant 0.0 : f32
        %src = hal.interface.binding.subspan layout(#layout) binding(0) alignment(64) offset(%c0) flags("ReadOnly|Indirect") : !iree_tensor_ext.dispatch.tensor<readonly:tensor<4x128x128xf32>>
        %out = hal.interface.binding.subspan layout(#layout) binding(1) alignment(64) offset(%c0) flags(Indirect) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<4x128xf32>>
        %src_t = iree_tensor_ext.dispatch.tensor.load %src, offsets = [0, 0, 0], sizes = [4, 128, 128], strides = [1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<4x128x128xf32>> -> tensor<4x128x128xf32>
        %red_empty = tensor.empty() : tensor<4x128xf32>
        %red_init = linalg.fill ins(%cst : f32) outs(%red_empty : tensor<4x128xf32>) -> tensor<4x128xf32>
        %reduced = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%src_t : tensor<4x128x128xf32>) outs(%red_init : tensor<4x128xf32>) {
        ^bb0(%in: f32, %out0: f32):
          %sum = arith.addf %in, %out0 : f32
          linalg.yield %sum : f32
        } -> tensor<4x128xf32>
        %ew_empty = tensor.empty() : tensor<4x128xf32>
        %ew_init = linalg.fill ins(%cst : f32) outs(%ew_empty : tensor<4x128xf32>) -> tensor<4x128xf32>
        %result = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%reduced : tensor<4x128xf32>) outs(%ew_init : tensor<4x128xf32>) {
        ^bb0(%in: f32, %out0: f32):
          %sum = arith.addf %in, %out0 : f32
          linalg.yield %sum : f32
        } -> tensor<4x128xf32>
        iree_tensor_ext.dispatch.tensor.store %result, %out, offsets = [0, 0], sizes = [4, 128], strides = [1, 1] : tensor<4x128xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<4x128xf32>>
        return
      }
    }
  }
}

// -----

#target = #hal.executable.target<"hexagon", "embedded-elf-hexagon", {cpu = "hexagonv79", cpu_features = "+hvxv79,+hvx-length128b", data_layout = "e-m:e-p:32:32:32-a:0-n16:32-i64:64:64-i32:32:32-i16:16:16-i1:8:8-f32:32:32-f64:64:64-v32:32:32-v64:64:64-v512:512:512-v1024:1024:1024-v2048:2048:2048", hexagon.version = "79", iree.encoding.resolver = #iree_hexagon.hexagon_encoding_resolver<>, link_embedded = false, max_stack_allocation_size = 16384 : i64, native_vector_size = 32 : i64, target_triple = "hexagon-unknown-unknown-elf"}>
#layout = #hal.pipeline.layout<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>

// CHECK: #translation = #iree_codegen.translation_info<pipeline = CPUDefault>
// CHECK-LABEL: func.func @fallback_dispatch()
// CHECK-SAME: translation_info = #translation
// CHECK: tensor.pad
// CHECK: lowering_config = #config

hal.executable private @fallback_root {
  hal.executable.variant public @embedded_elf_hexagon target(#target) {
    builtin.module {
      func.func @fallback_dispatch() {
        %c0 = arith.constant 0 : index
        %src = hal.interface.binding.subspan layout(#layout) binding(0) alignment(64) offset(%c0) flags("ReadOnly|Indirect") : !iree_tensor_ext.dispatch.tensor<readonly:tensor<96x96xf32>>
        %out = hal.interface.binding.subspan layout(#layout) binding(1) alignment(64) offset(%c0) flags(Indirect) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<98x98xf32>>
        %src_t = iree_tensor_ext.dispatch.tensor.load %src, offsets = [0, 0], sizes = [96, 96], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<96x96xf32>> -> tensor<96x96xf32>
        %padded = tensor.pad %src_t low[1, 1] high[1, 1] {
        ^bb0(%arg0: index, %arg1: index):
          %cst = arith.constant 0.0 : f32
          tensor.yield %cst : f32
        } : tensor<96x96xf32> to tensor<98x98xf32>
        iree_tensor_ext.dispatch.tensor.store %padded, %out, offsets = [0, 0], sizes = [98, 98], strides = [1, 1] : tensor<98x98xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<98x98xf32>>
        return
      }
    }
  }
}

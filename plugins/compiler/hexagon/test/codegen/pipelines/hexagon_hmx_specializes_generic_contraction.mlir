// Checks that configuration specializes an f16 generic contraction with f32
// accumulation and an elementwise epilogue to an HMX matmul, and that lowering
// emits HMX packing and MMA operations without oversized vector contracts.

// RUN: iree-opt \
// RUN:   --iree-hexagon-enable-vtcm-tiling --iree-hexagon-enable-hmx-matmul \
// RUN:   --pass-pipeline='builtin.module(hal.executable(hal.executable.variant(iree-hexagon-configuration-pipeline)))' \
// RUN:   %s | FileCheck %s --check-prefix=CONFIG

// RUN: iree-opt \
// RUN:   --iree-hexagon-enable-vtcm-tiling --iree-hexagon-enable-hmx-matmul \
// RUN:   --pass-pipeline='builtin.module(hal.executable(hal.executable.variant(iree-hexagon-configuration-pipeline, builtin.module(func.func(iree-hexagon-lower-executable-target)))))' \
// RUN:   %s | FileCheck %s --check-prefix=LOWERED

// K stays whole in the VTCM stage and untiled in the vector stage: HMX packing
// consumes it.
// CONFIG-DAG: #[[ROOT:.+]] = #iree_hexagon.lowering_config<distribution = [0, 0, 0], vector_common_parallel = [32, 32, 0], vtcm = [128, 128, 1536]>
// CONFIG-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = #iree_hexagon.pipeline<HmxMatmulExpert>>
// CONFIG: func.func @patch_embedding_dispatch() attributes {translation_info = #[[TRANSLATION]]}
// CONFIG-NOT: "reduction"
// CONFIG: linalg.matmul
// CONFIG-SAME: lowering_config = #[[ROOT]]
// CONFIG: linalg.generic
// CONFIG-SAME: iterator_types = ["parallel", "parallel"]

// LOWERED: func.func @patch_embedding_dispatch()
// LOWERED-NOT: vector<32x1536x
// LOWERED: iree_hexagon.hmx.pack
// LOWERED: iree_hexagon.hmx.mma
// LOWERED-NOT: vector<32x1536x
// LOWERED: return

hal.executable public @patch_embedding {
  hal.executable.variant public @embedded_elf_hexagon target(<"hexagon", "embedded-elf-hexagon", {cpu = "hexagonv79", cpu_features = "+hvxv79,+hvx-length128b", data_layout = "e-m:e-p:32:32:32-a:0-n16:32-i64:64:64-i32:32:32-i16:16:16-i1:8:8-f32:32:32-f64:64:64-v32:32:32-v64:64:64-v512:512:512-v1024:1024:1024-v2048:2048:2048", hexagon.version = "79", max_stack_allocation_size = 16384 : i64, native_vector_size = 128 : i64, target_triple = "hexagon-unknown-unknown-elf"}>) {
    hal.executable.export public @patch_embedding_dispatch ordinal(0) layout(#hal.pipeline.layout<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, ReadOnly>, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) count(%arg0: !hal.device) -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice()
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @patch_embedding_dispatch() {
        %cst = arith.constant 0.000000e+00 : f32
        %c393216 = arith.constant 393216 : index
        %c890880 = arith.constant 890880 : index
        %c589824 = arith.constant 589824 : index
        %c0 = arith.constant 0 : index
        %c1179648 = arith.constant 1179648 : index
        %0 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, ReadOnly>, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(0) alignment(128) offset(%c393216) flags("ReadOnly|Indirect") : !iree_tensor_ext.dispatch.tensor<readonly:tensor<256x1536xf16>>
        %1 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, ReadOnly>, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(1) alignment(128) offset(%c890880) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1024x1536xf16>>
        %2 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, ReadOnly>, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(1) alignment(128) offset(%c589824) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1024xf16>>
        %3 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, ReadOnly>, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(1) alignment(128) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<256x1024xf16>>
        %4 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, ReadOnly>, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(2) alignment(128) offset(%c1179648) flags(Indirect) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<256x1024xf16>>
        %5 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0], sizes = [256, 1536], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<256x1536xf16>> -> tensor<256x1536xf16>
        %6 = iree_tensor_ext.dispatch.tensor.load %1, offsets = [0, 0], sizes = [1024, 1536], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1024x1536xf16>> -> tensor<1024x1536xf16>
        %7 = iree_tensor_ext.dispatch.tensor.load %2, offsets = [0], sizes = [1024], strides = [1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1024xf16>> -> tensor<1024xf16>
        %8 = iree_tensor_ext.dispatch.tensor.load %3, offsets = [0, 0], sizes = [256, 1024], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<256x1024xf16>> -> tensor<256x1024xf16>
        %9 = tensor.empty() : tensor<256x1024xf16>
        %10 = tensor.empty() : tensor<256x1024xf32>
        %11 = linalg.fill ins(%cst : f32) outs(%10 : tensor<256x1024xf32>) -> tensor<256x1024xf32>
        %12 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%5, %6 : tensor<256x1536xf16>, tensor<1024x1536xf16>) outs(%11 : tensor<256x1024xf32>) {
        ^bb0(%in: f16, %in_0: f16, %out: f32):
          %14 = arith.extf %in : f16 to f32
          %15 = arith.extf %in_0 : f16 to f32
          %16 = arith.mulf %14, %15 : f32
          %17 = arith.addf %out, %16 : f32
          linalg.yield %17 : f32
        } -> tensor<256x1024xf32>
        %13 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%12, %7, %8 : tensor<256x1024xf32>, tensor<1024xf16>, tensor<256x1024xf16>) outs(%9 : tensor<256x1024xf16>) {
        ^bb0(%in: f32, %in_0: f16, %in_1: f16, %out: f16):
          %14 = arith.extf %in_0 : f16 to f32
          %15 = arith.addf %in, %14 : f32
          %16 = arith.truncf %15 : f32 to f16
          %17 = arith.addf %16, %in_1 : f16
          linalg.yield %17 : f16
        } -> tensor<256x1024xf16>
        iree_tensor_ext.dispatch.tensor.store %13, %4, offsets = [0, 0], sizes = [256, 1024], strides = [1, 1] : tensor<256x1024xf16> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<256x1024xf16>>
        return
      }
    }
  }
}

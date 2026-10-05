// The HMX pipeline must keep an independently tiled epilogue's lowering_config.
//
// An M=1 matmul (Gemma-4 drafter GeGLU gate) with a fused tanh-GELU epilogue
// on a unit dim. The planner tiles the epilogue independently of the 2048-wide
// VTCM tile. linalg-fold-unit-extent-dims used to run around VTCM tiling in
// this pipeline and rebuilt the generic without its lowering_config, so the
// epilogue was vectorized over the whole VTCM tile (vector<2048xf32>), which
// produced an 86 KB stack frame of register spills.

// RUN: iree-opt \
// RUN:   --iree-hexagon-enable-vtcm-tiling --iree-hexagon-enable-hmx-matmul \
// RUN:   --pass-pipeline='builtin.module(hal.executable(hal.executable.variant(builtin.module(func.func(iree-hexagon-lower-executable-target)))))' \
// RUN:   --mlir-print-ir-after=iree-hexagon-vtcm-tiling \
// RUN:   --mlir-print-ir-after=iree-codegen-generic-vectorization \
// RUN:   %s -o /dev/null 2>&1 | FileCheck %s

// CHECK-LABEL: IR Dump After HexagonVTCMTilingPass
// CHECK: func.func @hmx_m1_epilogue_dispatch
// CHECK: linalg.generic
// CHECK-SAME: lowering_config = #iree_cpu.lowering_config<vector_common_parallel = [1, 32]>

// CHECK-LABEL: IR Dump After GenericVectorizationPass
// CHECK: func.func @hmx_m1_epilogue_dispatch
// CHECK-NOT: vector<{{[0-9x]*}}2048x
// CHECK-NOT: vector<{{[0-9x]*}}2048xf
// CHECK: math.tanh {{.*}} : vector<1x32xf16>
// CHECK: return

#executable_target_embedded_elf_hexagon = #hal.executable.target<"hexagon", "embedded-elf-hexagon", {cpu = "hexagonv79", cpu_features = "+hvxv79,+hvx-length128b", data_layout = "e-m:e-p:32:32:32-a:0-n16:32-i64:64:64-i32:32:32-i16:16:16-i1:8:8-f32:32:32-f64:64:64-v32:32:32-v64:64:64-v512:512:512-v1024:1024:1024-v2048:2048:2048", hexagon.version = "79", iree.encoding.resolver = #iree_hexagon.hexagon_encoding_resolver<>, max_stack_allocation_size = 16384 : i64, native_vector_size = 32 : i64, target_triple = "hexagon-unknown-unknown-elf"}>
#pipeline_layout = #hal.pipeline.layout<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>
#config_fill = #iree_cpu.lowering_config<vector_common_parallel = [32, 32]>
#config_matmul = #iree_cpu.lowering_config<distribution = [0, 0, 0], vector_common_parallel = [32, 32, 0]>
#config_epilogue = #iree_cpu.lowering_config<vector_common_parallel = [1, 32]>
#vtcm = #iree_hexagon.vtcm_tiling_config<tile_sizes = [1, 2048, 256]>
#translation = #iree_codegen.translation_info<pipeline = #iree_hexagon.pipeline<HmxMatmulExpert>>

hal.executable private @test {
  hal.executable.variant public @embedded_elf_hexagon target(#executable_target_embedded_elf_hexagon) {
    builtin.module {
      func.func @hmx_m1_epilogue_dispatch() attributes {translation_info = #translation} {
        %c3_i64 = arith.constant 3 : i64
        %half = arith.constant 5.000000e-01 : f16
        %one = arith.constant 1.000000e+00 : f16
        %k0 = arith.constant 7.978520e-01 : f16
        %k1 = arith.constant 4.470830e-02 : f16
        %zero = arith.constant 0.000000e+00 : f32
        %c0 = arith.constant 0 : index
        %c512 = arith.constant 512 : index
        %c1049088 = arith.constant 1049088 : index
        %x_b = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(128) offset(%c0) flags("ReadOnly|Indirect") : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x256xf16>>
        %w_b = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(128) offset(%c512) flags("ReadOnly|Indirect") : !iree_tensor_ext.dispatch.tensor<readonly:tensor<2048x256xf16>>
        %up_b = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(128) offset(%c1049088) flags("ReadOnly|Indirect") : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x2048xf16>>
        %out_b = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(128) offset(%c0) flags(Indirect) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1x2048xf16>>
        %x = iree_tensor_ext.dispatch.tensor.load %x_b, offsets = [0, 0], sizes = [1, 256], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x256xf16>> -> tensor<1x256xf16>
        %w = iree_tensor_ext.dispatch.tensor.load %w_b, offsets = [0, 0], sizes = [2048, 256], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<2048x256xf16>> -> tensor<2048x256xf16>
        %up = iree_tensor_ext.dispatch.tensor.load %up_b, offsets = [0, 0], sizes = [1, 2048], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x2048xf16>> -> tensor<1x2048xf16>
        %empty_f16 = tensor.empty() : tensor<1x2048xf16>
        %empty_f32 = tensor.empty() : tensor<1x2048xf32>
        %init = linalg.fill {lowering_config = #config_fill} ins(%zero : f32) outs(%empty_f32 : tensor<1x2048xf32>) -> tensor<1x2048xf32>
        %mm = linalg.matmul
            indexing_maps = [
              affine_map<(d0, d1, d2) -> (d0, d2)>,
              affine_map<(d0, d1, d2) -> (d1, d2)>,
              affine_map<(d0, d1, d2) -> (d0, d1)>]
            {hexagon_vtcm_tiling_config = #vtcm, lowering_config = #config_matmul}
            ins(%x, %w : tensor<1x256xf16>, tensor<2048x256xf16>)
            outs(%init : tensor<1x2048xf32>) -> tensor<1x2048xf32>
        %res = linalg.generic {
            indexing_maps = [
              affine_map<(d0, d1) -> (d0, d1)>,
              affine_map<(d0, d1) -> (d0, d1)>,
              affine_map<(d0, d1) -> (d0, d1)>],
            iterator_types = ["parallel", "parallel"]}
            ins(%mm, %up : tensor<1x2048xf32>, tensor<1x2048xf16>)
            outs(%empty_f16 : tensor<1x2048xf16>)
            attrs = {lowering_config = #config_epilogue} {
        ^bb0(%acc: f32, %u: f16, %out: f16):
          %h = arith.truncf %acc : f32 to f16
          %h3 = math.fpowi %h, %c3_i64 : f16, i64
          %t0 = arith.mulf %h3, %k1 : f16
          %t1 = arith.addf %h, %t0 : f16
          %t2 = arith.mulf %t1, %k0 : f16
          %t3 = math.tanh %t2 : f16
          %t4 = arith.addf %t3, %one : f16
          %t5 = arith.mulf %t4, %half : f16
          %g = arith.mulf %h, %t5 : f16
          %gated = arith.mulf %g, %u : f16
          linalg.yield %gated : f16
        } -> tensor<1x2048xf16>
        iree_tensor_ext.dispatch.tensor.store %res, %out_b, offsets = [0, 0], sizes = [1, 2048], strides = [1, 1] : tensor<1x2048xf16> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1x2048xf16>>
        return
      }
    }
  }
}

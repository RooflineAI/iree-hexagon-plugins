// Copyright 2026 RooflineAI GmbH
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Derived from the local hexagon_lower_executable_target.mlir default case.
// RUN: iree-opt --iree-hexagon-check-linalg-vectorization=true --iree-hexagon-enable-vtcm-tiling=false --pass-pipeline='builtin.module(hal.executable(hal.executable.variant(iree-hexagon-translation-pipeline)))' --mlir-print-ir-after=iree-hexagon-emit-vectorization-remarks,iree-hexagon-check-ir-before-llvm-conversion,iree-hexagon-synchronize-symbol-visibility %s 2>&1 | FileCheck %s

// CHECK-LABEL: IR Dump After HexagonEmitVectorizationRemarksPass
// CHECK: func.func @default_dispatch
// CHECK-LABEL: IR Dump After HexagonCheckIRBeforeLLVMConversionPass
// CHECK: func.func @default_dispatch
// CHECK-LABEL: IR Dump After HexagonSynchronizeSymbolVisibilityPass
// CHECK: llvm.func @default_dispatch

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

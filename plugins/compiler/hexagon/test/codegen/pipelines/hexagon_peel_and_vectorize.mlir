// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Adapted from Codegen/LLVMCPU/test/pipeline_peel_and_vectorize_tests.mlir
// at IREE revision a45adeaa6115e446c898e6eb21fb6edc0e65ddc4.
// Uses the Hexagon pipeline/target and checks the peeling and vectorization stages.
// The AArch64 scalable case is omitted; a local peeling-off control is added.
// RUN: iree-opt --pass-pipeline='builtin.module(func.func(iree-hexagon-lower-executable-target))' --iree-hexagon-enable-vtcm-tiling=false --iree-hexagon-enable-hmx-matmul=false --mlir-print-ir-before=iree-hexagon-lower-executable-target --mlir-print-ir-after=iree-hexagon-peel,iree-codegen-generic-vectorization --split-input-file %s -o /dev/null 2>&1 | FileCheck %s

// CHECK-LABEL: IR Dump After HexagonPeelPass
// CHECK: func.func @no_peel_static_matmul(
// CHECK: scf.forall
// CHECK-COUNT-3: scf.for {{.*}}
// CHECK: linalg.matmul
// CHECK-SAME: -> tensor<8x32xf32>
// CHECK-NOT: scf.for {{.*}}
// CHECK: return
// CHECK-LABEL: IR Dump After GenericVectorizationPass
// CHECK: func.func @no_peel_static_matmul(
// CHECK: vector.contract

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
#config = #iree_hexagon.lowering_config<distribution = [64, 64, 0], vector_common_parallel = [8, 32, 0], vector_reduction = [0, 0, 16]>
#translation = #iree_codegen.translation_info<pipeline = #iree_hexagon.pipeline<MultiTilingExpert>, {enable_loop_peeling = true}>
#executable_target_embedded_elf_hexagon = #hal.executable.target<"hexagon", "embedded-elf-hexagon", {cpu = "hexagonv79", cpu_features = "+hvxv79,+hvx-length128b", data_layout = "e-m:e-p:32:32:32-a:0-n16:32-i64:64:64-i32:32:32-i16:16:16-i1:8:8-f32:32:32-f64:64:64-v32:32:32-v64:64:64-v512:512:512-v1024:1024:1024-v2048:2048:2048", hexagon.version = "79", iree.encoding.resolver = #iree_hexagon.hexagon_encoding_resolver<>, max_stack_allocation_size = 16384 : i64, native_vector_size = 128 : i64, target_triple = "hexagon-unknown-unknown-elf"}>
func.func @no_peel_static_matmul() attributes {hal.executable.target = #executable_target_embedded_elf_hexagon, translation_info = #translation} {
  %cst = arith.constant 0.000000e+00 : f32
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<128x64xf32>>
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<64x512xf32>>
  %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<128x512xf32>>
  %3 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0], sizes = [128, 64], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<128x64xf32>> -> tensor<128x64xf32>
  %4 = iree_tensor_ext.dispatch.tensor.load %1, offsets = [0, 0], sizes = [64, 512], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<64x512xf32>> -> tensor<64x512xf32>
  %5 = tensor.empty() : tensor<128x512xf32>
  %6 = linalg.fill ins(%cst : f32) outs(%5 : tensor<128x512xf32>) -> tensor<128x512xf32>
  %7 = linalg.matmul {lowering_config = #config} ins(%3, %4 : tensor<128x64xf32>, tensor<64x512xf32>) outs(%6 : tensor<128x512xf32>) -> tensor<128x512xf32>
  iree_tensor_ext.dispatch.tensor.store %7, %2, offsets = [0, 0], sizes = [128, 512], strides = [1, 1] : tensor<128x512xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<128x512xf32>>
  return
}

// -----

// CHECK-LABEL: IR Dump After HexagonPeelPass
// CHECK: func.func @peel_static_matmul(
// CHECK: scf.forall
// CHECK: linalg.matmul
// CHECK-SAME: -> tensor<8x32xf32>
// CHECK: linalg.matmul
// CHECK-SAME: -> tensor<8x32xf32>
// CHECK: linalg.matmul
// CHECK-SAME: -> tensor<8x?xf32>
// CHECK: linalg.matmul
// CHECK-SAME: -> tensor<?x?xf32>
// CHECK-LABEL: IR Dump After GenericVectorizationPass
// CHECK: func.func @peel_static_matmul(
// CHECK: vector.contract

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
#config = #iree_hexagon.lowering_config<distribution = [65, 65, 0], vector_common_parallel = [8, 32, 0], vector_reduction = [0, 0, 16]>
#translation = #iree_codegen.translation_info<pipeline = #iree_hexagon.pipeline<MultiTilingExpert>, {enable_loop_peeling = true}>
#executable_target_embedded_elf_hexagon = #hal.executable.target<"hexagon", "embedded-elf-hexagon", {cpu = "hexagonv79", cpu_features = "+hvxv79,+hvx-length128b", data_layout = "e-m:e-p:32:32:32-a:0-n16:32-i64:64:64-i32:32:32-i16:16:16-i1:8:8-f32:32:32-f64:64:64-v32:32:32-v64:64:64-v512:512:512-v1024:1024:1024-v2048:2048:2048", hexagon.version = "79", iree.encoding.resolver = #iree_hexagon.hexagon_encoding_resolver<>, max_stack_allocation_size = 16384 : i64, native_vector_size = 128 : i64, target_triple = "hexagon-unknown-unknown-elf"}>
func.func @peel_static_matmul() attributes {hal.executable.target = #executable_target_embedded_elf_hexagon, translation_info = #translation} {
  %cst = arith.constant 0.000000e+00 : f32
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<128x49xf32>>
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<49x512xf32>>
  %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<128x512xf32>>
  %3 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0], sizes = [128, 49], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<128x49xf32>> -> tensor<128x49xf32>
  %4 = iree_tensor_ext.dispatch.tensor.load %1, offsets = [0, 0], sizes = [49, 512], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<49x512xf32>> -> tensor<49x512xf32>
  %5 = tensor.empty() : tensor<128x512xf32>
  %6 = linalg.fill ins(%cst : f32) outs(%5 : tensor<128x512xf32>) -> tensor<128x512xf32>
  %7 = linalg.matmul {lowering_config = #config} ins(%3, %4 : tensor<128x49xf32>, tensor<49x512xf32>) outs(%6 : tensor<128x512xf32>) -> tensor<128x512xf32>
  iree_tensor_ext.dispatch.tensor.store %7, %2, offsets = [0, 0], sizes = [128, 512], strides = [1, 1] : tensor<128x512xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<128x512xf32>>
  return
}

// -----

// CHECK-LABEL: IR Dump After HexagonPeelPass
// CHECK: func.func @peel_dynamic_matmul(
// CHECK: scf.forall
// CHECK: linalg.matmul
// CHECK-SAME: -> tensor<8x32xf32>
// CHECK: linalg.matmul
// CHECK-SAME: -> tensor<8x32xf32>
// CHECK: linalg.matmul
// CHECK-SAME: -> tensor<8x?xf32>
// CHECK: linalg.matmul
// CHECK-SAME: -> tensor<?x?xf32>
// CHECK-LABEL: IR Dump After GenericVectorizationPass
// CHECK: func.func @peel_dynamic_matmul(
// CHECK: vector.contract

#pipeline_layout = #hal.pipeline.layout<constants = 3, bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
#config = #iree_hexagon.lowering_config<distribution = [64, 64, 0], vector_common_parallel = [8, 32, 0], vector_reduction = [0, 0, 16]>
#translation = #iree_codegen.translation_info<pipeline = #iree_hexagon.pipeline<MultiTilingExpert>, {enable_loop_peeling = true}>
#executable_target_embedded_elf_hexagon = #hal.executable.target<"hexagon", "embedded-elf-hexagon", {cpu = "hexagonv79", cpu_features = "+hvxv79,+hvx-length128b", data_layout = "e-m:e-p:32:32:32-a:0-n16:32-i64:64:64-i32:32:32-i16:16:16-i1:8:8-f32:32:32-f64:64:64-v32:32:32-v64:64:64-v512:512:512-v1024:1024:1024-v2048:2048:2048", hexagon.version = "79", iree.encoding.resolver = #iree_hexagon.hexagon_encoding_resolver<>, max_stack_allocation_size = 16384 : i64, native_vector_size = 128 : i64, target_triple = "hexagon-unknown-unknown-elf"}>
func.func @peel_dynamic_matmul() attributes {hal.executable.target = #executable_target_embedded_elf_hexagon, translation_info = #translation} {
  %cst = arith.constant 0.000000e+00 : f32
  %0 = hal.interface.constant.load layout(#pipeline_layout) ordinal(0) : i32
  %1 = hal.interface.constant.load layout(#pipeline_layout) ordinal(1) : i32
  %2 = hal.interface.constant.load layout(#pipeline_layout) ordinal(2) : i32
  %3 = arith.index_cast %0 : i32 to index
  %4 = arith.index_cast %1 : i32 to index
  %5 = arith.index_cast %2 : i32 to index
  %dim0 = iree_tensor_ext.dispatch.workload.ordinal %3, 0 : index
  %dim1 = iree_tensor_ext.dispatch.workload.ordinal %4, 1 : index
  %dim2 = iree_tensor_ext.dispatch.workload.ordinal %5, 2 : index
  %6 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x?xf32>>{%dim1, %dim0}
  %7 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x?xf32>>{%dim0, %dim2}
  %8 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?x?xf32>>{%dim1, %dim2}
  %9 = iree_tensor_ext.dispatch.tensor.load %6, offsets = [0, 0], sizes = [%dim1, %dim0], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x?xf32>>{%dim1, %dim0} -> tensor<?x?xf32>
  %10 = iree_tensor_ext.dispatch.tensor.load %7, offsets = [0, 0], sizes = [%dim0, %dim2], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x?xf32>>{%dim0, %dim2} -> tensor<?x?xf32>
  %11 = tensor.empty(%dim1, %dim2) : tensor<?x?xf32>
  %12 = linalg.fill ins(%cst : f32) outs(%11 : tensor<?x?xf32>) -> tensor<?x?xf32>
  %13 = linalg.matmul {lowering_config = #config} ins(%9, %10 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%12 : tensor<?x?xf32>) -> tensor<?x?xf32>
  iree_tensor_ext.dispatch.tensor.store %13, %8, offsets = [0, 0], sizes = [%dim1, %dim2], strides = [1, 1] : tensor<?x?xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?x?xf32>>{%dim1, %dim2}
  return
}

// -----

// Same ragged input with peeling disabled: vectorization still runs, while the
// peeling factory is not selected.
// CHECK-LABEL: IR Dump Before HexagonLowerExecutableTargetPass
// CHECK: func.func @peeling_disabled(
// CHECK-NOT: IR Dump After HexagonPeelPass
// CHECK-LABEL: IR Dump After GenericVectorizationPass
// CHECK: func.func @peeling_disabled(
// CHECK: vector.contract

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
#config = #iree_hexagon.lowering_config<distribution = [65, 65, 0], vector_common_parallel = [8, 32, 0], vector_reduction = [0, 0, 16]>
#translation = #iree_codegen.translation_info<pipeline = #iree_hexagon.pipeline<MultiTilingExpert>>
#executable_target_embedded_elf_hexagon = #hal.executable.target<"hexagon", "embedded-elf-hexagon", {cpu = "hexagonv79", cpu_features = "+hvxv79,+hvx-length128b", data_layout = "e-m:e-p:32:32:32-a:0-n16:32-i64:64:64-i32:32:32-i16:16:16-i1:8:8-f32:32:32-f64:64:64-v32:32:32-v64:64:64-v512:512:512-v1024:1024:1024-v2048:2048:2048", hexagon.version = "79", iree.encoding.resolver = #iree_hexagon.hexagon_encoding_resolver<>, max_stack_allocation_size = 16384 : i64, native_vector_size = 128 : i64, target_triple = "hexagon-unknown-unknown-elf"}>
func.func @peeling_disabled() attributes {hal.executable.target = #executable_target_embedded_elf_hexagon, translation_info = #translation} {
  %cst = arith.constant 0.000000e+00 : f32
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<128x49xf32>>
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<49x512xf32>>
  %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<128x512xf32>>
  %3 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0], sizes = [128, 49], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<128x49xf32>> -> tensor<128x49xf32>
  %4 = iree_tensor_ext.dispatch.tensor.load %1, offsets = [0, 0], sizes = [49, 512], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<49x512xf32>> -> tensor<49x512xf32>
  %5 = tensor.empty() : tensor<128x512xf32>
  %6 = linalg.fill ins(%cst : f32) outs(%5 : tensor<128x512xf32>) -> tensor<128x512xf32>
  %7 = linalg.matmul {lowering_config = #config} ins(%3, %4 : tensor<128x49xf32>, tensor<49x512xf32>) outs(%6 : tensor<128x512xf32>) -> tensor<128x512xf32>
  iree_tensor_ext.dispatch.tensor.store %7, %2, offsets = [0, 0], sizes = [128, 512], strides = [1, 1] : tensor<128x512xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<128x512xf32>>
  return
}

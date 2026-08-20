// Test written to reflect current Hexagon selector policy.
//
// This file is not intended as a long-term reference of expected behavior,
// but it keeps track of the current state of the kernelDispatch.cpp file.
//
// RUN: iree-opt --iree-load-plugin=hexagon=$HEXAGON_COMPILER_PLUGIN \
// RUN:   --pass-pipeline='builtin.module(iree-hexagon-select-lowering-strategy)' \
// RUN:   --split-input-file %s | FileCheck %s
//
// Same selection, but generalize named ops first so that convolution/pooling
// selection is exercised on their linalg.generic form too (mirrors upstream
// IREE). inferOpLoweringPlan classifies by structure before op type, so a
// generalized conv/pool is still routed through inferConvPlan and receives the
// same root tiling as the named op. The GENERIC checks assert only the root op:
// the fused fill is generalized into a linalg.generic and tiled as one, so its
// config intentionally differs from the named form and is not asserted.
//
// RUN: iree-opt --iree-load-plugin=hexagon=$HEXAGON_COMPILER_PLUGIN \
// RUN:   --pass-pipeline='builtin.module(func.func(linalg-generalize-named-ops),iree-hexagon-select-lowering-strategy)' \
// RUN:   --split-input-file %s | FileCheck %s --check-prefix=GENERIC

#target = #hal.executable.target<"hexagon", "embedded-elf-hexagon", {cpu = "hexagonv79", cpu_features = "+hvxv79,+hvx-length128b", data_layout = "e-m:e-p:32:32:32-a:0-n16:32-i64:64:64-i32:32:32-i16:16:16-i1:8:8-f32:32:32-f64:64:64-v32:32:32-v64:64:64-v512:512:512-v1024:1024:1024-v2048:2048:2048", hexagon.version = "79", iree.encoding.resolver = #iree_hexagon.hexagon_encoding_resolver<>, link_embedded = false, max_stack_allocation_size = 16384 : i64, native_vector_size = 32 : i64, target_triple = "hexagon-unknown-unknown-elf"}>
func.func @matmul_dispatch(%lhs: tensor<128x128xf32>, %rhs: tensor<128x128xf32>) -> tensor<128x128xf32> attributes {hal.executable.target = #target} {
  %cst = arith.constant 0.0 : f32
  %empty = tensor.empty() : tensor<128x128xf32>
  %init = linalg.fill ins(%cst : f32) outs(%empty : tensor<128x128xf32>) -> tensor<128x128xf32>
  %result = linalg.matmul ins(%lhs, %rhs : tensor<128x128xf32>, tensor<128x128xf32>) outs(%init : tensor<128x128xf32>) -> tensor<128x128xf32>
  return %result : tensor<128x128xf32>
}
// CHECK-DAG: #[[FILL:.+]] = #iree_cpu.lowering_config<vector_common_parallel = [8, 32]>
// CHECK-DAG: #[[MATMUL:.+]] = #iree_cpu.lowering_config<cache_parallel = [64, 64, 0], distribution = [0, 0, 0], vector_common_parallel = [8, 32, 0], vector_reduction = [0, 0, 8]>
// CHECK-NOT: #iree_cpu.lowering_config
// CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = #iree_cpu.pipeline<DoubleTilingExpert>, {enable_loop_peeling}>
// CHECK: func.func @matmul_dispatch(
// CHECK-SAME: translation_info = #[[TRANSLATION]]
// CHECK: linalg.fill {lowering_config = #[[FILL]]}
// CHECK: linalg.matmul {lowering_config = #[[MATMUL]]}

// -----

// matmul with n=49 (not divisible by the HVX vector width of 32).
#target = #hal.executable.target<"hexagon", "embedded-elf-hexagon", {cpu = "hexagonv79", cpu_features = "+hvxv79,+hvx-length128b", data_layout = "e-m:e-p:32:32:32-a:0-n16:32-i64:64:64-i32:32:32-i16:16:16-i1:8:8-f32:32:32-f64:64:64-v32:32:32-v64:64:64-v512:512:512-v1024:1024:1024-v2048:2048:2048", hexagon.version = "79", iree.encoding.resolver = #iree_hexagon.hexagon_encoding_resolver<>, link_embedded = false, max_stack_allocation_size = 16384 : i64, native_vector_size = 32 : i64, target_triple = "hexagon-unknown-unknown-elf"}>
func.func @matmul_512x49x4608_dispatch(%lhs: tensor<512x4608xf32>, %rhs: tensor<4608x49xf32>) -> tensor<512x49xf32> attributes {hal.executable.target = #target} {
  %cst = arith.constant 0.0 : f32
  %empty = tensor.empty() : tensor<512x49xf32>
  %init = linalg.fill ins(%cst : f32) outs(%empty : tensor<512x49xf32>) -> tensor<512x49xf32>
  %result = linalg.matmul ins(%lhs, %rhs : tensor<512x4608xf32>, tensor<4608x49xf32>) outs(%init : tensor<512x49xf32>) -> tensor<512x49xf32>
  return %result : tensor<512x49xf32>
}
// CHECK-DAG: #[[FILL:.+]] = #iree_cpu.lowering_config<vector_common_parallel = [8, 32]>
// CHECK-DAG: #[[MATMUL:.+]] = #iree_cpu.lowering_config<cache_parallel = [64, 49, 0], distribution = [0, 0, 0], vector_common_parallel = [8, 32, 0], vector_reduction = [0, 0, 8]>
// CHECK-NOT: #iree_cpu.lowering_config
// CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = #iree_cpu.pipeline<DoubleTilingExpert>, {enable_loop_peeling}>
// CHECK: func.func @matmul_512x49x4608_dispatch(
// CHECK-SAME: translation_info = #[[TRANSLATION]]
// CHECK: linalg.fill {lowering_config = #[[FILL]]}
// CHECK: linalg.matmul {lowering_config = #[[MATMUL]]}

// -----

#target = #hal.executable.target<"hexagon", "embedded-elf-hexagon", {cpu = "hexagonv79", cpu_features = "+hvxv79,+hvx-length128b", data_layout = "e-m:e-p:32:32:32-a:0-n16:32-i64:64:64-i32:32:32-i16:16:16-i1:8:8-f32:32:32-f64:64:64-v32:32:32-v64:64:64-v512:512:512-v1024:1024:1024-v2048:2048:2048", hexagon.version = "79", iree.encoding.resolver = #iree_hexagon.hexagon_encoding_resolver<>, link_embedded = false, max_stack_allocation_size = 16384 : i64, native_vector_size = 32 : i64, target_triple = "hexagon-unknown-unknown-elf"}>
func.func @batch_matmul_dispatch(%lhs: tensor<4x128x128xf32>, %rhs: tensor<4x128x128xf32>) -> tensor<4x128x128xf32> attributes {hal.executable.target = #target} {
  %cst = arith.constant 0.0 : f32
  %empty = tensor.empty() : tensor<4x128x128xf32>
  %init = linalg.fill ins(%cst : f32) outs(%empty : tensor<4x128x128xf32>) -> tensor<4x128x128xf32>
  %result = linalg.batch_matmul ins(%lhs, %rhs : tensor<4x128x128xf32>, tensor<4x128x128xf32>) outs(%init : tensor<4x128x128xf32>) -> tensor<4x128x128xf32>
  return %result : tensor<4x128x128xf32>
}
// CHECK-DAG: #[[FILL:.+]] = #iree_cpu.lowering_config<vector_common_parallel = [1, 8, 32]>
// CHECK-DAG: #[[MATMUL:.+]] = #iree_cpu.lowering_config<cache_parallel = [1, 64, 64, 0], distribution = [0, 0, 0, 0], vector_common_parallel = [1, 8, 32, 0], vector_reduction = [0, 0, 0, 8]>
// CHECK-NOT: #iree_cpu.lowering_config
// CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = #iree_cpu.pipeline<DoubleTilingExpert>, {enable_loop_peeling}>
// CHECK: func.func @batch_matmul_dispatch(
// CHECK-SAME: translation_info = #[[TRANSLATION]]
// CHECK: linalg.fill {lowering_config = #[[FILL]]}
// CHECK: linalg.batch_matmul {lowering_config = #[[MATMUL]]}

// -----

#target = #hal.executable.target<"hexagon", "embedded-elf-hexagon", {cpu = "hexagonv79", cpu_features = "+hvxv79,+hvx-length128b", data_layout = "e-m:e-p:32:32:32-a:0-n16:32-i64:64:64-i32:32:32-i16:16:16-i1:8:8-f32:32:32-f64:64:64-v32:32:32-v64:64:64-v512:512:512-v1024:1024:1024-v2048:2048:2048", hexagon.version = "79", iree.encoding.resolver = #iree_hexagon.hexagon_encoding_resolver<>, link_embedded = false, max_stack_allocation_size = 16384 : i64, native_vector_size = 32 : i64, target_triple = "hexagon-unknown-unknown-elf"}>
func.func @generic_dispatch(%src: tensor<4x128x128xf32>) -> tensor<4x128xf32> attributes {hal.executable.target = #target} {
  %cst = arith.constant 0.0 : f32
  %red_empty = tensor.empty() : tensor<4x128xf32>
  %red_init = linalg.fill ins(%cst : f32) outs(%red_empty : tensor<4x128xf32>) -> tensor<4x128xf32>
  %reduced = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%src : tensor<4x128x128xf32>) outs(%red_init : tensor<4x128xf32>) {
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
  return %result : tensor<4x128xf32>
}
// CHECK-DAG: #[[FILL:.+]] = #iree_cpu.lowering_config<vector_common_parallel = [1, 1]>
// CHECK-DAG: #[[REDUCE:.+]] = #iree_cpu.lowering_config<cache_parallel = [4, 0, 0], distribution = [0, 0, 0], vector_common_parallel = [1, 1, 0], vector_reduction = [0, 0, 32]>
// CHECK-DAG: #[[ELEMWISE:.+]] = #iree_cpu.lowering_config<cache_parallel = [4, 0], vector_common_parallel = [1, 1]>
// CHECK-NOT: #iree_cpu.lowering_config
// CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = #iree_cpu.pipeline<DoubleTilingExpert>, {enable_loop_peeling}>
// CHECK: func.func @generic_dispatch(
// CHECK-SAME: translation_info = #[[TRANSLATION]]
// CHECK: linalg.fill {lowering_config = #[[FILL]]}
// CHECK: linalg.generic
// CHECK-SAME: lowering_config = #[[REDUCE]]
// CHECK: linalg.fill {lowering_config = #[[FILL]]}
// CHECK: linalg.generic
// CHECK-SAME: lowering_config = #[[ELEMWISE]]

// -----

#target = #hal.executable.target<"hexagon", "embedded-elf-hexagon", {cpu = "hexagonv79", cpu_features = "+hvxv79,+hvx-length128b", data_layout = "e-m:e-p:32:32:32-a:0-n16:32-i64:64:64-i32:32:32-i16:16:16-i1:8:8-f32:32:32-f64:64:64-v32:32:32-v64:64:64-v512:512:512-v1024:1024:1024-v2048:2048:2048", hexagon.version = "79", iree.encoding.resolver = #iree_hexagon.hexagon_encoding_resolver<>, link_embedded = false, max_stack_allocation_size = 16384 : i64, native_vector_size = 32 : i64, target_triple = "hexagon-unknown-unknown-elf"}>
func.func @fallback_dispatch(%src: tensor<96x96xf32>) -> tensor<98x98xf32> attributes {hal.executable.target = #target} {
  %padded = tensor.pad %src low[1, 1] high[1, 1] {
  ^bb0(%arg0: index, %arg1: index):
    %cst = arith.constant 0.0 : f32
    tensor.yield %cst : f32
  } : tensor<96x96xf32> to tensor<98x98xf32>
  return %padded : tensor<98x98xf32>
}
// CHECK-DAG: #[[PAD:.+]] = #iree_cpu.lowering_config<vector_common_parallel = [1, 32]>
// CHECK-NOT: #iree_cpu.lowering_config
// CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = #iree_cpu.pipeline<Default>>
// CHECK: func.func @fallback_dispatch(
// CHECK-SAME: translation_info = #[[TRANSLATION]]
// CHECK: tensor.pad
// CHECK: lowering_config = #[[PAD]]

// -----

#target = #hal.executable.target<"hexagon", "embedded-elf-hexagon", {cpu = "hexagonv79", cpu_features = "+hvxv79,+hvx-length128b", data_layout = "e-m:e-p:32:32:32-a:0-n16:32-i64:64:64-i32:32:32-i16:16:16-i1:8:8-f32:32:32-f64:64:64-v32:32:32-v64:64:64-v512:512:512-v1024:1024:1024-v2048:2048:2048", hexagon.version = "79", iree.encoding.resolver = #iree_hexagon.hexagon_encoding_resolver<>, link_embedded = false, max_stack_allocation_size = 16384 : i64, native_vector_size = 32 : i64, target_triple = "hexagon-unknown-unknown-elf"}>
func.func @fill_root_dispatch() -> tensor<64x128xf32> attributes {hal.executable.target = #target} {
  %cst = arith.constant 0.0 : f32
  %empty = tensor.empty() : tensor<64x128xf32>
  %filled = linalg.fill ins(%cst : f32) outs(%empty : tensor<64x128xf32>) -> tensor<64x128xf32>
  return %filled : tensor<64x128xf32>
}
// CHECK-DAG: #[[FILL:.+]] = #iree_cpu.lowering_config<distribution = [0, 0], vector_common_parallel = [1, 32]>
// CHECK-NOT: #iree_cpu.lowering_config
// CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = #iree_cpu.pipeline<DoubleTilingExpert>, {enable_loop_peeling}>
// CHECK: func.func @fill_root_dispatch(
// CHECK-SAME: translation_info = #[[TRANSLATION]]
// CHECK: linalg.fill {lowering_config = #[[FILL]]}

// -----

#target = #hal.executable.target<"hexagon", "embedded-elf-hexagon", {cpu = "hexagonv79", cpu_features = "+hvxv79,+hvx-length128b", data_layout = "e-m:e-p:32:32:32-a:0-n16:32-i64:64:64-i32:32:32-i16:16:16-i1:8:8-f32:32:32-f64:64:64-v32:32:32-v64:64:64-v512:512:512-v1024:1024:1024-v2048:2048:2048", hexagon.version = "79", iree.encoding.resolver = #iree_hexagon.hexagon_encoding_resolver<>, link_embedded = false, max_stack_allocation_size = 16384 : i64, native_vector_size = 32 : i64, target_triple = "hexagon-unknown-unknown-elf"}>
func.func @buffer_copy_root_dispatch(%src: memref<64x112x112xf32, strided<[12544, 112, 1], offset: ?>>, %dst: memref<64x114x114xf32, strided<[12996, 114, 1], offset: ?>>) attributes {hal.executable.target = #target} {
  %subview = memref.subview %dst[0, 1, 1] [64, 112, 112] [1, 1, 1] : memref<64x114x114xf32, strided<[12996, 114, 1], offset: ?>> to memref<64x112x112xf32, strided<[12996, 114, 1], offset: ?>>
  linalg.generic {
    indexing_maps = [
      affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
      affine_map<(d0, d1, d2) -> (d0, d1, d2)>
    ],
    iterator_types = ["parallel", "parallel", "parallel"]
  } ins(%src : memref<64x112x112xf32, strided<[12544, 112, 1], offset: ?>>)
    outs(%subview : memref<64x112x112xf32, strided<[12996, 114, 1], offset: ?>>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  }
  return
}
// CHECK-DAG: #[[COPY:.+]] = #iree_cpu.lowering_config<distribution = [1, 1, 32], vector_common_parallel = [1, 1, 32]>
// CHECK-NOT: #iree_cpu.lowering_config
// CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = #iree_cpu.pipeline<BufferOpsTileAndVectorize>>
// CHECK: func.func @buffer_copy_root_dispatch(
// CHECK-SAME: translation_info = #[[TRANSLATION]]
// CHECK: linalg.generic
// CHECK-SAME: lowering_config = #[[COPY]]

// -----

// Note that we still expect 32 as tile target despite it not being a divisor of the tensor shape
#target = #hal.executable.target<"hexagon", "embedded-elf-hexagon", {cpu = "hexagonv79", cpu_features = "+hvxv79,+hvx-length128b", data_layout = "e-m:e-p:32:32:32-a:0-n16:32-i64:64:64-i32:32:32-i16:16:16-i1:8:8-f32:32:32-f64:64:64-v32:32:32-v64:64:64-v512:512:512-v1024:1024:1024-v2048:2048:2048", hexagon.version = "79", iree.encoding.resolver = #iree_hexagon.hexagon_encoding_resolver<>, link_embedded = false, max_stack_allocation_size = 16384 : i64, native_vector_size = 32 : i64, target_triple = "hexagon-unknown-unknown-elf"}>
func.func @pooling_root_dispatch(%src: tensor<1x64x114x114xf32>) -> tensor<1x64x56x56xf32> attributes {hal.executable.target = #target} {
  %cst = arith.constant 0xFF800000 : f32
  %empty = tensor.empty() : tensor<1x64x56x56xf32>
  %kernel = tensor.empty() : tensor<3x3xf32>
  %init = linalg.fill ins(%cst : f32) outs(%empty : tensor<1x64x56x56xf32>) -> tensor<1x64x56x56xf32>
  %result = linalg.pooling_nchw_max {dilations = dense<1> : vector<2xi64>, strides = dense<2> : vector<2xi64>} ins(%src, %kernel : tensor<1x64x114x114xf32>, tensor<3x3xf32>) outs(%init : tensor<1x64x56x56xf32>) -> tensor<1x64x56x56xf32>
  return %result : tensor<1x64x56x56xf32>
}
// CHECK-DAG: #[[FILL:.+]] = #iree_cpu.lowering_config<vector_common_parallel = [1, 32, 1, 28]>
// CHECK-DAG: #[[POOL:.+]] = #iree_cpu.lowering_config<distribution = [1, 32, 1, 28, 0, 0], vector_common_parallel = [1, 32, 1, 28, 0, 0]>
// CHECK-NOT: #iree_cpu.lowering_config
// CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = #iree_cpu.pipeline<ConvTileAndDecomposeExpert>>
// CHECK: func.func @pooling_root_dispatch(
// CHECK-SAME: translation_info = #[[TRANSLATION]]
// CHECK: linalg.fill {lowering_config = #[[FILL]]}
// CHECK: linalg.pooling_nchw_max
// CHECK-SAME: lowering_config = #[[POOL]]
// GENERIC-DAG: #[[GPOOL:.+]] = #iree_cpu.lowering_config<distribution = [1, 32, 1, 28, 0, 0], vector_common_parallel = [1, 32, 1, 28, 0, 0]>
// GENERIC-DAG: #[[GTRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = #iree_cpu.pipeline<ConvTileAndDecomposeExpert>>
// GENERIC: func.func @pooling_root_dispatch(
// GENERIC-SAME: translation_info = #[[GTRANSLATION]]
// GENERIC: linalg.generic
// GENERIC: lowering_config = #[[GPOOL]]

// -----

#target = #hal.executable.target<"hexagon", "embedded-elf-hexagon", {cpu = "hexagonv79", cpu_features = "+hvxv79,+hvx-length128b", data_layout = "e-m:e-p:32:32:32-a:0-n16:32-i64:64:64-i32:32:32-i16:16:16-i1:8:8-f32:32:32-f64:64:64-v32:32:32-v64:64:64-v512:512:512-v1024:1024:1024-v2048:2048:2048", hexagon.version = "79", iree.encoding.resolver = #iree_hexagon.hexagon_encoding_resolver<>, link_embedded = false, max_stack_allocation_size = 16384 : i64, native_vector_size = 32 : i64, target_triple = "hexagon-unknown-unknown-elf"}>
func.func @transpose_root_dispatch(%src: tensor<64x128xf32>) -> tensor<128x64xf32> attributes {hal.executable.target = #target} {
  %empty = tensor.empty() : tensor<128x64xf32>
  %transposed = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d1, d0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%src : tensor<64x128xf32>) outs(%empty : tensor<128x64xf32>) {
  ^bb0(%in: f32, %out0: f32):
    linalg.yield %in : f32
  } -> tensor<128x64xf32>
  return %transposed : tensor<128x64xf32>
}
// CHECK-DAG: #[[TRANSPOSE:.+]] = #iree_cpu.lowering_config<cache_parallel = [64, 0], distribution = [0, 0], vector_common_parallel = [1, 32]>
// CHECK-NOT: #iree_cpu.lowering_config
// CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = #iree_cpu.pipeline<DoubleTilingExpert>, {enable_loop_peeling}>
// CHECK: func.func @transpose_root_dispatch(
// CHECK-SAME: translation_info = #[[TRANSLATION]]
// CHECK: linalg.generic
// CHECK-SAME: iterator_types = ["parallel", "parallel"]
// CHECK-SAME: lowering_config = #[[TRANSPOSE]]

// -----

#target = #hal.executable.target<"hexagon", "embedded-elf-hexagon", {cpu = "hexagonv79", cpu_features = "+hvxv79,+hvx-length128b", data_layout = "e-m:e-p:32:32:32-a:0-n16:32-i64:64:64-i32:32:32-i16:16:16-i1:8:8-f32:32:32-f64:64:64-v32:32:32-v64:64:64-v512:512:512-v1024:1024:1024-v2048:2048:2048", hexagon.version = "79", iree.encoding.resolver = #iree_hexagon.hexagon_encoding_resolver<>, link_embedded = false, max_stack_allocation_size = 16384 : i64, native_vector_size = 32 : i64, target_triple = "hexagon-unknown-unknown-elf"}>
func.func @matmul_epilogue_dispatch(%lhs: tensor<128x128xf32>, %rhs: tensor<128x128xf32>, %bias: tensor<128x128xf32>) -> tensor<128x128xf32> attributes {hal.executable.target = #target} {
  %cst = arith.constant 0.0 : f32
  %empty = tensor.empty() : tensor<128x128xf32>
  %init = linalg.fill ins(%cst : f32) outs(%empty : tensor<128x128xf32>) -> tensor<128x128xf32>
  %matmul = linalg.matmul ins(%lhs, %rhs : tensor<128x128xf32>, tensor<128x128xf32>) outs(%init : tensor<128x128xf32>) -> tensor<128x128xf32>
  %epilogue_empty = tensor.empty() : tensor<128x128xf32>
  %epilogue_init = linalg.fill ins(%cst : f32) outs(%epilogue_empty : tensor<128x128xf32>) -> tensor<128x128xf32>
  %result = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%matmul, %bias : tensor<128x128xf32>, tensor<128x128xf32>) outs(%epilogue_init : tensor<128x128xf32>) {
  ^bb0(%acc: f32, %bias_in: f32, %out0: f32):
    %sum = arith.addf %acc, %bias_in : f32
    linalg.yield %sum : f32
  } -> tensor<128x128xf32>
  return %result : tensor<128x128xf32>
}
// CHECK-DAG: #[[FILL:.+]] = #iree_cpu.lowering_config<vector_common_parallel = [8, 32]>
// CHECK-DAG: #[[MATMUL:.+]] = #iree_cpu.lowering_config<cache_parallel = [64, 64, 0], distribution = [0, 0, 0], vector_common_parallel = [8, 32, 0], vector_reduction = [0, 0, 8]>
// CHECK-DAG: #[[EPILOGUE:.+]] = #iree_cpu.lowering_config<cache_parallel = [64, 0], vector_common_parallel = [8, 32]>
// CHECK-NOT: #iree_cpu.lowering_config
// CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = #iree_cpu.pipeline<DoubleTilingExpert>, {enable_loop_peeling}>
// CHECK: func.func @matmul_epilogue_dispatch(
// CHECK-SAME: translation_info = #[[TRANSLATION]]
// CHECK: linalg.fill {lowering_config = #[[FILL]]}
// CHECK: linalg.matmul {lowering_config = #[[MATMUL]]}
// CHECK: linalg.generic
// CHECK-SAME: iterator_types = ["parallel", "parallel"]
// CHECK-SAME: lowering_config = #[[EPILOGUE]]

// -----

#target = #hal.executable.target<"hexagon", "embedded-elf-hexagon", {cpu = "hexagonv79", cpu_features = "+hvxv79,+hvx-length128b", data_layout = "e-m:e-p:32:32:32-a:0-n16:32-i64:64:64-i32:32:32-i16:16:16-i1:8:8-f32:32:32-f64:64:64-v32:32:32-v64:64:64-v512:512:512-v1024:1024:1024-v2048:2048:2048", hexagon.version = "79", iree.encoding.resolver = #iree_hexagon.hexagon_encoding_resolver<>, link_embedded = false, max_stack_allocation_size = 16384 : i64, native_vector_size = 32 : i64, target_triple = "hexagon-unknown-unknown-elf"}>
func.func @conv_fallback_dispatch(%input: tensor<1x32x32x8xf32>, %filter: tensor<3x3x8x16xf32>) -> tensor<1x30x30x16xf32> attributes {hal.executable.target = #target} {
  %cst = arith.constant 0.0 : f32
  %empty = tensor.empty() : tensor<1x30x30x16xf32>
  %init = linalg.fill ins(%cst : f32) outs(%empty : tensor<1x30x30x16xf32>) -> tensor<1x30x30x16xf32>
  %result = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%input, %filter : tensor<1x32x32x8xf32>, tensor<3x3x8x16xf32>) outs(%init : tensor<1x30x30x16xf32>) -> tensor<1x30x30x16xf32>
  return %result : tensor<1x30x30x16xf32>
}
// CHECK-DAG: #[[FILL:.+]] = #iree_cpu.lowering_config<vector_common_parallel = [1, 1, 30, 16]>
// CHECK-DAG: #[[CONV:.+]] = #iree_cpu.lowering_config<distribution = [1, 1, 30, 16, 0, 0, 0], vector_common_parallel = [1, 1, 30, 16, 0, 0, 0]>
// CHECK-NOT: #iree_cpu.lowering_config
// CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = #iree_cpu.pipeline<ConvTileAndDecomposeExpert>>
// CHECK: func.func @conv_fallback_dispatch(
// CHECK-SAME: translation_info = #[[TRANSLATION]]
// CHECK: linalg.fill {lowering_config = #[[FILL]]}
// CHECK: linalg.conv_2d_nhwc_hwcf
// CHECK-SAME: lowering_config = #[[CONV]]
// GENERIC-DAG: #[[GCONV:.+]] = #iree_cpu.lowering_config<distribution = [1, 1, 30, 16, 0, 0, 0], vector_common_parallel = [1, 1, 30, 16, 0, 0, 0]>
// GENERIC-DAG: #[[GTRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = #iree_cpu.pipeline<ConvTileAndDecomposeExpert>>
// GENERIC: func.func @conv_fallback_dispatch(
// GENERIC-SAME: translation_info = #[[GTRANSLATION]]
// GENERIC: linalg.generic
// GENERIC: lowering_config = #[[GCONV]]

// -----

#target = #hal.executable.target<"hexagon", "embedded-elf-hexagon", {cpu = "hexagonv79", cpu_features = "+hvxv79,+hvx-length128b", data_layout = "e-m:e-p:32:32:32-a:0-n16:32-i64:64:64-i32:32:32-i16:16:16-i1:8:8-f32:32:32-f64:64:64-v32:32:32-v64:64:64-v512:512:512-v1024:1024:1024-v2048:2048:2048", hexagon.version = "79", iree.encoding.resolver = #iree_hexagon.hexagon_encoding_resolver<>, link_embedded = false, max_stack_allocation_size = 16384 : i64, native_vector_size = 32 : i64, target_triple = "hexagon-unknown-unknown-elf"}>
func.func @fft_fallback_dispatch(%twiddle_real: tensor<2xf32>, %twiddle_imag: tensor<2xf32>) -> (tensor<32xf32>, tensor<32xf32>) attributes {hal.executable.target = #target} {
  %c2 = arith.constant 2 : index
  %empty_real = tensor.empty() : tensor<32xf32>
  %empty_imag = tensor.empty() : tensor<32xf32>
  %fft_real, %fft_imag = iree_linalg_ext.fft ins(%c2, %twiddle_real, %twiddle_imag : index, tensor<2xf32>, tensor<2xf32>) outs(%empty_real, %empty_imag : tensor<32xf32>, tensor<32xf32>) : tensor<32xf32>, tensor<32xf32>
  return %fft_real, %fft_imag : tensor<32xf32>, tensor<32xf32>
}
// CHECK-DAG: #[[FFT:.+]] = #iree_cpu.lowering_config<vector_common_parallel = [32]>
// CHECK-NOT: #iree_cpu.lowering_config
// CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = #iree_cpu.pipeline<Default>>
// CHECK: func.func @fft_fallback_dispatch(
// CHECK-SAME: translation_info = #[[TRANSLATION]]
// CHECK: iree_linalg_ext.fft
// CHECK-SAME: lowering_config = #[[FFT]]

// -----

// Unsupported contraction (dot): expect the Default pipeline fallback with no tiling
// selected on any op.
#target = #hal.executable.target<"hexagon", "embedded-elf-hexagon", {cpu = "hexagonv79", cpu_features = "+hvxv79,+hvx-length128b", data_layout = "e-m:e-p:32:32:32-a:0-n16:32-i64:64:64-i32:32:32-i16:16:16-i1:8:8-f32:32:32-f64:64:64-v32:32:32-v64:64:64-v512:512:512-v1024:1024:1024-v2048:2048:2048", hexagon.version = "79", iree.encoding.resolver = #iree_hexagon.hexagon_encoding_resolver<>, link_embedded = false, max_stack_allocation_size = 16384 : i64, native_vector_size = 32 : i64, target_triple = "hexagon-unknown-unknown-elf"}>
func.func @dot_dispatch(%lhs: tensor<128xf32>, %rhs: tensor<128xf32>) -> tensor<f32> attributes {hal.executable.target = #target} {
  %cst = arith.constant 0.0 : f32
  %empty = tensor.empty() : tensor<f32>
  %init = linalg.fill ins(%cst : f32) outs(%empty : tensor<f32>) -> tensor<f32>
  %result = linalg.dot ins(%lhs, %rhs : tensor<128xf32>, tensor<128xf32>) outs(%init : tensor<f32>) -> tensor<f32>
  return %result : tensor<f32>
}
// CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = #iree_cpu.pipeline<Default>>
// CHECK: func.func @dot_dispatch(
// CHECK-SAME: translation_info = #[[TRANSLATION]]
// CHECK: linalg.fill
// CHECK-NOT: lowering_config
// CHECK: linalg.dot
// CHECK-NOT: lowering_config
// CHECK: return

// -----

// Transposed-RHS batch_matmul: RHS layout is B[b, n, k]
// This is the shape of attention's Q*K^T dispatch (b=4, m=1024, n=1024, k=128).
#map_lhs = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
#map_rhs_t = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3)>
#map_out = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
#target = #hal.executable.target<"hexagon", "embedded-elf-hexagon", {cpu = "hexagonv79", cpu_features = "+hvxv79,+hvx-length128b", data_layout = "e-m:e-p:32:32:32-a:0-n16:32-i64:64:64-i32:32:32-i16:16:16-i1:8:8-f32:32:32-f64:64:64-v32:32:32-v64:64:64-v512:512:512-v1024:1024:1024-v2048:2048:2048", hexagon.version = "79", iree.encoding.resolver = #iree_hexagon.hexagon_encoding_resolver<>, link_embedded = false, max_stack_allocation_size = 16384 : i64, native_vector_size = 32 : i64, target_triple = "hexagon-unknown-unknown-elf"}>
func.func @batch_matmul_transposed_rhs_dispatch(%lhs: tensor<4x64x128xf32>, %rhs: tensor<4x64x128xf32>) -> tensor<4x64x64xf32> attributes {hal.executable.target = #target} {
  %cst = arith.constant 0.0 : f32
  %empty = tensor.empty() : tensor<4x64x64xf32>
  %init = linalg.fill ins(%cst : f32) outs(%empty : tensor<4x64x64xf32>) -> tensor<4x64x64xf32>
  %result = linalg.batch_matmul indexing_maps = [#map_lhs, #map_rhs_t, #map_out]
      ins(%lhs, %rhs : tensor<4x64x128xf32>, tensor<4x64x128xf32>)
      outs(%init : tensor<4x64x64xf32>) -> tensor<4x64x64xf32>
  return %result : tensor<4x64x64xf32>
}
// CHECK-DAG: #[[FILL:.+]] = #iree_cpu.lowering_config<vector_common_parallel = [1, 1, 1]>
// CHECK-DAG: #[[MATMUL:.+]] = #iree_cpu.lowering_config<cache_parallel = [1, 64, 64, 0], distribution = [0, 0, 0, 0], vector_common_parallel = [1, 1, 1, 0], vector_reduction = [0, 0, 0, 32]>
// CHECK-NOT: #iree_cpu.lowering_config
// CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = #iree_cpu.pipeline<DoubleTilingExpert>, {enable_loop_peeling}>
// CHECK: func.func @batch_matmul_transposed_rhs_dispatch(
// CHECK-SAME: translation_info = #[[TRANSLATION]]
// CHECK: linalg.fill {lowering_config = #[[FILL]]}
// CHECK: linalg.batch_matmul {{.*}} {lowering_config = #[[MATMUL]]}

// -----

// Transposed-RHS batch_matmul with producers. The root reduction dimension d3
// maps to the producers' innermost parallel dimension, so root vector intent on
// k should propagate as producer parallel tiling.
#map_lhs = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
#map_rhs_t = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3)>
#map_out = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
#producer_map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#target = #hal.executable.target<"hexagon", "embedded-elf-hexagon", {cpu = "hexagonv79", cpu_features = "+hvxv79,+hvx-length128b", data_layout = "e-m:e-p:32:32:32-a:0-n16:32-i64:64:64-i32:32:32-i16:16:16-i1:8:8-f32:32:32-f64:64:64-v32:32:32-v64:64:64-v512:512:512-v1024:1024:1024-v2048:2048:2048", hexagon.version = "79", iree.encoding.resolver = #iree_hexagon.hexagon_encoding_resolver<>, link_embedded = false, max_stack_allocation_size = 16384 : i64, native_vector_size = 32 : i64, target_triple = "hexagon-unknown-unknown-elf"}>
func.func @batch_matmul_transposed_rhs_with_producers_dispatch(%lhs: tensor<4x64x128xf32>, %rhs: tensor<4x64x128xf32>) -> tensor<4x64x64xf32> attributes {hal.executable.target = #target} {
  %cst = arith.constant 0.0 : f32
  %lhs_empty = tensor.empty() : tensor<4x64x128xf32>
  %rhs_empty = tensor.empty() : tensor<4x64x128xf32>
  %lhs_producer = linalg.generic {indexing_maps = [#producer_map, #producer_map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%lhs : tensor<4x64x128xf32>) outs(%lhs_empty : tensor<4x64x128xf32>) {
  ^bb0(%in: f32, %out0: f32):
    linalg.yield %in : f32
  } -> tensor<4x64x128xf32>
  %rhs_producer = linalg.generic {indexing_maps = [#producer_map, #producer_map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%rhs : tensor<4x64x128xf32>) outs(%rhs_empty : tensor<4x64x128xf32>) {
  ^bb0(%in: f32, %out0: f32):
    linalg.yield %in : f32
  } -> tensor<4x64x128xf32>
  %empty = tensor.empty() : tensor<4x64x64xf32>
  %init = linalg.fill ins(%cst : f32) outs(%empty : tensor<4x64x64xf32>) -> tensor<4x64x64xf32>
  %result = linalg.batch_matmul indexing_maps = [#map_lhs, #map_rhs_t, #map_out]
      ins(%lhs_producer, %rhs_producer : tensor<4x64x128xf32>, tensor<4x64x128xf32>)
      outs(%init : tensor<4x64x64xf32>) -> tensor<4x64x64xf32>
  return %result : tensor<4x64x64xf32>
}
// CHECK-DAG: #[[ATTN_PRODUCER_CONFIG:.+]] = #iree_cpu.lowering_config<cache_parallel = [4, 64, 0], vector_common_parallel = [1, 1, 32]>
// CHECK-DAG: #[[ATTN_FILL_CONFIG:.+]] = #iree_cpu.lowering_config<vector_common_parallel = [1, 1, 1]>
// CHECK-DAG: #[[ATTN_ROOT_CONFIG:.+]] = #iree_cpu.lowering_config<cache_parallel = [1, 64, 64, 0], distribution = [0, 0, 0, 0], vector_common_parallel = [1, 1, 1, 0], vector_reduction = [0, 0, 0, 32]>
// CHECK-NOT: #iree_cpu.lowering_config
// CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = #iree_cpu.pipeline<DoubleTilingExpert>, {enable_loop_peeling}>
// CHECK: func.func @batch_matmul_transposed_rhs_with_producers_dispatch(
// CHECK-SAME: translation_info = #[[TRANSLATION]]
// CHECK: linalg.generic
// CHECK-SAME: iterator_types = ["parallel", "parallel", "parallel"]
// CHECK-SAME: lowering_config = #[[ATTN_PRODUCER_CONFIG]]
// CHECK: linalg.generic
// CHECK-SAME: iterator_types = ["parallel", "parallel", "parallel"]
// CHECK-SAME: lowering_config = #[[ATTN_PRODUCER_CONFIG]]
// CHECK: linalg.fill {lowering_config = #[[ATTN_FILL_CONFIG]]}
// CHECK: linalg.batch_matmul
// CHECK-SAME: lowering_config = #[[ATTN_ROOT_CONFIG]]

// -----

// Softmax dispatch from the attention path.
#target = #hal.executable.target<"hexagon", "embedded-elf-hexagon", {cpu = "hexagonv79", cpu_features = "+hvxv79,+hvx-length128b", data_layout = "e-m:e-p:32:32:32-a:0-n16:32-i64:64:64-i32:32:32-i16:16:16-i1:8:8-f32:32:32-f64:64:64-v32:32:32-v64:64:64-v512:512:512-v1024:1024:1024-v2048:2048:2048", hexagon.version = "79", iree.encoding.resolver = #iree_hexagon.hexagon_encoding_resolver<>, link_embedded = false, max_stack_allocation_size = 16384 : i64, native_vector_size = 32 : i64, target_triple = "hexagon-unknown-unknown-elf"}>
func.func @softmax_dispatch(%src: tensor<4x1024x1024xf32>, %mask: tensor<4x1024xi8>) -> tensor<4x1024x1024xf32> attributes {hal.executable.target = #target} {
  %cst = arith.constant 0xFFC00000 : f32
  %cst_0 = arith.constant 0.000000e+00 : f32
  %empty = tensor.empty() : tensor<4x1024x1024xf32>
  %red_empty = tensor.empty() : tensor<4x1024xf32>
  %max_init = linalg.fill ins(%cst : f32) outs(%red_empty : tensor<4x1024xf32>) -> tensor<4x1024xf32>
  %max = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%src : tensor<4x1024x1024xf32>) outs(%max_init : tensor<4x1024xf32>) {
  ^bb0(%in: f32, %out0: f32):
    %m = arith.maxnumf %in, %out0 : f32
    linalg.yield %m : f32
  } -> tensor<4x1024xf32>
  %sum_init = linalg.fill ins(%cst_0 : f32) outs(%red_empty : tensor<4x1024xf32>) -> tensor<4x1024xf32>
  %sum = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%src, %max : tensor<4x1024x1024xf32>, tensor<4x1024xf32>) outs(%sum_init : tensor<4x1024xf32>) {
  ^bb0(%in: f32, %max_in: f32, %out0: f32):
    %shifted = arith.subf %in, %max_in : f32
    %exp = math.exp %shifted : f32
    %s = arith.addf %exp, %out0 : f32
    linalg.yield %s : f32
  } -> tensor<4x1024xf32>
  %result = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%mask, %src, %max, %sum : tensor<4x1024xi8>, tensor<4x1024x1024xf32>, tensor<4x1024xf32>, tensor<4x1024xf32>) outs(%empty : tensor<4x1024x1024xf32>) {
  ^bb0(%mask_in: i8, %in: f32, %max_in: f32, %sum_in: f32, %out0: f32):
    %shifted = arith.subf %in, %max_in : f32
    %exp = math.exp %shifted : f32
    %normalized = arith.divf %exp, %sum_in : f32
    %is_masked = arith.trunci %mask_in : i8 to i1
    %selected = arith.select %is_masked, %cst_0, %normalized : f32
    linalg.yield %selected : f32
  } -> tensor<4x1024x1024xf32>
  return %result : tensor<4x1024x1024xf32>
}
// CHECK-DAG: #[[FILL:.+]] = #iree_cpu.lowering_config<vector_common_parallel = [1, 1]>
// CHECK-DAG: #[[MAX:.+]] = #iree_cpu.lowering_config<cache_parallel = [4, 0, 0], vector_common_parallel = [1, 1, 0], vector_reduction = [0, 0, 32]>
// CHECK-DAG: #[[SUM:.+]] = #iree_cpu.lowering_config<cache_parallel = [4, 0, 0], distribution = [0, 0, 0], vector_common_parallel = [1, 1, 0], vector_reduction = [0, 0, 32]>
// CHECK-DAG: #[[NORM:.+]] = #iree_cpu.lowering_config<cache_parallel = [4, 64, 0], vector_common_parallel = [1, 1, 32]>
// CHECK-NOT: #iree_cpu.lowering_config
// CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = #iree_cpu.pipeline<DoubleTilingExpert>, {enable_loop_peeling}>
// CHECK: func.func @softmax_dispatch(
// CHECK-SAME: translation_info = #[[TRANSLATION]]
// CHECK: linalg.fill {lowering_config = #[[FILL]]}
// CHECK: linalg.generic
// CHECK-SAME: iterator_types = ["parallel", "parallel", "reduction"]
// CHECK-SAME: lowering_config = #[[MAX]]
// CHECK: linalg.fill {lowering_config = #[[FILL]]}
// CHECK: linalg.generic
// CHECK-SAME: iterator_types = ["parallel", "parallel", "reduction"]
// CHECK-SAME: lowering_config = #[[SUM]]
// CHECK: linalg.generic
// CHECK-SAME: iterator_types = ["parallel", "parallel", "parallel"]
// CHECK-SAME: lowering_config = #[[NORM]]

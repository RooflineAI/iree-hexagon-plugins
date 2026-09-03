// Test written to reflect current Hexagon selector policy.
//
// This file exercises the non-VTCM, non-HMX strategy policies. Dedicated tests
// cover those opt-in paths below the planner boundary.
//
// RUN: iree-opt --iree-load-plugin=hexagon=$ROOF_HEXAGON_COMPILER_PLUGIN \
// RUN:   --iree-hexagon-enable-vtcm-tiling=false \
// RUN:   --pass-pipeline='builtin.module(iree-hexagon-select-lowering-strategy)' \
// RUN:   --split-input-file %s | FileCheck %s

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

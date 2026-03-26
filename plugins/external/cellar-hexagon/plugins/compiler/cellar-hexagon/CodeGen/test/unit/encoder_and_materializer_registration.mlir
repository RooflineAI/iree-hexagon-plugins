// This test is based on the 4x4 matmul example test.
// It ensures the Hexagon encoder and materializer are registered: the host encoding pass should lower the matmul into pack/mmt4d/unpack.
//
// RUN: iree-opt --iree-load-plugin=cellar_hexagon=$CELLAR_HEXAGON_COMPILER_PLUGIN \
// RUN:   --pass-pipeline='builtin.module(util.func(iree-dispatch-creation-set-encoding), iree-global-opt-materialize-homogeneous-encodings)' \
// RUN:   %s | FileCheck %s
//
// CHECK: linalg.pack
// CHECK: linalg.mmt4d
// CHECK: linalg.unpack
// CHECK-NOT: linalg.matmul

#executable_target_embedded_elf_hexagon = #hal.executable.target<"hexagon", "embedded-elf-hexagon", {cpu = "hexagonv79", cpu_features = "+hvxv79,+hvx-length128b", data_layout = "e-m:e-p:32:32:32-a:0-n16:32-i64:64:64-i32:32:32-i16:16:16-i1:8:8-f32:32:32-f64:64:64-v32:32:32-v64:64:64-v512:512:512-v1024:1024:1024-v2048:2048:2048", hexagon.version = "79", iree.encoding.resolver = #iree_hexagon.hexagon_encoding_resolver<>, link_embedded = false, max_stack_allocation_size = 32768 : i64, native_vector_size = 32 : i64, target_triple = "hexagon-unknown-unknown-elf"}>
#device_target_hexagon = #hal.device.target<"hexagon", [#executable_target_embedded_elf_hexagon]> : !hal.device

// -----// IR Dump After AnnotateDataTilingHintsPass (iree-dispatch-creation-annotate-data-tiling-hints) //----- //
module @test attributes {hal.device.targets = [#device_target_hexagon], stream.affinity.default = #hal.device.affinity<@__device_0>} {
  util.global private @__device_0 = #device_target_hexagon
  util.func public @matmul(%arg0: !hal.buffer_view, %arg1: !hal.buffer_view) -> !hal.buffer_view attributes {iree.abi.stub, iree.reflection = {iree.abi.declaration = "sync func @matmul(%input0: tensor<4x4xf32>, %input1: tensor<4x4xf32>) -> (%output0: tensor<4x4xf32>)"}} {
    %0 = hal.tensor.import %arg0 "input0" : !hal.buffer_view -> tensor<4x4xf32>
    %1 = hal.tensor.import %arg1 "input1" : !hal.buffer_view -> tensor<4x4xf32>
    %2 = tensor.empty() : tensor<4x4xf32>
    %3 = linalg.matmul {iree.opt.data_tiling} ins(%0, %1 : tensor<4x4xf32>, tensor<4x4xf32>) outs(%2 : tensor<4x4xf32>) -> tensor<4x4xf32>
    %4 = hal.tensor.export %3 "output0" : tensor<4x4xf32> -> !hal.buffer_view
    util.return %4 : !hal.buffer_view
  }
}

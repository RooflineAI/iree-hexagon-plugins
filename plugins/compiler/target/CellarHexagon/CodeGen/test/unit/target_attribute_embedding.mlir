// This test makes sure that we get the expected embedded attributes in the IR.
// Right now, this tests default hardcoded information such as the DataLayout but also flag defined info such as the features.

// RUN: iree-compile --iree-hal-target-backends=hexagon \
// RUN:   --iree-hal-target-device=hexagon \
// RUN:   --iree-hexagon-v=69 \
// RUN:   --iree-hexagon-features=+hvxv79,+hvx-length128b \
// RUN:   --compile-to=preprocessing \
// RUN:   %s | FileCheck %s

// CHECK: #hal.device.target<"hexagon", [#hal.executable.target<"hexagon", "embedded-elf-hexagon", {cpu = "hexagonv69", cpu_features = "+hvxv79,+hvx-length128b", data_layout = "e-m:e-p:32:32:32-a:0-n16:32-i64:64:64-i32:32:32-i16:16:16-i1:8:8-f32:32:32-f64:64:64-v32:32:32-v64:64:64-v512:512:512-v1024:1024:1024-v2048:2048:2048", hexagon.version = "69", iree.encoding.resolver = #iree_hexagon.hexagon_encoding_resolver<>, link_embedded = false, max_stack_allocation_size = 32768 : i64, native_vector_size = 32 : i64, target_triple = "hexagon-unknown-unknown-elf"}>]>

module {
  util.func public @matmul(%arg0: !hal.buffer_view, %arg1: !hal.buffer_view) -> !hal.buffer_view attributes {iree.abi.stub, iree.reflection = {iree.abi.declaration = "sync func @matmul(%input0: tensor<4x4xf32>, %input1: tensor<4x4xf32>) -> (%output0: tensor<4x4xf32>)"}} {
    %0 = hal.tensor.import %arg0 "input0" : !hal.buffer_view -> tensor<4x4xf32>
    %1 = hal.tensor.import %arg1 "input1" : !hal.buffer_view -> tensor<4x4xf32>
    %2 = tensor.empty() : tensor<4x4xf32>
    %3 = linalg.matmul ins(%0, %1 : tensor<4x4xf32>, tensor<4x4xf32>) outs(%2 : tensor<4x4xf32>) -> tensor<4x4xf32>
    %4 = hal.tensor.export %3 "output0" : tensor<4x4xf32> -> !hal.buffer_view
    util.return %4 : !hal.buffer_view
  }
}

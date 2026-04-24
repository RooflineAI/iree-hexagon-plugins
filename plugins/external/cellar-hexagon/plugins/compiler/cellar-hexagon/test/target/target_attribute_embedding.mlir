// This test makes sure that we get the expected embedded attributes in the IR.
// Right now, this tests default hardcoded information such as the DataLayout but also flag defined info such as the features.

// RUN: iree-compile --iree-load-plugin=cellar_hexagon=$CELLAR_HEXAGON_COMPILER_PLUGIN --iree-hal-target-backends=hexagon \
// RUN:   --iree-hal-target-device=hexagon \
// RUN:   --iree-hexagon-v=69 \
// RUN:   --iree-hexagon-features=+hvxv79,+hvx-length128b \
// RUN:   --compile-to=preprocessing \
// RUN:   %s | FileCheck %s

// CHECK: #hal.device.target<"hexagon", [#hal.executable.target<"hexagon", "embedded-elf-hexagon", {cpu = "hexagonv69", cpu_features = "+hvxv79,+hvx-length128b", data_layout = "e-m:e-p:32:32:32-a:0-n16:32-i64:64:64-i32:32:32-i16:16:16-i1:8:8-f32:32:32-f64:64:64-v32:32:32-v64:64:64-v512:512:512-v1024:1024:1024-v2048:2048:2048", hexagon.version = "69", iree.encoding.resolver = #iree_hexagon.hexagon_encoding_resolver<>, link_embedded = false, max_stack_allocation_size = 16384 : i64, native_vector_size = 32 : i64, target_triple = "hexagon-unknown-unknown-elf"}>]>

module @test {
  func.func @matmul(%lhs: tensor<4x4xi32>, %rhs: tensor<4x4xi32>)
      -> tensor<4x4xi32> {
    %cst = arith.constant 0 : i32
    %init = tensor.empty() : tensor<4x4xi32>
    %filled = linalg.fill ins(%cst : i32) outs(%init : tensor<4x4xi32>) -> tensor<4x4xi32>
    %result = linalg.matmul
        ins(%lhs, %rhs : tensor<4x4xi32>, tensor<4x4xi32>)
        outs(%filled : tensor<4x4xi32>)
        -> tensor<4x4xi32>
    return %result : tensor<4x4xi32>
  }
}

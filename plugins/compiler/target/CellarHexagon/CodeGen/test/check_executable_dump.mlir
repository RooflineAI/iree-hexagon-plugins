// TODO: This test is currently a placeholder only meant to check that the serializer is properly being called.
// In the future, modify this to not expected an error anymore during execution (right now returns a not implemented error)

// RUN: not iree-compile --iree-hal-target-backends=hexagon \
// RUN:   --iree-hal-target-device=hexagon \
// RUN:   --iree-hexagon-v=79 \
// RUN:   --iree-hal-dump-executable-intermediates-to=%t \
// RUN:   %s -o /dev/null 2>&1 | FileCheck %s

// CHECK: Hexagon serialization not implemented yet

module {
  func.func @matmul(%lhs: tensor<4x4xf32>, %rhs: tensor<4x4xf32>)
      -> tensor<4x4xf32> {
    %init = tensor.empty() : tensor<4x4xf32>
    %result = linalg.matmul
        ins(%lhs, %rhs : tensor<4x4xf32>, tensor<4x4xf32>)
        outs(%init : tensor<4x4xf32>)
        -> tensor<4x4xf32>
    return %result : tensor<4x4xf32>
  }
}

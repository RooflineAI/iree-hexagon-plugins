// This test extend the 4x4 one by forcing to create multiple tile instead of a single one. Does not check for correctness.
// TODO: Once vector/tensor instructions are generated, they can be added here.

// RUN: rm -f %t.vmfb
// RUN: iree-compile %s --iree-hal-target-backends=hexagon \
// RUN: --iree-hal-target-backends=hexagon \
// RUN: --iree-hexagon-v=79 \
// RUN: --iree-hexagon-features=+hvxv79,+hvx-length128b \
// RUN: -o %t.vmfb
// RUN: test -s %t.vmfb

module {
  func.func @matmul(%lhs: tensor<64x64xf32>, %rhs: tensor<64x64xf32>)
      -> tensor<64x64xf32> {
    %init = tensor.empty() : tensor<64x64xf32>
    %result = linalg.matmul
        ins(%lhs, %rhs : tensor<64x64xf32>, tensor<64x64xf32>)
        outs(%init : tensor<64x64xf32>)
        -> tensor<64x64xf32>
    return %result : tensor<64x64xf32>
  }
}

// This test extends the 4x4 one by forcing to create multiple tile instead of a single one. Does not check for correctness.
// TODO: Once vector/tensor instructions are generated, they can be added here.

// RUN: rm -f %t.vmfb
// RUN: iree-compile %s --iree-hal-target-device=hexagon \
// RUN: --iree-hexagon-v=79 \
// RUN: --iree-hexagon-features=+hvxv79,+hvx-length128b \
// RUN: -o %t.vmfb
// RUN: test -s %t.vmfb

module {
  func.func @matmul(%lhs: tensor<64x64xi32>, %rhs: tensor<64x64xi32>)
      -> tensor<64x64xi32> {
    %cst = arith.constant 0 : i32
    %init = tensor.empty() : tensor<64x64xi32>
    %filled = linalg.fill ins(%cst : i32) outs(%init : tensor<64x64xi32>) -> tensor<64x64xi32>
    %result = linalg.matmul
        ins(%lhs, %rhs : tensor<64x64xi32>, tensor<64x64xi32>)
        outs(%filled : tensor<64x64xi32>)
        -> tensor<64x64xi32>
    return %result : tensor<64x64xi32>
  }
}

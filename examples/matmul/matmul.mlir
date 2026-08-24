module {
  func.func @matmul(%lhs: tensor<32x32xf16>, %rhs: tensor<32x32xf16>)
      -> tensor<32x32xf32> {
    %c0 = arith.constant 0.0 : f32
    %init = tensor.empty() : tensor<32x32xf32>
    %zero = linalg.fill ins(%c0 : f32) outs(%init : tensor<32x32xf32>)
        -> tensor<32x32xf32>
    %result = linalg.matmul
        ins(%lhs, %rhs : tensor<32x32xf16>, tensor<32x32xf16>)
        outs(%zero : tensor<32x32xf32>)
        -> tensor<32x32xf32>
    return %result : tensor<32x32xf32>
  }
}

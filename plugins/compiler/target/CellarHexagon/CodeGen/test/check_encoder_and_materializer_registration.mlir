// This test requires the encoder and materializer to be properly registered. Without correct registration, the materialize-host-encoding pass will not lower the matmul op.
// Additionally, it requires compilation to arrive to the executable-configurations stage.

// RUN: iree-compile %s --iree-hal-target-backends=hexagon \
// RUN: --mlir-disable-threading \
// RUN: --compile-to=executable-configurations \ 
// RUN: --mlir-print-ir-after=iree-codegen-materialize-host-encoding \
// RUN: -o /dev/null  2>&1 | FileCheck %s

// CHECK: linalg.pack
// CHECK: linalg.mmt4d
// CHECK: linalg.unpack
// CHECK-NOT: linalg.matmul

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

// This test makes sure that this small matmul is correctly lowered throughout the whole pipeline.
// Additional intermediate checks to guarantee current behavior:
// Check that we generate a mmtd4d operation and that the matmul does not get lowered as a linalg.generic

// RUN: rm -f %t.vmfb
// RUN: iree-compile %s --iree-hal-target-backends=hexagon \
// RUN: --iree-opt-data-tiling=true \
// RUN: --iree-hal-target-backends=hexagon \
// RUN: --iree-hexagon-v=79 \
// RUN: --iree-hexagon-features=+hvxv79,+hvx-length128b \
// RUN: --mlir-print-ir-after=iree-codegen-materialize-device-encoding \
// RUN: -o %t.vmfb 2>&1 | FileCheck %s
// RUN: test -s %t.vmfb

// CHECK: // -----// IR Dump After MaterializeDeviceEncodingPass
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

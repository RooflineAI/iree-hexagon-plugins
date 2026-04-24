// This test makes sure that this small matmul is correctly lowered throughout the whole pipeline.
// Additional intermediate checks to guarantee current behavior:
// Check that we generate a mmtd4d operation and that the matmul does not get lowered as a linalg.generic

// RUN: rm -f %t.vmfb
// RUN: iree-compile %s --iree-load-plugin=cellar_hexagon=$CELLAR_HEXAGON_COMPILER_PLUGIN --iree-hal-target-backends=hexagon \
// RUN: --iree-opt-data-tiling=true \
// RUN: --iree-hal-target-backends=hexagon \
// RUN: --iree-hexagon-v=79 \
// RUN: --iree-hexagon-features=+hvxv79,+hvx-length128b \
// RUN: --mlir-print-ir-after=iree-codegen-materialize-device-encoding \
// RUN: -o %t.vmfb 2>&1 | FileCheck %s
// RUN: test -s %t.vmfb

// CHECK: linalg.mmt4d
// CHECK-NOT: linalg.matmul

module {
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

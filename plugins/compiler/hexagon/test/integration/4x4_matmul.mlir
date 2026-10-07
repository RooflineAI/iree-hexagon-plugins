// Compile a small integer matmul through the HVX pipeline, checking its
// vectorized contraction and the resulting VMFB.

// RUN: rm -f %t.vmfb
// RUN: iree-compile %s --iree-hal-target-device=hexagon \
// RUN: --iree-opt-data-tiling=false \
// RUN: --iree-hexagon-enable-vtcm-tiling=false \
// RUN: --iree-hexagon-enable-hmx-matmul=false \
// RUN: --iree-hexagon-v=79 \
// RUN: --iree-hexagon-features=+hvxv79,+hvx-length128b \
// RUN: --mlir-print-ir-after=iree-codegen-generic-vectorization \
// RUN: -o %t.vmfb 2>&1 | FileCheck %s
// RUN: test -s %t.vmfb

// CHECK-LABEL: func.func @matmul_dispatch_0_matmul_4x4x4_i32()
// CHECK: vector.contract {{.*}} : vector<4x4xi32>, vector<4x4xi32> into vector<4x4xi32>

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

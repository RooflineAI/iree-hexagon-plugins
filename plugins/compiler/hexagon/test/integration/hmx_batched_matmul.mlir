// Minimized compilation regression for the loop-carried lowering exercised by
// testing/matmul/batched_4x1024x128x1024. This test only checks that a VMFB is
// produced; the HMX runtime simulator tests cover primitive numerical
// correctness.

// RUN: rm -f %t.vmfb
// RUN: iree-compile %s \
// RUN:   --iree-load-plugin=hexagon=$ROOF_HEXAGON_COMPILER_PLUGIN \
// RUN:   --iree-hal-target-device=hexagon \
// RUN:   --iree-hexagon-v=79 \
// RUN:   --iree-hexagon-features=+hvxv79,+hvx-length128b \
// RUN:   --iree-hexagon-enable-hmx-matmul=true \
// RUN:   --iree-hexagon-launch-config-selector=hexagon \
// RUN:   -o %t.vmfb
// RUN: test -s %t.vmfb

module {
  func.func @matmul(%lhs: tensor<2x64x32xf16>,
                    %rhs: tensor<2x32x64xf16>)
      -> tensor<2x64x64xf32> {
    %zero = arith.constant 0.0 : f32
    %init = tensor.empty() : tensor<2x64x64xf32>
    %filled = linalg.fill ins(%zero : f32)
        outs(%init : tensor<2x64x64xf32>) -> tensor<2x64x64xf32>
    %result = linalg.batch_matmul
        ins(%lhs, %rhs : tensor<2x64x32xf16>, tensor<2x32x64xf16>)
        outs(%filled : tensor<2x64x64xf32>) -> tensor<2x64x64xf32>
    return %result : tensor<2x64x64xf32>
  }
}

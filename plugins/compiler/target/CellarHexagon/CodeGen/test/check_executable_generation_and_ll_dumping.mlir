// This test makes sure that:
//  - debug files are properly dumped when passing the --iree-hal-dump-executable-intermediates-to flag
//  - that the full lowering pipeline runs and outputs a vmfb. This includes serialization and a call to llvm right now
// This test does not check for correctness in any way, only that everything is properly connected

// RUN: iree-compile --iree-hal-target-backends=hexagon \
// RUN:   --iree-hal-target-device=hexagon \
// RUN:   --iree-hexagon-v=79 \
// RUN:   --iree-hal-dump-executable-intermediates-to=%T \
// RUN:   %s -o %T/matmul.vmfb 2>&1
// RUN: test -f %T/matmul.vmfb
// RUN: find %T -name "*.ll" | grep .

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

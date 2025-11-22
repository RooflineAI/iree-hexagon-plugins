// This test makes sure that the generated dynamically linked libraries have the correct file type, machine and ABI.

// RUN: iree-compile --iree-hal-target-backends=hexagon \
// RUN:   --iree-hal-target-device=hexagon \
// RUN:   --iree-hexagon-v=79 \
// RUN:   --iree-hal-dump-executable-intermediates-to=%T \
// RUN:   --iree-hal-dump-executable-binaries-to=%T \
// RUN:   %s -o %T/matmul.vmfb
// RUN: readelf -h %T/module_matmul_dispatch_0_embedded_elf_hexagon.so | FileCheck %s
// RUN: readelf -h %T/module_matmul_dispatch_1_embedded_elf_hexagon.so | FileCheck %s

// OS/ABI
// CHECK: UNIX - System V
// File type
// CHECK: DYN (Shared object file)
// Machine
// CHECK: QUALCOMM DSP6 Processor

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

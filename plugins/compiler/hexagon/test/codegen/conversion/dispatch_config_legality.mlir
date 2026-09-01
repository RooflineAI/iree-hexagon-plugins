// `iree_codegen.dispatch_config` is a module-level op whose region computes the
// workgroup count that IREE later moves into `hal.executable.export`. The
// conversion below marks the arith dialect illegal, so without an explicit
// exemption the partial conversion descends into that region and rewrites it
// into LLVM dialect ops stitched together with unrealized casts. Mirror the
// LLVMCPU backend and keep the region intact.
// RUN: iree-opt \
// RUN:   --pass-pipeline='builtin.module(iree-hexagon-convert-to-llvm)' \
// RUN:   --split-input-file %s | FileCheck %s

// CHECK-LABEL: iree_codegen.dispatch_config @static_workgroup_count
// CHECK-NEXT:  ^bb0(%{{.+}}: !hal.device):
// CHECK-NEXT:    %[[C1:.+]] = arith.constant 1 : index
// CHECK-NEXT:    iree_codegen.yield %[[C1]], %[[C1]], %[[C1]]
// CHECK-NOT:   llvm.mlir.constant
// CHECK-NOT:   builtin.unrealized_conversion_cast
module {
  llvm.func @static_workgroup_count(%environment: !llvm.ptr, %dispatch_state: !llvm.ptr, %workgroup_state: !llvm.ptr) {
    llvm.return
  }
  iree_codegen.dispatch_config @static_workgroup_count workgroup_size = [1, 1, 1] {
  ^bb0(%device: !hal.device):
    %c1 = arith.constant 1 : index
    iree_codegen.yield %c1, %c1, %c1 : index, index, index
  }
}

// -----

// A workload-dependent count is the case that actually breaks: the arith and
// affine ops become `llvm.icmp`/`llvm.sdiv`/`llvm.select` chains that the
// export region cannot consume.

// CHECK-LABEL: iree_codegen.dispatch_config @dynamic_workgroup_count
// CHECK:         affine.apply
// CHECK:         iree_codegen.yield
// CHECK-NOT:   llvm.sdiv
// CHECK-NOT:   llvm.select
// CHECK-NOT:   builtin.unrealized_conversion_cast
module {
  llvm.func @dynamic_workgroup_count(%environment: !llvm.ptr, %dispatch_state: !llvm.ptr, %workgroup_state: !llvm.ptr) {
    llvm.return
  }
  iree_codegen.dispatch_config @dynamic_workgroup_count workgroup_size = [64, 1, 1] {
  ^bb0(%workload: index):
    %count = affine.apply affine_map<()[s0] -> (s0 ceildiv 64)>()[%workload]
    %c1 = arith.constant 1 : index
    iree_codegen.yield %count, %c1, %c1 : index, index, index
  }
}

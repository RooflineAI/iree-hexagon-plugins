// This file checks that malformed Hexagon runtime conversion inputs fail with
// diagnostics instead of crashing: runtime-state access requires dispatch ABI
// arguments, existing profiler helper declarations must be compatible, and
// unexpanded HMX operations are rejected.
// RUN: iree-opt --verify-diagnostics --split-input-file \
// RUN:   --pass-pipeline='builtin.module(iree-hexagon-convert-to-llvm)' %s

#hexagon_target = #hal.executable.target<"hexagon", "embedded-elf-hexagon", {
  data_layout = "e-m:e-p:32:32:32-a:0-n16:32-i64:64:64-i32:32:32-i16:16:16-i1:8:8-f32:32:32-f64:64:64-v32:32:32-v64:64:64-v512:512:512-v1024:1024:1024-v2048:2048:2048",
  target_triple = "hexagon-unknown-unknown-elf"
}>

module attributes {hal.executable.target = #hexagon_target} {
  // Verifies get_runtime_state diagnoses an enclosing LLVM function without
  // the required dispatch-state argument.
  llvm.func @missing_dispatch_state() {
    // expected-error@+1 {{failed to legalize operation 'iree_hexagon.get_runtime_state' that was explicitly marked illegal}}
    %state = iree_hexagon.get_runtime_state : !iree_hexagon.runtime_state
    %record = iree_hexagon.profiler.begin %state <
      zone_type = marker
    > : !iree_hexagon.runtime_state -> !iree_hexagon.profiler_record
    iree_hexagon.profiler.end %record : !iree_hexagon.profiler_record
    llvm.return
  }
}

// -----

#hexagon_target = #hal.executable.target<"hexagon", "embedded-elf-hexagon", {
  data_layout = "e-m:e-p:32:32:32-a:0-n16:32-i64:64:64-i32:32:32-i16:16:16-i1:8:8-f32:32:32-f64:64:64-v32:32:32-v64:64:64-v512:512:512-v1024:1024:1024-v2048:2048:2048",
  target_triple = "hexagon-unknown-unknown-elf"
}>

module attributes {hal.executable.target = #hexagon_target} {
  // Deliberately incompatible with the runtime helper signature. Verifies the
  // symbol conflict is diagnosed instead of emitting a mismatched call.
  // expected-error@+1 {{redefinition of function 'hexagon_runtime_profiler_zone_begin' of different type}}
  llvm.func @hexagon_runtime_profiler_zone_begin(i32) -> i32

  llvm.func @incompatible_profiler_helper(%environment: !llvm.ptr,
                                           %dispatch_state: !llvm.ptr,
                                           %workgroup_state: !llvm.ptr) {
    %state = iree_hexagon.get_runtime_state : !iree_hexagon.runtime_state
    // expected-error@+1 {{failed to legalize operation 'iree_hexagon.profiler.begin' that was explicitly marked illegal}}
    %record = iree_hexagon.profiler.begin %state <
      zone_type = marker
    > : !iree_hexagon.runtime_state -> !iree_hexagon.profiler_record
    iree_hexagon.profiler.end %record : !iree_hexagon.profiler_record
    llvm.return
  }
}

// -----

// An HMX operation without a lowering to a kernel call must not survive the
// conversion: earlier stages are expected to expand it.
func.func private @reject_unexpanded_hmx_matmul(
    %lhs: memref<1x1x16x32x2xf16, 1>,
    %rhs: memref<1x1x16x32x2xf16, 1>,
    %acc: memref<16x32x2xf16, 1>) {
  // expected-error @+1 {{failed to legalize operation 'iree_hexagon.hmx.matmul' that was explicitly marked illegal}}
  iree_hexagon.hmx.matmul ins(%lhs, %rhs : memref<1x1x16x32x2xf16, 1>, memref<1x1x16x32x2xf16, 1>) outs(%acc : memref<16x32x2xf16, 1>)
  return
}

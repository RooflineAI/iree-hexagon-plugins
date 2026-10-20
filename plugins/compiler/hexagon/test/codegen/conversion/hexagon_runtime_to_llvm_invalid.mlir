// This file checks that malformed Hexagon runtime conversion inputs fail with
// diagnostics instead of crashing: runtime-state access requires dispatch ABI
// arguments, existing profiler helper declarations must be compatible,
// unexpanded HMX operations are rejected, and VTCM allocations require the
// dispatch state to reach the dispatch's VTCM partition.
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

// -----

// VTCM comes from the dispatch's partition, which is only reachable through the
// dispatch state, so an allocation outside a dispatch entry point is rejected.
func.func private @reject_alloc_outside_dispatch() {
  // expected-error @+1 {{'memref.alloc' op outside a dispatch function is not supported}}
  %buf = memref.alloc() : memref<4xf32>
  // expected-error @+1 {{'memref.dealloc' op outside a dispatch function is not supported}}
  memref.dealloc %buf : memref<4xf32>
  return
}

// -----

func.func private @reject_vtcm_alloc_outside_dispatch() {
  // expected-error @+1 {{'hexagonmem.alloc' op outside a dispatch function is not supported}}
  %buf = hexagonmem.alloc() <alignment = 2048> : memref<4xf32, 1>
  // expected-error @+1 {{'hexagonmem.dealloc' op outside a dispatch function is not supported}}
  hexagonmem.dealloc %buf : memref<4xf32, 1>
  return
}

// -----

// The dispatch's VTCM partition hands out 128 byte and 2 KiB aligned VTCM only.
func.func @reject_unsupported_vtcm_alignment() {
  // expected-error @+1 {{requests VTCM aligned to 256 bytes; only up to 128 and 2048 are supported}}
  %buf = hexagonmem.alloc() <alignment = 256> : memref<4xf32, 1>
  hexagonmem.dealloc %buf : memref<4xf32, 1>
  return
}

// -----

// The VTCM scratch of a HexKL matmul is allocated at the dispatch entry, so its
// dims must be known there.
#pipeline_layout = #hal.pipeline.layout<constants = 1, bindings = []>
func.func @reject_dynamic_hexkl_matmul() {
  %rows = hal.interface.constant.load layout(#pipeline_layout) ordinal(0) : index
  %lhs = memref.alloca(%rows) : memref<?x4xf16>
  %rhs = memref.alloca() : memref<4x3xf16>
  %out = memref.alloca(%rows) : memref<?x3xf32>
  // expected-error @+1 {{with dynamic dims is not supported; its VTCM scratch is allocated at the dispatch entry}}
  hexkl.matmul ins(%lhs, %rhs : memref<?x4xf16>, memref<4x3xf16>)
               outs(%out : memref<?x3xf32>)
  return
}

// -----

// A conflicting declaration of the partition's allocation function must fail
// the pass instead of falling back to `malloc`, whose frees are dropped.
// expected-error @+1 {{redefinition of function 'hexagon_runtime_vtcm_alloc' of different type}}
llvm.func @hexagon_runtime_vtcm_alloc(i32) -> i32

func.func @reject_conflicting_vtcm_alloc_declaration() {
  // expected-error @+1 {{can not declare 'hexagon_runtime_vtcm_alloc' to allocate from the dispatch's VTCM partition}}
  %buf = memref.alloc() : memref<4xf32>
  memref.dealloc %buf : memref<4xf32>
  return
}

// -----

// expected-error @+1 {{redefinition of function 'hexagon_runtime_vtcm_alloc' of different type}}
llvm.func @hexagon_runtime_vtcm_alloc(i32) -> i32

func.func @reject_conflicting_vtcm_alloc_declaration_hexagonmem() {
  // expected-error @+1 {{can not declare 'hexagon_runtime_vtcm_alloc' to allocate from the dispatch's VTCM partition}}
  %buf = hexagonmem.alloc() <alignment = 2048> : memref<4xf32, 1>
  hexagonmem.dealloc %buf : memref<4xf32, 1>
  return
}

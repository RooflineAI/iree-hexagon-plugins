// Hexagon supports direct calls resolved through static or native DSP runtime
// linking. Generic HAL dynamic imports and bitcode imports are currently rejected.
// RUN: iree-opt --verify-diagnostics --split-input-file \
// RUN:   --pass-pipeline='builtin.module(iree-hexagon-convert-to-llvm)' %s

#hexagon_target = #hal.executable.target<"hexagon", "embedded-elf-hexagon", {
  data_layout = "e-m:e-p:32:32:32-a:0-n16:32-i64:64:64-i32:32:32-i16:16:16-i1:8:8-f32:32:32-f64:64:64-v32:32:32-v64:64:64-v512:512:512-v1024:1024:1024-v2048:2048:2048",
  target_triple = "hexagon-unknown-unknown-elf"
}>

module attributes {hal.executable.target = #hexagon_target} {
  llvm.func @dynamic_import(i32) -> i32

  llvm.func @entry(%environment: !llvm.ptr, %dispatch_state: !llvm.ptr,
                   %workgroup_state: !llvm.ptr) -> i32 {
    %c0 = llvm.mlir.constant(0 : i32) : i32
    // expected-error@+1 {{calls unsupported external function 'dynamic_import'; the Hexagon plugins do not currently support generic dynamic or bitcode external calls; provide the function through static or native DSP runtime linking and mark it with 'hal.import.static'}}
    %0 = llvm.call @dynamic_import(%c0) : (i32) -> i32
    llvm.return %0 : i32
  }
}

// -----

#hexagon_target = #hal.executable.target<"hexagon", "embedded-elf-hexagon", {
  data_layout = "e-m:e-p:32:32:32-a:0-n16:32-i64:64:64-i32:32:32-i16:16:16-i1:8:8-f32:32:32-f64:64:64-v32:32:32-v64:64:64-v512:512:512-v1024:1024:1024-v2048:2048:2048",
  target_triple = "hexagon-unknown-unknown-elf"
}>

module attributes {hal.executable.target = #hexagon_target} {
  llvm.func @bitcode_import(i32) -> i32 attributes {hal.import.bitcode}

  llvm.func @entry(%environment: !llvm.ptr, %dispatch_state: !llvm.ptr,
                   %workgroup_state: !llvm.ptr) -> i32 {
    %c0 = llvm.mlir.constant(0 : i32) : i32
    // expected-error@+1 {{calls unsupported external function 'bitcode_import'; the Hexagon plugins do not currently support generic dynamic or bitcode external calls; provide the function through static or native DSP runtime linking and mark it with 'hal.import.static'}}
    %0 = llvm.call @bitcode_import(%c0) : (i32) -> i32
    llvm.return %0 : i32
  }
}

// -----

#hexagon_target = #hal.executable.target<"hexagon", "embedded-elf-hexagon", {
  data_layout = "e-m:e-p:32:32:32-a:0-n16:32-i64:64:64-i32:32:32-i16:16:16-i1:8:8-f32:32:32-f64:64:64-v32:32:32-v64:64:64-v512:512:512-v1024:1024:1024-v2048:2048:2048",
  target_triple = "hexagon-unknown-unknown-elf"
}>

module attributes {hal.executable.target = #hexagon_target} {
  llvm.func @static_import(i32) -> i32 attributes {hal.import.static}

  llvm.func @entry(%environment: !llvm.ptr, %dispatch_state: !llvm.ptr,
                   %workgroup_state: !llvm.ptr) -> i32 {
    %c0 = llvm.mlir.constant(0 : i32) : i32
    %0 = llvm.call @static_import(%c0) : (i32) -> i32
    llvm.return %0 : i32
  }
}

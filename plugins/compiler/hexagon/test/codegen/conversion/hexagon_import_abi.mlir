// This file checks every post-conversion import rewrite: dynamic imports,
// aliased imports, direct static/native calls, and matching
// function/call rewrites for bitcode imports with extra ABI fields.
// RUN: iree-opt \
// RUN:   --pass-pipeline='builtin.module(iree-hexagon-convert-to-llvm)' \
// RUN:   --split-input-file %s | FileCheck %s

#hexagon_target = #hal.executable.target<"hexagon", "embedded-elf-hexagon", {
  data_layout = "e-m:e-p:32:32:32-a:0-n16:32-i64:64:64-i32:32:32-i16:16:16-i1:8:8-f32:32:32-f64:64:64-v32:32:32-v64:64:64-v512:512:512-v1024:1024:1024-v2048:2048:2048",
  target_triple = "hexagon-unknown-unknown-elf"
}>

module attributes {hal.executable.target = #hexagon_target} {
  llvm.func @dynamic_import(i32) -> i32
  llvm.func @aliased_import(i32) -> i32 attributes {hal.import.name = "shared_import"}
  llvm.func @dynamic_with_fields(i32) -> i32 attributes {
    hal.import.fields = ["processor_data", "processor_id"]
  }
  llvm.func @static_import(i32) -> i32 attributes {hal.import.static}
  llvm.func @native_import(i32) -> i32 attributes {hexagon.native_runtime_link}

  // Verifies declared and aliased external calls use the HAL import thunk,
  // while static and native runtime calls remain direct.
  // CHECK-DAG: llvm.mlir.global internal @__import_ordinal_dynamic_import() {{.+}}hal.executable.import.key = "dynamic_import"
  // CHECK-DAG: llvm.mlir.global internal @__import_ordinal_dynamic_with_fields() {{.+}}hal.executable.import.key = "dynamic_with_fields"
  // CHECK-DAG: llvm.mlir.global internal @__import_ordinal_shared_import() {{.+}}hal.executable.import.key = "shared_import"
  // CHECK-LABEL: llvm.func @dynamic_entry(
  // CHECK: %[[PARAMS:.+]] = llvm.alloca {{.+}} x !llvm.struct<(i32, i32)>
  // CHECK: %[[ENV0:.+]] = llvm.load %arg0
  // CHECK: %[[THUNK:.+]] = llvm.extractvalue %[[ENV0]][1]
  // CHECK: %[[ENV1:.+]] = llvm.load %arg0
  // CHECK: %[[FUNCS:.+]] = llvm.extractvalue %[[ENV1]][2]
  // CHECK: %[[ENV2:.+]] = llvm.load %arg0
  // CHECK: %[[CONTEXTS:.+]] = llvm.extractvalue %[[ENV2]][3]
  // CHECK: llvm.call %[[THUNK]]({{.+}}) : !llvm.ptr, (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> i32
  // CHECK: llvm.cond_br
  // CHECK: llvm.call %{{.+}}({{.+}}) : !llvm.ptr, (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> i32
  // CHECK: llvm.getelementptr inbounds %arg0[4]
  // CHECK: llvm.extractvalue {{.+}}[4]
  // CHECK: llvm.call %{{.+}}({{.+}}) : !llvm.ptr, (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> i32
  // CHECK: llvm.call @static_import
  // CHECK: llvm.call @native_import
  llvm.func @dynamic_entry(%environment: !llvm.ptr, %dispatch_state: !llvm.ptr,
                           %workgroup_state: !llvm.ptr) -> i32 {
    %c0 = llvm.mlir.constant(0 : i32) : i32
    %0 = llvm.call @dynamic_import(%c0) : (i32) -> i32
    %1 = llvm.call @aliased_import(%0) : (i32) -> i32
    %2 = llvm.call @dynamic_with_fields(%1) : (i32) -> i32
    %3 = llvm.call @static_import(%2) : (i32) -> i32
    %4 = llvm.call @native_import(%3) : (i32) -> i32
    llvm.return %4 : i32
  }
}

// CHECK-NOT: llvm.call @dynamic_import
// CHECK-NOT: llvm.call @aliased_import
// CHECK-NOT: llvm.call @dynamic_with_fields

// -----

#hexagon_target = #hal.executable.target<"hexagon", "embedded-elf-hexagon", {
  data_layout = "e-m:e-p:32:32:32-a:0-n16:32-i64:64:64-i32:32:32-i16:16:16-i1:8:8-f32:32:32-f64:64:64-v32:32:32-v64:64:64-v512:512:512-v1024:1024:1024-v2048:2048:2048",
  target_triple = "hexagon-unknown-unknown-elf"
}>

module attributes {hal.executable.target = #hexagon_target} {
  // Verifies both the bitcode declaration and its call are rewritten to append
  // processor_data and processor_id using the declared default convention.
  // CHECK: llvm.func @bitcode_with_fields(!llvm.ptr, i32, f64, !llvm.ptr, i32) -> f32
  func.func private @bitcode_with_fields(memref<f32>, i32, f64) -> f32
      attributes {
        hal.import.bitcode = true,
        hal.import.cconv = 0 : i32,
        hal.import.fields = ["processor_data", "processor_id"],
        llvm.bareptr = true
      }

  // CHECK-LABEL: llvm.func @bitcode_entry(
  // CHECK-DAG: %[[I32:.+]] = llvm.mlir.constant(42 : i32) : i32
  // CHECK-DAG: %[[F64:.+]] = llvm.mlir.constant(4.200000e+01 : f64) : f64
  // CHECK-DAG: %[[ALLOCA:.+]] = llvm.alloca
  // CHECK: %[[PROCESSOR_DATA:.+]] = llvm.getelementptr inbounds %arg0[4]
  // CHECK: %[[WORKGROUP:.+]] = llvm.load %arg2
  // CHECK: %[[PROCESSOR_ID:.+]] = llvm.extractvalue %[[WORKGROUP]][4]
  // CHECK: llvm.call @bitcode_with_fields(%[[ALLOCA]], %[[I32]], %[[F64]], %[[PROCESSOR_DATA]], %[[PROCESSOR_ID]])
  func.func @bitcode_entry() {
    %i32 = arith.constant 42 : i32
    %f64 = arith.constant 42.0 : f64
    %buffer = memref.alloca() : memref<f32>
    %result = func.call @bitcode_with_fields(%buffer, %i32, %f64)
        : (memref<f32>, i32, f64) -> f32
    func.call @sink_f32(%result) : (f32) -> ()
    return
  }

  func.func private @sink_f32(%value: f32) {
    return
  }
}

// CHECK-NOT: builtin.unrealized_conversion_cast

// -----

#hexagon_target = #hal.executable.target<"hexagon", "embedded-elf-hexagon", {
  data_layout = "e-m:e-p:32:32:32-a:0-n16:32-i64:64:64-i32:32:32-i16:16:16-i1:8:8-f32:32:32-f64:64:64-v32:32:32-v64:64:64-v512:512:512-v1024:1024:1024-v2048:2048:2048",
  target_triple = "hexagon-unknown-unknown-elf"
}>

module attributes {hal.executable.target = #hexagon_target} {
  // Verifies a bitcode import with multiple source results keeps a consistent
  // packed LLVM declaration and call signature through both ABI rewrites.
  // CHECK: llvm.func @bitcode_multi_result(i32) -> !llvm.struct<(i32, i64)>
  func.func private @bitcode_multi_result(i32) -> (i32, i64)
      attributes {hal.import.bitcode = true}

  // CHECK-LABEL: llvm.func @bitcode_multi_result_entry(
  // CHECK: %[[RESULTS:.+]] = llvm.call @bitcode_multi_result({{.+}}) : (i32) -> !llvm.struct<(i32, i64)>
  // CHECK: %[[RESULT0:.+]] = llvm.extractvalue %[[RESULTS]][0]
  // CHECK: %[[RESULT1:.+]] = llvm.extractvalue %[[RESULTS]][1]
  // CHECK: llvm.call @sink_multi(%[[RESULT0]], %[[RESULT1]])
  func.func @bitcode_multi_result_entry() {
    %input = arith.constant 7 : i32
    %result0, %result1 = func.call @bitcode_multi_result(%input)
        : (i32) -> (i32, i64)
    func.call @sink_multi(%result0, %result1) : (i32, i64) -> ()
    return
  }

  func.func private @sink_multi(%result0: i32, %result1: i64) {
    return
  }
}

// CHECK-NOT: builtin.unrealized_conversion_cast

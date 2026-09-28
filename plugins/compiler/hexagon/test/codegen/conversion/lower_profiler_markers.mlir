// This file checks Hexagon-owned runtime conversion: extended dispatch-state
// access, profiler helper reuse, zone encoding, marker string materialization
// and deduplication, empty metadata, and begin/end value threading.
// RUN: iree-opt \
// RUN:   --pass-pipeline='builtin.module(iree-hexagon-convert-to-llvm)' \
// RUN:   %s | FileCheck %s

#hexagon_target = #hal.executable.target<"hexagon", "embedded-elf-hexagon", {
  data_layout = "e-m:e-p:32:32:32-a:0-n16:32-i64:64:64-i32:32:32-i16:16:16-i1:8:8-f32:32:32-f64:64:64-v32:32:32-v64:64:64-v512:512:512-v1024:1024:1024-v2048:2048:2048",
  target_triple = "hexagon-unknown-unknown-elf"
}>

module attributes {hal.executable.target = #hexagon_target} {
  // These compatible declarations verify lookupOrCreateFn reuses existing
  // helpers. Native runtime classification should tag, not duplicate, them.
  llvm.func @hexagon_runtime_profiler_zone_begin(!llvm.ptr, i32, !llvm.ptr) -> !llvm.ptr
  llvm.func @hexagon_runtime_profiler_zone_end(!llvm.ptr)

  // Verifies one runtime-state load feeds all markers, equal strings share a
  // global, distinct strings do not alias, empty metadata becomes null, and
  // representative zone enum endpoints retain their runtime numeric values.
  // CHECK-COUNT-1: llvm.mlir.global internal constant {{.*}}("unique\00")
  // CHECK-COUNT-1: llvm.mlir.global internal constant {{.*}}("shared\00")
  // CHECK-COUNT-1: llvm.func @hexagon_runtime_profiler_zone_begin(!llvm.ptr, i32, !llvm.ptr) -> !llvm.ptr attributes {hexagon.native_runtime_link}
  // CHECK-COUNT-1: llvm.func @hexagon_runtime_profiler_zone_end(!llvm.ptr) attributes {hexagon.native_runtime_link}
  // CHECK-LABEL: llvm.func @kernel(
  // CHECK-SAME: %{{.+}}: !llvm.ptr, %[[DISPATCH:.+]]: !llvm.ptr, %{{.+}}: !llvm.ptr
  // CHECK-DAG: %[[ZONE_0:.+]] = llvm.mlir.constant(0 : i32) : i32
  // CHECK-DAG: %[[ZONE_4:.+]] = llvm.mlir.constant(4 : i32) : i32
  // CHECK-DAG: %[[ZONE_7:.+]] = llvm.mlir.constant(7 : i32) : i32
  // CHECK-DAG: %[[ZONE_8:.+]] = llvm.mlir.constant(8 : i32) : i32
  // CHECK-DAG: %[[NULL:.+]] = llvm.mlir.zero : !llvm.ptr
  // CHECK: %[[RUNTIME_ADDR:.+]] = llvm.getelementptr inbounds %[[DISPATCH]][0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(struct<"iree_hal_executable_dispatch_state_v0_t", {{.*}}>, ptr)>
  // CHECK: %[[RUNTIME:.+]] = llvm.load %[[RUNTIME_ADDR]] : !llvm.ptr -> !llvm.ptr
  // CHECK: %[[SHARED0:.+]] = llvm.call @hexagon_runtime_profiler_zone_begin(%[[RUNTIME]], %[[ZONE_7]], {{.+}}) : (!llvm.ptr, i32, !llvm.ptr) -> !llvm.ptr
  // CHECK: llvm.call @hexagon_runtime_profiler_zone_end(%[[SHARED0]])
  // CHECK: %[[SHARED1:.+]] = llvm.call @hexagon_runtime_profiler_zone_begin(%[[RUNTIME]], %[[ZONE_4]], {{.+}}) : (!llvm.ptr, i32, !llvm.ptr) -> !llvm.ptr
  // CHECK: llvm.call @hexagon_runtime_profiler_zone_end(%[[SHARED1]])
  // CHECK: %[[UNIQUE:.+]] = llvm.call @hexagon_runtime_profiler_zone_begin(%[[RUNTIME]], %[[ZONE_8]], {{.+}}) : (!llvm.ptr, i32, !llvm.ptr) -> !llvm.ptr
  // CHECK: llvm.call @hexagon_runtime_profiler_zone_end(%[[UNIQUE]])
  // CHECK: %[[EMPTY:.+]] = llvm.call @hexagon_runtime_profiler_zone_begin(%[[RUNTIME]], %[[ZONE_0]], %[[NULL]]) : (!llvm.ptr, i32, !llvm.ptr) -> !llvm.ptr
  // CHECK: llvm.call @hexagon_runtime_profiler_zone_end(%[[EMPTY]])
  llvm.func @kernel(%environment: !llvm.ptr, %dispatch_state: !llvm.ptr,
                    %workgroup_state: !llvm.ptr) {
    %state = iree_hexagon.get_runtime_state : !iree_hexagon.runtime_state

    // Checks a non-empty marker and zone value 7.
    %shared0 = iree_hexagon.profiler.begin %state {
      extra_info = "shared",
      zone_type = #iree_hexagon.profiler_zone<marker>
    } : !iree_hexagon.runtime_state -> !iree_hexagon.profiler_record
    iree_hexagon.profiler.end %shared0 : !iree_hexagon.profiler_record

    // Checks that an identical marker string reuses the same global.
    %shared1 = iree_hexagon.profiler.begin %state {
      extra_info = "shared",
      zone_type = #iree_hexagon.profiler_zone<copy>
    } : !iree_hexagon.runtime_state -> !iree_hexagon.profiler_record
    iree_hexagon.profiler.end %shared1 : !iree_hexagon.profiler_record

    // Checks a distinct marker string and the maximum zone value 8.
    %unique = iree_hexagon.profiler.begin %state {
      extra_info = "unique",
      zone_type = #iree_hexagon.profiler_zone<unknown>
    } : !iree_hexagon.runtime_state -> !iree_hexagon.profiler_record
    iree_hexagon.profiler.end %unique : !iree_hexagon.profiler_record

    // Checks absent extra_info becomes a null pointer and zone 0 is preserved.
    %empty = iree_hexagon.profiler.begin %state {
      zone_type = #iree_hexagon.profiler_zone<dsp_execution>
    } : !iree_hexagon.runtime_state -> !iree_hexagon.profiler_record
    iree_hexagon.profiler.end %empty : !iree_hexagon.profiler_record
    llvm.return
  }
}

// CHECK-NOT: iree_hexagon.get_runtime_state
// CHECK-NOT: iree_hexagon.profiler
// CHECK-NOT: builtin.unrealized_conversion_cast

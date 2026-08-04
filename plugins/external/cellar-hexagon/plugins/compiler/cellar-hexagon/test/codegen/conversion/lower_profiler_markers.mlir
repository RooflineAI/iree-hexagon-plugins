// The runtime state is read from the extended dispatch state and the profiler
// markers around it are lowered to native runtime calls as part of the
// convert-to-llvm phase-2 conversion (default pass options), which also tags
// the emitted runtime helpers as native runtime links.
// RUN: iree-opt --iree-load-plugin=cellar_hexagon=$CELLAR_HEXAGON_COMPILER_PLUGIN \
// RUN:   --pass-pipeline='builtin.module(iree-hexagon-convert-to-llvm)' \
// RUN:   %s | FileCheck %s

module {
  // The dispatch entry point carries the HAL ABI arguments
  // (environment, dispatch_state, workgroup_state) after the entry-point
  // conversion that runs before this pass.
  llvm.func @kernel(%environment: !llvm.ptr, %dispatch_state: !llvm.ptr, %workgroup_state: !llvm.ptr) {
    %context = iree_hexagon.get_runtime_state : !iree_hexagon.runtime_state
    %record = iree_hexagon.profiler.begin %context {extra_info = "hexagonmem.copy", zone_type = #iree_hexagon.profiler_zone<marker>} : !iree_hexagon.runtime_state -> !iree_hexagon.profiler_record
    iree_hexagon.profiler.end %record : !iree_hexagon.profiler_record
    llvm.return
  }
}

// CHECK-DAG: llvm.mlir.global internal constant @__hexagon_profiler_marker_0("hexagonmem.copy\00")
// CHECK-DAG: llvm.func @hexagon_runtime_profiler_zone_begin(!llvm.ptr, i32, !llvm.ptr) -> !llvm.ptr attributes {cellar_hexagon.native_runtime_link}
// CHECK-DAG: llvm.func @hexagon_runtime_profiler_zone_end(!llvm.ptr) attributes {cellar_hexagon.native_runtime_link}

// CHECK-LABEL: llvm.func @kernel(
// CHECK-SAME: %[[EXEC:[a-zA-Z0-9_]+]]: !llvm.ptr, %[[DISP:[a-zA-Z0-9_]+]]: !llvm.ptr, %[[WRKGRP:[a-zA-Z0-9_]+]]: !llvm.ptr)
// CHECK: %[[GLOB:.*]] = llvm.mlir.addressof @__hexagon_profiler_marker_0 : !llvm.ptr
// CHECK: %[[ZONE:.*]] = llvm.mlir.constant(7 : i32) : i32
// The runtime-state pointer is loaded inline from the extended dispatch state
// and, because the type converter maps `runtime_state` to `!llvm.ptr`, is passed
// straight to the zone begin call without any intermediate cast.
// CHECK: %[[RT_ST_ADDR:.*]] = llvm.getelementptr inbounds %[[DISP]][0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(struct<"iree_hal_executable_dispatch_state_v0_t", {{.*}}>, ptr)>
// CHECK: %[[RT_ST_PTR:.*]] = llvm.load %[[RT_ST_ADDR]] : !llvm.ptr -> !llvm.ptr
// CHECK-NOT: unrealized_conversion_cast
// CHECK: %[[INFO:.*]] = llvm.getelementptr %[[GLOB]]
// CHECK: %[[RECORD:.*]] = llvm.call @hexagon_runtime_profiler_zone_begin(%[[RT_ST_PTR]], %[[ZONE]], %[[INFO]]) : (!llvm.ptr, i32, !llvm.ptr) -> !llvm.ptr
// CHECK: llvm.call @hexagon_runtime_profiler_zone_end(%[[RECORD]]) : (!llvm.ptr) -> ()

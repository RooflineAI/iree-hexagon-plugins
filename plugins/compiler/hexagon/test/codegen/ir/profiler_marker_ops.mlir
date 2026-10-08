// RUN: iree-opt \
// RUN:   %s | FileCheck %s

// Test parsing and emitting of Hexagon profiler ops and zone types.

func.func @profiler_marker_ops() {
  %state = iree_hexagon.get_runtime_state : !iree_hexagon.runtime_state
  %record = iree_hexagon.profiler.begin %state <zone_type = marker, extra_info = "hexagonmem.copy"> : !iree_hexagon.runtime_state -> !iree_hexagon.profiler_record
  iree_hexagon.profiler.end %record : !iree_hexagon.profiler_record
  return
}

// CHECK-LABEL: func.func @profiler_marker_ops()
// CHECK: %[[STATE:.*]] = iree_hexagon.get_runtime_state : !iree_hexagon.runtime_state
// CHECK: %[[RECORD:.*]] = iree_hexagon.profiler.begin %[[STATE]] <zone_type = marker, extra_info = "hexagonmem.copy"> : !iree_hexagon.runtime_state -> !iree_hexagon.profiler_record
// CHECK: iree_hexagon.profiler.end %[[RECORD]] : !iree_hexagon.profiler_record

// Every zone type of the dialect enum round-trips. The cases mirror
// HEXAGON_PROFILER_ZONES in plugins/runtime/hexagon/arm_dsp/profiler.h.
func.func @profiler_zone_types() {
  %state = iree_hexagon.get_runtime_state : !iree_hexagon.runtime_state
  %dsp_execution = iree_hexagon.profiler.begin %state <zone_type = dsp_execution> : !iree_hexagon.runtime_state -> !iree_hexagon.profiler_record
  %dispatch = iree_hexagon.profiler.begin %state <zone_type = dispatch> : !iree_hexagon.runtime_state -> !iree_hexagon.profiler_record
  %kernel = iree_hexagon.profiler.begin %state <zone_type = kernel> : !iree_hexagon.runtime_state -> !iree_hexagon.profiler_record
  %barrier = iree_hexagon.profiler.begin %state <zone_type = barrier> : !iree_hexagon.runtime_state -> !iree_hexagon.profiler_record
  %copy = iree_hexagon.profiler.begin %state <zone_type = copy> : !iree_hexagon.runtime_state -> !iree_hexagon.profiler_record
  %fill = iree_hexagon.profiler.begin %state <zone_type = fill> : !iree_hexagon.runtime_state -> !iree_hexagon.profiler_record
  %memory_management = iree_hexagon.profiler.begin %state <zone_type = memory_management> : !iree_hexagon.runtime_state -> !iree_hexagon.profiler_record
  %marker = iree_hexagon.profiler.begin %state <zone_type = marker> : !iree_hexagon.runtime_state -> !iree_hexagon.profiler_record
  %unknown = iree_hexagon.profiler.begin %state <zone_type = unknown> : !iree_hexagon.runtime_state -> !iree_hexagon.profiler_record
  return
}

// CHECK-LABEL: func.func @profiler_zone_types()
// CHECK: zone_type = dsp_execution
// CHECK: zone_type = dispatch
// CHECK: zone_type = kernel
// CHECK: zone_type = barrier
// CHECK: zone_type = copy
// CHECK: zone_type = fill
// CHECK: zone_type = memory_management
// CHECK: zone_type = marker
// CHECK: zone_type = unknown

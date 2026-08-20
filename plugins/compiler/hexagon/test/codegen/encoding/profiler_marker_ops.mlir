// RUN: iree-opt --iree-load-plugin=hexagon=$HEXAGON_COMPILER_PLUGIN \
// RUN:   %s | FileCheck %s

// Test parsing and emitting of Hexagon profiler ops and zone types.

func.func @profiler_marker_ops() {
  %state = iree_hexagon.get_runtime_state : !iree_hexagon.runtime_state
  %record = iree_hexagon.profiler.begin %state {extra_info = "hexagonmem.copy", zone_type = #iree_hexagon.profiler_zone<marker>} : !iree_hexagon.runtime_state -> !iree_hexagon.profiler_record
  iree_hexagon.profiler.end %record : !iree_hexagon.profiler_record
  return
}

// CHECK-LABEL: func.func @profiler_marker_ops()
// CHECK: %[[STATE:.*]] = iree_hexagon.get_runtime_state : !iree_hexagon.runtime_state
// CHECK: %[[RECORD:.*]] = iree_hexagon.profiler.begin %[[STATE]] {extra_info = "hexagonmem.copy", zone_type = #iree_hexagon.profiler_zone<marker>} : !iree_hexagon.runtime_state -> !iree_hexagon.profiler_record
// CHECK: iree_hexagon.profiler.end %[[RECORD]] : !iree_hexagon.profiler_record

// Every zone type of the dialect enum round-trips. The cases mirror
// HEXAGON_PROFILER_ZONES in plugins/runtime/hexagon/arm_dsp/profiler.h.
func.func @profiler_zone_types() {
  %state = iree_hexagon.get_runtime_state : !iree_hexagon.runtime_state
  %dsp_execution = iree_hexagon.profiler.begin %state {zone_type = #iree_hexagon.profiler_zone<dsp_execution>} : !iree_hexagon.runtime_state -> !iree_hexagon.profiler_record
  %dispatch = iree_hexagon.profiler.begin %state {zone_type = #iree_hexagon.profiler_zone<dispatch>} : !iree_hexagon.runtime_state -> !iree_hexagon.profiler_record
  %kernel = iree_hexagon.profiler.begin %state {zone_type = #iree_hexagon.profiler_zone<kernel>} : !iree_hexagon.runtime_state -> !iree_hexagon.profiler_record
  %barrier = iree_hexagon.profiler.begin %state {zone_type = #iree_hexagon.profiler_zone<barrier>} : !iree_hexagon.runtime_state -> !iree_hexagon.profiler_record
  %copy = iree_hexagon.profiler.begin %state {zone_type = #iree_hexagon.profiler_zone<copy>} : !iree_hexagon.runtime_state -> !iree_hexagon.profiler_record
  %fill = iree_hexagon.profiler.begin %state {zone_type = #iree_hexagon.profiler_zone<fill>} : !iree_hexagon.runtime_state -> !iree_hexagon.profiler_record
  %memory_management = iree_hexagon.profiler.begin %state {zone_type = #iree_hexagon.profiler_zone<memory_management>} : !iree_hexagon.runtime_state -> !iree_hexagon.profiler_record
  %marker = iree_hexagon.profiler.begin %state {zone_type = #iree_hexagon.profiler_zone<marker>} : !iree_hexagon.runtime_state -> !iree_hexagon.profiler_record
  %unknown = iree_hexagon.profiler.begin %state {zone_type = #iree_hexagon.profiler_zone<unknown>} : !iree_hexagon.runtime_state -> !iree_hexagon.profiler_record
  return
}

// CHECK-LABEL: func.func @profiler_zone_types()
// CHECK: zone_type = #iree_hexagon.profiler_zone<dsp_execution>
// CHECK: zone_type = #iree_hexagon.profiler_zone<dispatch>
// CHECK: zone_type = #iree_hexagon.profiler_zone<kernel>
// CHECK: zone_type = #iree_hexagon.profiler_zone<barrier>
// CHECK: zone_type = #iree_hexagon.profiler_zone<copy>
// CHECK: zone_type = #iree_hexagon.profiler_zone<fill>
// CHECK: zone_type = #iree_hexagon.profiler_zone<memory_management>
// CHECK: zone_type = #iree_hexagon.profiler_zone<marker>
// CHECK: zone_type = #iree_hexagon.profiler_zone<unknown>

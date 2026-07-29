// RUN: iree-opt --iree-load-plugin=cellar_hexagon=$CELLAR_HEXAGON_COMPILER_PLUGIN \
// RUN:   %s | FileCheck %s

func.func @profiler_marker_ops() {
  %record = iree_hexagon.profiler.begin {extra_info = "hexagonmem.copy", zone_type = 7 : i32} : !iree_hexagon.profiler_record
  iree_hexagon.profiler.end %record : !iree_hexagon.profiler_record
  return
}

// CHECK-LABEL: func.func @profiler_marker_ops()
// CHECK: %[[RECORD:.*]] = iree_hexagon.profiler.begin {extra_info = "hexagonmem.copy", zone_type = 7 : i32} : !iree_hexagon.profiler_record
// CHECK: iree_hexagon.profiler.end %[[RECORD]] : !iree_hexagon.profiler_record

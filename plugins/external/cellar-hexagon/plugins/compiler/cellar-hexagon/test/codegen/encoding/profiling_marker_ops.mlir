// RUN: iree-opt --iree-load-plugin=cellar_hexagon=$CELLAR_HEXAGON_COMPILER_PLUGIN \
// RUN:   %s | FileCheck %s

func.func @profiling_marker_ops() {
  %record = iree_hexagon.profiling.begin {extra_info = "hexagonmem.copy", zone_type = 7 : i32} : !iree_hexagon.profiling_record
  iree_hexagon.profiling.end %record : !iree_hexagon.profiling_record
  return
}

// CHECK-LABEL: func.func @profiling_marker_ops()
// CHECK: %[[RECORD:.*]] = iree_hexagon.profiling.begin {extra_info = "hexagonmem.copy", zone_type = 7 : i32} : !iree_hexagon.profiling_record
// CHECK: iree_hexagon.profiling.end %[[RECORD]] : !iree_hexagon.profiling_record

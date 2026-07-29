// RUN: iree-opt --iree-load-plugin=cellar_hexagon=$CELLAR_HEXAGON_COMPILER_PLUGIN \
// RUN:   --pass-pipeline='builtin.module(iree-hexagon-lower-profiler-markers,iree-hexagon-mark-native-runtime-links)' \
// RUN:   %s | FileCheck %s

module {
  llvm.func @kernel() {
    %record = iree_hexagon.profiler.begin {extra_info = "hexagonmem.copy", zone_type = 7 : i32} : !iree_hexagon.profiler_record
    iree_hexagon.profiler.end %record : !iree_hexagon.profiler_record
    llvm.return
  }
}

// CHECK-DAG: llvm.mlir.global internal constant @__hexagon_profiler_marker_0("hexagonmem.copy\00")
// CHECK-DAG: llvm.func @hexagon_runtime_profiler_zone_begin(i32, !llvm.ptr) -> !llvm.ptr attributes {cellar_hexagon.native_runtime_link}
// CHECK-DAG: llvm.func @hexagon_runtime_profiler_zone_end(!llvm.ptr) attributes {cellar_hexagon.native_runtime_link}
// CHECK-LABEL: llvm.func @kernel()
// CHECK: %[[ZONE:.*]] = llvm.mlir.constant(7 : i32) : i32
// CHECK: %[[INFO:.*]] = llvm.getelementptr
// CHECK: %[[RECORD:.*]] = llvm.call @hexagon_runtime_profiler_zone_begin(%[[ZONE]], %[[INFO]]) : (i32, !llvm.ptr) -> !llvm.ptr
// CHECK: llvm.call @hexagon_runtime_profiler_zone_end(%[[RECORD]]) : (!llvm.ptr) -> ()

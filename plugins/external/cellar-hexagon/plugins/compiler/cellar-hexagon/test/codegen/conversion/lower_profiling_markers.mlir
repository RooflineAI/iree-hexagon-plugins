// RUN: iree-opt --iree-load-plugin=cellar_hexagon=$CELLAR_HEXAGON_COMPILER_PLUGIN \
// RUN:   --pass-pipeline='builtin.module(iree-hexagon-lower-profiling-markers,iree-hexagon-mark-native-runtime-links)' \
// RUN:   %s | FileCheck %s

module {
  llvm.func @kernel() {
    iree_hexagon.profiling.begin {extra_info = "hexagonmem.copy", zone_type = 7 : i32}
    iree_hexagon.profiling.end
    llvm.return
  }
}

// CHECK-DAG: llvm.mlir.global internal constant @__hexagon_profiling_marker_0("hexagonmem.copy\00")
// CHECK-DAG: llvm.func @hexagon_runtime_profiling_zone_begin(i32, !llvm.ptr) attributes {cellar_hexagon.native_runtime_link}
// CHECK-DAG: llvm.func @hexagon_runtime_profiling_zone_end() attributes {cellar_hexagon.native_runtime_link}
// CHECK-LABEL: llvm.func @kernel()
// CHECK: %[[ZONE:.*]] = llvm.mlir.constant(7 : i32) : i32
// CHECK: %[[INFO:.*]] = llvm.getelementptr
// CHECK: llvm.call @hexagon_runtime_profiling_zone_begin(%[[ZONE]], %[[INFO]]) : (i32, !llvm.ptr) -> ()
// CHECK: llvm.call @hexagon_runtime_profiling_zone_end() : () -> ()

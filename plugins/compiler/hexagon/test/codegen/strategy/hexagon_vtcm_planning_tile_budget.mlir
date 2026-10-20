// --iree-hexagon-vtcm-tile-budget-bytes bounds the VTCM tile search. One row
// of the 64x65536xf32 input is 256 KiB and the reduction is staged whole, so
// the parallel tile shrinks with the budget until a single row does not fit.
//
// RUN: iree-opt \
// RUN:   --iree-hexagon-enable-vtcm-tiling \
// RUN:   --iree-hexagon-enable-hmx-matmul=false \
// RUN:   --pass-pipeline='builtin.module(iree-hexagon-select-lowering-strategy)' \
// RUN:   %s | FileCheck %s --check-prefix=DEFAULT
// RUN: iree-opt \
// RUN:   --iree-hexagon-enable-vtcm-tiling \
// RUN:   --iree-hexagon-enable-hmx-matmul=false \
// RUN:   --iree-hexagon-vtcm-tile-budget-bytes=524288 \
// RUN:   --pass-pipeline='builtin.module(iree-hexagon-select-lowering-strategy)' \
// RUN:   %s | FileCheck %s --check-prefix=SMALL
// RUN: iree-opt \
// RUN:   --iree-hexagon-enable-vtcm-tiling \
// RUN:   --iree-hexagon-enable-hmx-matmul=false \
// RUN:   --iree-hexagon-vtcm-tile-budget-bytes=131072 \
// RUN:   --pass-pipeline='builtin.module(iree-hexagon-select-lowering-strategy)' \
// RUN:   %s | FileCheck %s --check-prefix=TOO-SMALL

#target = #hal.executable.target<"hexagon", "embedded-elf-hexagon", {cpu = "hexagonv79", cpu_features = "+hvxv79,+hvx-length128b", max_stack_allocation_size = 16384 : i64, target_triple = "hexagon-unknown-unknown-elf"}>

func.func @row_reduction(%input: tensor<64x65536xf32>) -> tensor<64xf32> attributes {hal.executable.target = #target} {
  %cst = arith.constant 0.0 : f32
  %empty = tensor.empty() : tensor<64xf32>
  %init = linalg.fill ins(%cst : f32) outs(%empty : tensor<64xf32>) -> tensor<64xf32>
  %result = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>],
      iterator_types = ["parallel", "reduction"]}
      ins(%input : tensor<64x65536xf32>) outs(%init : tensor<64xf32>) {
  ^bb0(%value: f32, %out: f32):
    %sum = arith.addf %value, %out : f32
    linalg.yield %sum : f32
  } -> tensor<64xf32>
  return %result : tensor<64xf32>
}

// The default 2 MiB budget stages 4 rows (1 MiB) plus the fill and init tiles.
// DEFAULT: #[[ROOT:.+]] = #iree_hexagon.lowering_config<{{.*}}vtcm = [4, 65536]>
// DEFAULT: func.func @row_reduction(
// DEFAULT: linalg.generic
// DEFAULT-SAME: lowering_config = #[[ROOT]]

// 512 KiB leaves room for a single row.
// SMALL: #[[ROOT:.+]] = #iree_hexagon.lowering_config<{{.*}}vtcm = [1, 65536]>
// SMALL: func.func @row_reduction(
// SMALL: linalg.generic
// SMALL-SAME: lowering_config = #[[ROOT]]

// 128 KiB fits no row; the optional pipeline keeps its plan without VTCM.
// TOO-SMALL-NOT: vtcm = [
// TOO-SMALL: func.func @row_reduction(

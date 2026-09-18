// RUN: iree-opt --iree-load-plugin=hexagon=$ROOF_HEXAGON_COMPILER_PLUGIN \
// RUN:   --pass-pipeline='builtin.module(func.func(iree-hexagon-expand-hmx-matmul))' \
// RUN:   --split-input-file --verify-diagnostics %s

// A rank-5 output grid is structurally valid HMX buffer IR, but it violates the
// expansion boundary: the selected pipeline must tile/fuse it to one output
// tile first.
func.func @reject_untiled_output_grid(
    %lhs: memref<2x16x16x32x2xf16, 1>,
    %rhs: memref<16x3x16x32x2xf16, 1>,
    %acc: memref<2x3x16x32x2xf16, 1>) {
  // expected-error @+1 {{hmx.matmul expansion expects singleton M and N tile grids}}
  iree_hexagon.hmx.matmul
      ins(%lhs, %rhs : memref<2x16x16x32x2xf16, 1>,
                       memref<16x3x16x32x2xf16, 1>)
      outs(%acc : memref<2x3x16x32x2xf16, 1>)
  return
}

// -----

// A packed grid that enters the function from outside has its producer beyond
// this pass, so it cannot be re-allocated to fix its alignment: substituting a
// fresh buffer would silently discard the packed data. Report it instead.
func.func @reject_unalignable_block_argument(
    // expected-error @+1 {{HMX operand must be backed by an allocation this pass can align to 2048 bytes}}
    %lhs: memref<1x1x16x32x2xf16, 1>,
    // expected-error @+1 {{HMX operand must be backed by an allocation this pass can align to 2048 bytes}}
    %rhs: memref<1x1x16x32x2xf16, 1>) {
  %acc = hexagonmem.alloc() {alignment = 2048 : i64} : memref<16x32x2xf16, 1>
  iree_hexagon.hmx.matmul
      ins(%lhs, %rhs : memref<1x1x16x32x2xf16, 1>, memref<1x1x16x32x2xf16, 1>)
      outs(%acc : memref<16x32x2xf16, 1>)
  return
}


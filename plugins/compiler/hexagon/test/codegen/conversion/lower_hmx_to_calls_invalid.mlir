// RUN: iree-opt --iree-load-plugin=hexagon=$ROOF_HEXAGON_COMPILER_PLUGIN \
// RUN:   --pass-pipeline='builtin.module(iree-hexagon-lower-hmx-to-calls)' \
// RUN:   --split-input-file --verify-diagnostics %s

func.func @reject_unknown_alignment(
    %src: memref<32x32xf16, 1>,
    %packed: memref<1x1x16x32x2xf16, 1>) {
  %packed_aligned = memref.assume_alignment %packed, 2048 : memref<1x1x16x32x2xf16, 1>
  // expected-error @+1 {{source must be known to be 64-byte aligned}}
  iree_hexagon.hmx.pack ins(%src : memref<32x32xf16, 1>)
      outs(%packed_aligned : memref<1x1x16x32x2xf16, 1>) {dim = 0 : i64}
  return
}

// -----

func.func @reject_non_vtcm(
    %src: memref<32x32xf16>,
    %packed: memref<1x1x16x32x2xf16, 1>) {
  %src_aligned = memref.assume_alignment %src, 128 : memref<32x32xf16>
  %packed_aligned = memref.assume_alignment %packed, 2048 : memref<1x1x16x32x2xf16, 1>
  // expected-error @+1 {{source must be in VTCM memory space 1}}
  iree_hexagon.hmx.pack ins(%src_aligned : memref<32x32xf16>)
      outs(%packed_aligned : memref<1x1x16x32x2xf16, 1>) {dim = 0 : i64}
  return
}

// -----

func.func @reject_misaligned_mma(
    %lhs: memref<16x32x2xf16, 1>, %rhs: memref<16x32x2xf16, 1>) {
  %lhs_aligned = memref.assume_alignment %lhs, 128 : memref<16x32x2xf16, 1>
  %rhs_aligned = memref.assume_alignment %rhs, 2048 : memref<16x32x2xf16, 1>
  %acc = iree_hexagon.hmx.acc.zero : !iree_hexagon.hmx.acc<32x32xf32>
  // expected-error @+1 {{lhs must be known to be 2048-byte aligned}}
  %next = iree_hexagon.hmx.mma %lhs_aligned, %rhs_aligned, %acc
      : memref<16x32x2xf16, 1>, memref<16x32x2xf16, 1>,
        !iree_hexagon.hmx.acc<32x32xf32> -> !iree_hexagon.hmx.acc<32x32xf32>
  return
}

// -----

func.func @reject_arbitrary_dynamic_subview_offset(%col: index) {
  %source = memref.alloc() {alignment = 2048 : i64} : memref<16x32x2xf16, 1>
  %dest = memref.alloc() {alignment = 128 : i64} : memref<32x64xf32, 1>
  %dest_tile = memref.subview %dest[0, %col] [32, 32] [1, 1]
      : memref<32x64xf32, 1> to memref<32x32xf32, strided<[64, 1], offset: ?>, 1>
  // expected-error @+1 {{destination must be known to be 128-byte aligned}}
  iree_hexagon.hmx.unpack ins(%source : memref<16x32x2xf16, 1>)
      outs(%dest_tile : memref<32x32xf32, strided<[64, 1], offset: ?>, 1>)
      {dim = 0 : i64}
  return
}

// -----

// This has currently not been considered, so just reject such a case
func.func @reject_unbounded_dynamic_pack_grid(%rows: index) {
  %source_storage = hexagonmem.alloc() : memref<128x64xf16, 1>
  %source = memref.subview %source_storage[0, 0] [%rows, 64] [1, 1]
      : memref<128x64xf16, 1>
      to memref<?x64xf16, strided<[64, 1]>, 1>
  %packed = hexagonmem.alloc() {alignment = 2048 : i64}
      : memref<2x2x16x32x2xf16, 1>
  // expected-error @+1 {{could not prove an upper bound for source dimension 0}}
  iree_hexagon.hmx.pack ins(%source : memref<?x64xf16, strided<[64, 1]>, 1>)
      outs(%packed : memref<2x2x16x32x2xf16, 1>) {dim = 0 : i64}
  return
}

// -----

// Same observation as for the test reject_unbounded_dynamic_pack_grid
func.func @reject_unbounded_dynamic_rank3_unpack(%cols: index) {
  %source = hexagonmem.alloc() {alignment = 2048 : i64}
      : memref<16x32x2xf16, 1>
  %dest_storage = hexagonmem.alloc() : memref<32x64xf32, 1>
  %dest = memref.subview %dest_storage[0, 0] [32, %cols] [1, 1]
      : memref<32x64xf32, 1>
      to memref<32x?xf32, strided<[64, 1]>, 1>
  // expected-error @+1 {{could not prove an upper bound for rank-3 unpack destination dimension 1}}
  iree_hexagon.hmx.unpack ins(%source : memref<16x32x2xf16, 1>)
      outs(%dest : memref<32x?xf32, strided<[64, 1]>, 1>) {dim = 0 : i64}
  return
}

// -----

func.func @reject_strided_accumulator_config(
    %config: memref<256xi8, strided<[2]>, 1>) {
  %aligned = memref.assume_alignment %config, 2048
      : memref<256xi8, strided<[2]>, 1>
  // expected-error @+1 {{config must be contiguous}}
  iree_hexagon.hmx.acc.setup_read %aligned
      : memref<256xi8, strided<[2]>, 1>
  return
}

// -----

func.func @reject_unexpanded_hmx_matmul(
    %lhs: memref<1x1x16x32x2xf16, 1>,
    %rhs: memref<1x1x16x32x2xf16, 1>,
    %acc: memref<16x32x2xf16, 1>) {
  // expected-error @+1 {{unexpected HMX operation remained after lowering}}
  iree_hexagon.hmx.matmul ins(%lhs, %rhs : memref<1x1x16x32x2xf16, 1>, memref<1x1x16x32x2xf16, 1>) outs(%acc : memref<16x32x2xf16, 1>)
  return
}

// -----

// expected-error @+1 {{unexpected HMX accumulator in function signature after lowering}}
func.func @reject_residual_accumulator_argument(%acc: !iree_hexagon.hmx.acc<32x32xf32>) {
  return
}

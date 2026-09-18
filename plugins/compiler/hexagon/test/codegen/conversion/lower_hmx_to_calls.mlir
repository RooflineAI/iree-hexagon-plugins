// RUN: iree-opt --iree-load-plugin=hexagon=$ROOF_HEXAGON_COMPILER_PLUGIN \
// RUN:   --pass-pipeline='builtin.module(iree-hexagon-lower-hmx-to-calls)' \
// RUN:   --split-input-file %s | FileCheck %s

// CHECK-LABEL: func.func @lower_layout_ops(
// The pack/unpack calls take (dest, src, stride, actual_rows, actual_cols,
// row_tiles, col_tiles): tile counts from the padded grid, actual dims from the
// (possibly ragged) staged buffer.
// CHECK:       call @iree_hexagon_hmx_pack_f16(%{{.+}}, %{{.+}}, %{{.+}}, %{{.+}}, %{{.+}}, %{{.+}}, %{{.+}}) : (i32, i32, i32, i32, i32, i32, i32) -> ()
// CHECK:       call @iree_hexagon_hmx_unpack_acc_f16_to_f32(%{{.+}}, %{{.+}}, %{{.+}}, %{{.+}}, %{{.+}}, %{{.+}}, %{{.+}}) : (i32, i32, i32, i32, i32, i32, i32) -> ()
// CHECK-NOT:   iree_hexagon.hmx.pack
// CHECK-NOT:   iree_hexagon.hmx.unpack
// The operands in this focused unit test are function arguments and therefore
// have no allocation provenance. These assumptions model the alignment
// guarantees that allocations carry in the full lowering pipeline.
func.func @lower_layout_ops(
    %src: memref<32x512xf16, strided<[512, 1], offset: ?>, 1>,
    %packed: memref<1x16x16x32x2xf16, 1>,
    %acc: memref<16x32x2xf16, 1>,
    %dst: memref<32x32xf32, strided<[512, 1], offset: ?>, 1>) {
  %src_aligned = memref.assume_alignment %src, 128 : memref<32x512xf16, strided<[512, 1], offset: ?>, 1>
  %packed_aligned = memref.assume_alignment %packed, 2048 : memref<1x16x16x32x2xf16, 1>
  %acc_aligned = memref.assume_alignment %acc, 2048 : memref<16x32x2xf16, 1>
  %dst_aligned = memref.assume_alignment %dst, 128 : memref<32x32xf32, strided<[512, 1], offset: ?>, 1>
  iree_hexagon.hmx.pack ins(%src_aligned : memref<32x512xf16, strided<[512, 1], offset: ?>, 1>)
      outs(%packed_aligned : memref<1x16x16x32x2xf16, 1>) {dim = 0 : i64}
  iree_hexagon.hmx.unpack ins(%acc_aligned : memref<16x32x2xf16, 1>)
      outs(%dst_aligned : memref<32x32xf32, strided<[512, 1], offset: ?>, 1>) {dim = 0 : i64}
  return
}

// -----

// This mirrors the normal pipeline: alignment comes from the allocations and
// the dynamically selected 32x32 tile, without memref.assume_alignment.
// CHECK-LABEL: func.func @lower_aligned_tiled_unpack(
// CHECK:       call @iree_hexagon_hmx_unpack_acc_f16_to_f32(%{{.+}}, %{{.+}}, %{{.+}}, %{{.+}}, %{{.+}}, %{{.+}}, %{{.+}}) : (i32, i32, i32, i32, i32, i32, i32) -> ()
// CHECK-NOT:   iree_hexagon.hmx.unpack
func.func @lower_aligned_tiled_unpack(%row_tile: index, %col_tile: index) {
  %c32 = arith.constant 32 : index
  %row = arith.muli %row_tile, %c32 : index
  %col = arith.muli %col_tile, %c32 : index
  %acc_grid = hexagonmem.alloc() {alignment = 2048 : i64} : memref<16x16x16x32x2xf16, 1>
  %dst = hexagonmem.alloc() : memref<512x512xf32, 1>
  %acc_tile = memref.subview %acc_grid[%row_tile, %col_tile, 0, 0, 0] [1, 1, 16, 32, 2] [1, 1, 1, 1, 1]
      : memref<16x16x16x32x2xf16, 1> to memref<16x32x2xf16, strided<[64, 2, 1], offset: ?>, 1>
  %dst_tile = memref.subview %dst[%row, %col] [32, 32] [1, 1]
      : memref<512x512xf32, 1> to memref<32x32xf32, strided<[512, 1], offset: ?>, 1>
  iree_hexagon.hmx.unpack ins(%acc_tile : memref<16x32x2xf16, strided<[64, 2, 1], offset: ?>, 1>)
      outs(%dst_tile : memref<32x32xf32, strided<[512, 1], offset: ?>, 1>)
      {dim = 0 : i64}
  return
}

// -----

// Value-bounds analysis proves that the dynamic logical source occupies at
// most the statically allocated 2x2 physical grid.
// CHECK-LABEL: func.func @lower_bounded_dynamic_pack(
// CHECK:       call @iree_hexagon_hmx_pack_f16
// CHECK-NOT:   iree_hexagon.hmx.pack
func.func @lower_bounded_dynamic_pack(%requested_rows: index) {
  %rows = affine.min affine_map<(d0) -> (d0, 64)>(%requested_rows)
  %source_storage = hexagonmem.alloc() : memref<64x64xf16, 1>
  %source = memref.subview %source_storage[0, 0] [%rows, 64] [1, 1]
      : memref<64x64xf16, 1>
      to memref<?x64xf16, strided<[64, 1]>, 1>
  %packed = hexagonmem.alloc() {alignment = 2048 : i64}
      : memref<2x2x16x32x2xf16, 1>
  iree_hexagon.hmx.pack ins(%source : memref<?x64xf16, strided<[64, 1]>, 1>)
      outs(%packed : memref<2x2x16x32x2xf16, 1>) {dim = 0 : i64}
  return
}

// -----

// A dynamic boundary tile remains legal for rank-3 unpack when its extent is
// provably no larger than the one 32x32 physical tile.
// CHECK-LABEL: func.func @lower_bounded_dynamic_rank3_unpack(
// CHECK:       call @iree_hexagon_hmx_unpack_acc_f16_to_f32
// CHECK-NOT:   iree_hexagon.hmx.unpack
func.func @lower_bounded_dynamic_rank3_unpack(%requested_cols: index) {
  %cols = affine.min affine_map<(d0) -> (d0, 32)>(%requested_cols)
  %source = hexagonmem.alloc() {alignment = 2048 : i64}
      : memref<16x32x2xf16, 1>
  %dest_storage = hexagonmem.alloc() : memref<32x32xf32, 1>
  %dest = memref.subview %dest_storage[0, 0] [32, %cols] [1, 1]
      : memref<32x32xf32, 1>
      to memref<32x?xf32, strided<[32, 1]>, 1>
  iree_hexagon.hmx.unpack ins(%source : memref<16x32x2xf16, 1>)
      outs(%dest : memref<32x?xf32, strided<[32, 1]>, 1>) {dim = 0 : i64}
  return
}

// -----

// A transpose-b operand pack (`dim = 1`, source stored N x K) lowers to the
// transposed runtime packer, which fuses the 32x32 transpose into the pack. The
// call takes (dest, src, stride, actual_interleave=K=src dim1, actual_other=N=src
// dim0, interleave_tiles=k=dest dim0, other_tiles=n=dest dim1).
// CHECK-LABEL: func.func @lower_transposed_pack(
// CHECK:       call @iree_hexagon_hmx_pack_transposed_f16(%{{.+}}, %{{.+}}, %{{.+}}, %{{.+}}, %{{.+}}, %{{.+}}, %{{.+}}) : (i32, i32, i32, i32, i32, i32, i32) -> ()
// CHECK-NOT:   call @iree_hexagon_hmx_pack_f16(
// CHECK-NOT:   iree_hexagon.hmx.pack
func.func @lower_transposed_pack(
    %src: memref<32x512xf16, strided<[512, 1], offset: ?>, 1>,
    %packed: memref<16x1x16x32x2xf16, 1>) {
  %src_aligned = memref.assume_alignment %src, 128 : memref<32x512xf16, strided<[512, 1], offset: ?>, 1>
  %packed_aligned = memref.assume_alignment %packed, 2048 : memref<16x1x16x32x2xf16, 1>
  iree_hexagon.hmx.pack ins(%src_aligned : memref<32x512xf16, strided<[512, 1], offset: ?>, 1>)
      outs(%packed_aligned : memref<16x1x16x32x2xf16, 1>) {dim = 1 : i64}
  return
}

// -----

// An f16 output matmul unpacks into an f16 destination: the accumulator read-out
// tile is f16, so the lowering selects the non-widening runtime unpack
// (de-interleave only) instead of the f16->f32 (widen + add) variant.
// CHECK-LABEL: func.func @lower_unpack_f16_dest(
// CHECK:       call @iree_hexagon_hmx_unpack_acc_f16_to_f16(%{{.+}}, %{{.+}}, %{{.+}}, %{{.+}}, %{{.+}}, %{{.+}}, %{{.+}}) : (i32, i32, i32, i32, i32, i32, i32) -> ()
// CHECK-NOT:   iree_hexagon.hmx.unpack
func.func @lower_unpack_f16_dest(
    %acc: memref<1x1x16x32x2xf16, 1>,
    %dst: memref<32x32xf16, strided<[512, 1], offset: ?>, 1>) {
  %acc_aligned = memref.assume_alignment %acc, 2048 : memref<1x1x16x32x2xf16, 1>
  %dst_aligned = memref.assume_alignment %dst, 128 : memref<32x32xf16, strided<[512, 1], offset: ?>, 1>
  iree_hexagon.hmx.unpack ins(%acc_aligned : memref<1x1x16x32x2xf16, 1>)
      outs(%dst_aligned : memref<32x32xf16, strided<[512, 1], offset: ?>, 1>)
      {dim = 0 : i64}
  return
}

// -----

// The rank-5 accumulator grid unpacks with a single call; the per-tile loop and
// boundary clamping live in the runtime, not in MLIR.
// CHECK-LABEL: func.func @lower_unpack_grid(
// CHECK:       call @iree_hexagon_hmx_unpack_acc_f16_to_f32(%{{.+}}, %{{.+}}, %{{.+}}, %{{.+}}, %{{.+}}, %{{.+}}, %{{.+}}) : (i32, i32, i32, i32, i32, i32, i32) -> ()
// CHECK-NOT:   iree_hexagon.hmx.unpack
// CHECK-NOT:   scf.for
func.func @lower_unpack_grid(
    %acc: memref<2x3x16x32x2xf16, 1>,
    %dst: memref<64x96xf32, strided<[128, 1], offset: ?>, 1>) {
  %acc_aligned = memref.assume_alignment %acc, 2048 : memref<2x3x16x32x2xf16, 1>
  %dst_aligned = memref.assume_alignment %dst, 128 : memref<64x96xf32, strided<[128, 1], offset: ?>, 1>
  iree_hexagon.hmx.unpack ins(%acc_aligned : memref<2x3x16x32x2xf16, 1>)
      outs(%dst_aligned : memref<64x96xf32, strided<[128, 1], offset: ?>, 1>)
      {dim = 0 : i64}
  return
}

// -----

// CHECK-LABEL: func.func @lower_acc_setup_read(
// CHECK:       call @iree_hexagon_hmx_acc_setup_read_f16(%{{.+}}) : (i32) -> ()
// CHECK-NOT:   iree_hexagon.hmx.acc.setup_read
func.func @lower_acc_setup_read(%config: memref<256xi8, 1>) {
  %aligned = memref.assume_alignment %config, 2048 : memref<256xi8, 1>
  iree_hexagon.hmx.acc.setup_read %aligned : memref<256xi8, 1>
  return
}

// -----

// CHECK-LABEL: func.func @lower_direct_single_mma(
// CHECK:       call @iree_hexagon_hmx_acc_clear_f16() : () -> ()
// CHECK:       call @iree_hexagon_hmx_mma_f16(%{{.+}}, %{{.+}}) : (i32, i32) -> ()
// CHECK:       call @iree_hexagon_hmx_acc_read_f16(%{{.+}}) : (i32) -> ()
// CHECK-NOT:   iree_hexagon.hmx.acc.zero
// CHECK-NOT:   iree_hexagon.hmx.mma
// CHECK-NOT:   iree_hexagon.hmx.acc.read
func.func @lower_direct_single_mma(
    %lhs: memref<1x1x16x32x2xf16, 1>,
    %rhs: memref<1x1x16x32x2xf16, 1>,
    %dst: memref<16x32x2xf16, 1>) {
  %lhs_aligned = memref.assume_alignment %lhs, 2048 : memref<1x1x16x32x2xf16, 1>
  %rhs_aligned = memref.assume_alignment %rhs, 2048 : memref<1x1x16x32x2xf16, 1>
  %dst_aligned = memref.assume_alignment %dst, 2048 : memref<16x32x2xf16, 1>
  %acc0 = iree_hexagon.hmx.acc.zero : !iree_hexagon.hmx.acc<32x32xf32>
  %lhs_tile = memref.subview %lhs_aligned[0, 0, 0, 0, 0] [1, 1, 16, 32, 2] [1, 1, 1, 1, 1]
      : memref<1x1x16x32x2xf16, 1> to memref<1x1x16x32x2xf16, strided<[1024, 1024, 64, 2, 1]>, 1>
  %rhs_tile = memref.subview %rhs_aligned[0, 0, 0, 0, 0] [1, 1, 16, 32, 2] [1, 1, 1, 1, 1]
      : memref<1x1x16x32x2xf16, 1> to memref<1x1x16x32x2xf16, strided<[1024, 1024, 64, 2, 1]>, 1>
  %acc1 = iree_hexagon.hmx.mma %lhs_tile, %rhs_tile, %acc0
      : memref<1x1x16x32x2xf16, strided<[1024, 1024, 64, 2, 1]>, 1>,
        memref<1x1x16x32x2xf16, strided<[1024, 1024, 64, 2, 1]>, 1>,
        !iree_hexagon.hmx.acc<32x32xf32> -> !iree_hexagon.hmx.acc<32x32xf32>
  iree_hexagon.hmx.acc.read %acc1, %dst_aligned : !iree_hexagon.hmx.acc<32x32xf32>, memref<16x32x2xf16, 1>
  return
}

// -----

// CHECK-LABEL: func.func @lower_acc_loop(
// CHECK:       call @iree_hexagon_hmx_acc_clear_f16() : () -> ()
// CHECK-NOT:   !iree_hexagon.hmx.acc
// CHECK:       cf.br ^[[LOOP:bb[0-9]+]](%{{.+}} : index)
// CHECK:       ^[[LOOP]](%{{.+}}: index):
// CHECK:         cf.cond_br %{{.+}}, ^[[BODY:bb[0-9]+]], ^[[EXIT:bb[0-9]+]]{{$}}
// CHECK:       ^[[BODY]]:
// CHECK:         call @iree_hexagon_hmx_mma_f16(%{{.+}}, %{{.+}}) : (i32, i32) -> ()
// CHECK:         call @iree_hexagon_hmx_mma_f16(%{{.+}}, %{{.+}}) : (i32, i32) -> ()
// CHECK:         cf.br ^[[LOOP]](%{{.+}} : index)
// CHECK:       ^[[EXIT]]:
// CHECK:       call @iree_hexagon_hmx_acc_read_f16(%{{.+}}) : (i32) -> ()
// CHECK-NOT:   !iree_hexagon.hmx.acc
func.func @lower_acc_loop(
    %lhs: memref<1x16x16x32x2xf16, 1>,
    %rhs: memref<16x1x16x32x2xf16, 1>,
    %dst: memref<16x32x2xf16, 1>) {
  %lhs_aligned = memref.assume_alignment %lhs, 2048 : memref<1x16x16x32x2xf16, 1>
  %rhs_aligned = memref.assume_alignment %rhs, 2048 : memref<16x1x16x32x2xf16, 1>
  %dst_aligned = memref.assume_alignment %dst, 2048 : memref<16x32x2xf16, 1>
  %c0 = arith.constant 0 : index
  %c512 = arith.constant 512 : index
  %c32 = arith.constant 32 : index
  %acc0 = iree_hexagon.hmx.acc.zero : !iree_hexagon.hmx.acc<32x32xf32>
  cf.br ^loop(%c0, %acc0 : index, !iree_hexagon.hmx.acc<32x32xf32>)
^loop(%k: index, %iter: !iree_hexagon.hmx.acc<32x32xf32>):
  %continue = arith.cmpi slt, %k, %c512 : index
  cf.cond_br %continue, ^body, ^exit(%iter : !iree_hexagon.hmx.acc<32x32xf32>)
^body:
    %kt = arith.divui %k, %c32 : index
    %lhs_tile = memref.subview %lhs_aligned[0, %kt, 0, 0, 0] [1, 1, 16, 32, 2] [1, 1, 1, 1, 1]
        : memref<1x16x16x32x2xf16, 1> to memref<1x1x16x32x2xf16, strided<[16384, 1024, 64, 2, 1], offset: ?>, 1>
    %rhs_tile = memref.subview %rhs_aligned[%kt, 0, 0, 0, 0] [1, 1, 16, 32, 2] [1, 1, 1, 1, 1]
        : memref<16x1x16x32x2xf16, 1> to memref<1x1x16x32x2xf16, strided<[1024, 1024, 64, 2, 1], offset: ?>, 1>
    %next0 = iree_hexagon.hmx.mma %lhs_tile, %rhs_tile, %iter
        : memref<1x1x16x32x2xf16, strided<[16384, 1024, 64, 2, 1], offset: ?>, 1>,
          memref<1x1x16x32x2xf16, strided<[1024, 1024, 64, 2, 1], offset: ?>, 1>,
          !iree_hexagon.hmx.acc<32x32xf32> -> !iree_hexagon.hmx.acc<32x32xf32>
    %next1 = iree_hexagon.hmx.mma %lhs_tile, %rhs_tile, %next0
        : memref<1x1x16x32x2xf16, strided<[16384, 1024, 64, 2, 1], offset: ?>, 1>,
          memref<1x1x16x32x2xf16, strided<[1024, 1024, 64, 2, 1], offset: ?>, 1>,
          !iree_hexagon.hmx.acc<32x32xf32> -> !iree_hexagon.hmx.acc<32x32xf32>
    %next_k = arith.addi %k, %c32 : index
    cf.br ^loop(%next_k, %next1 : index, !iree_hexagon.hmx.acc<32x32xf32>)
^exit(%acc: !iree_hexagon.hmx.acc<32x32xf32>):
  iree_hexagon.hmx.acc.read %acc, %dst_aligned : !iree_hexagon.hmx.acc<32x32xf32>, memref<16x32x2xf16, 1>
  return
}

// -----

// CHECK-LABEL: func.func @lower_first_function()
// CHECK:         call @iree_hexagon_hmx_acc_clear_f16() : () -> ()
func.func @lower_first_function() {
  %acc = iree_hexagon.hmx.acc.zero : !iree_hexagon.hmx.acc<32x32xf32>
  return
}

// CHECK-LABEL: func.func @lower_second_function()
// CHECK:         call @iree_hexagon_hmx_acc_clear_f16() : () -> ()
func.func @lower_second_function() {
  %acc = iree_hexagon.hmx.acc.zero : !iree_hexagon.hmx.acc<32x32xf32>
  return
}

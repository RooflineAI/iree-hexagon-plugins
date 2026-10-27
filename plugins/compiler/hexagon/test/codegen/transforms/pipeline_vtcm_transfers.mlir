// RUN: iree-opt --pass-pipeline="builtin.module(func.func(iree-hexagon-pipeline-vtcm-transfers, canonicalize, cse))" --split-input-file %s | FileCheck %s
// RUN: iree-opt --pass-pipeline="builtin.module(func.func(iree-hexagon-pipeline-vtcm-transfers{outer-tile-loops=true}, canonicalize, cse))" --split-input-file %s | FileCheck %s --check-prefix=OUTER

// Two loads and a store become DMAs into and out of two buffer slots. The
// loop is rotated: the loads of the first tile are issued before it, each
// iteration prefetches the next tile, waits for its own loads, computes,
// issues its store and waits for the store of the previous tile. The compute
// is not duplicated.

#id = affine_map<(d0, d1) -> (d0, d1)>
// CHECK-DAG: #[[$UNROLL:.+]] = #llvm.loop_unroll<disable = true>
// CHECK-DAG: #[[$SLOT:.+]] = affine_map<(d0) -> ((d0 floordiv 128) mod 2)>
// CHECK-DAG: #[[$ANNOTATION:.+]] = #llvm.loop_annotation<unroll = #[[$UNROLL]]>
// CHECK-LABEL: func.func @loads_and_store(
//  CHECK-SAME:     %[[A:[a-zA-Z0-9]+]]: memref<2048x512xf32>, %[[B:[a-zA-Z0-9]+]]: memref<2048x512xf32>, %[[C:[a-zA-Z0-9]+]]: memref<2048x512xf32>
//   CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//   CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
//   CHECK-DAG:   %[[C128:.+]] = arith.constant 128 : index
//   CHECK-DAG:   %[[C2048:.+]] = arith.constant 2048 : index
//   CHECK-DAG:   %[[N:.+]] = arith.constant 65536 : index
//   CHECK-DAG:   %[[TAG_A:.+]] = memref.alloca() : memref<2xi32>
//   CHECK-DAG:   %[[TAG_B:.+]] = memref.alloca() : memref<2xi32>
//   CHECK-DAG:   %[[TAG_C:.+]] = memref.alloca() : memref<2xi32>
//   CHECK-DAG:   %[[BUF_A:.+]] = memref.alloc() : memref<2x128x512xf32, 1>
//   CHECK-DAG:   %[[BUF_B:.+]] = memref.alloc() : memref<2x128x512xf32, 1>
//   CHECK-DAG:   %[[BUF_C:.+]] = memref.alloc() : memref<2x128x512xf32, 1>
// Prologue: the loads of the first tile into slot 0.
//   CHECK-DAG:   %[[A0:.+]] = memref.subview %[[A]][0, 0] [128, 512] [1, 1]
//   CHECK-DAG:   %[[A0_SLOT:.+]] = memref.subview %[[BUF_A]][0, 0, 0] [1, 128, 512] [1, 1, 1]
//       CHECK:   memref.dma_start %[[A0]][%[[C0]], %[[C0]]], %[[A0_SLOT]][%[[C0]], %[[C0]]], %[[N]], %[[TAG_A]][%[[C0]]]
//       CHECK:   memref.dma_start %{{.+}}[%[[C0]], %[[C0]]], %{{.+}}[%[[C0]], %[[C0]]], %[[N]], %[[TAG_B]][%[[C0]]]
//       CHECK:   scf.for %[[T:.+]] = %[[C0]] to %[[C2048]] step %[[C128]] {
// Head: prefetch the next tile.
//       CHECK:     %[[NEXT:.+]] = arith.addi %[[T]], %[[C128]]
//       CHECK:     %[[HAS_NEXT:.+]] = arith.cmpi slt, %[[NEXT]], %[[C2048]]
//       CHECK:     scf.if %[[HAS_NEXT]] {
//       CHECK:       %[[NEXT_SLOT:.+]] = affine.apply #[[$SLOT]](%[[NEXT]])
//   CHECK-DAG:       %[[A_NEXT:.+]] = memref.subview %[[A]][%[[NEXT]], 0] [128, 512] [1, 1]
//   CHECK-DAG:       %[[A_NEXT_SLOT:.+]] = memref.subview %[[BUF_A]][%[[NEXT_SLOT]], 0, 0] [1, 128, 512] [1, 1, 1]
//       CHECK:       memref.dma_start %[[A_NEXT]][%[[C0]], %[[C0]]], %[[A_NEXT_SLOT]][%[[C0]], %[[C0]]], %[[N]], %[[TAG_A]][%[[NEXT_SLOT]]]
//       CHECK:       memref.dma_start %{{.+}}, %[[TAG_B]][%[[NEXT_SLOT]]]
//       CHECK:     }
// Wait for the loads of this tile, compute once, issue the store.
//       CHECK:     %[[CUR_SLOT:.+]] = affine.apply #[[$SLOT]](%[[T]])
//       CHECK:     memref.dma_wait %[[TAG_A]][%[[CUR_SLOT]]], %[[N]]
//       CHECK:     memref.dma_wait %[[TAG_B]][%[[CUR_SLOT]]], %[[N]]
//   CHECK-DAG:     %[[C_CUR_SLOT:.+]] = memref.subview %[[BUF_C]][%[[CUR_SLOT]], 0, 0]
//   CHECK-DAG:     %[[B_CUR_SLOT:.+]] = memref.subview %[[BUF_B]][%[[CUR_SLOT]], 0, 0]
//   CHECK-DAG:     %[[A_CUR_SLOT:.+]] = memref.subview %[[BUF_A]][%[[CUR_SLOT]], 0, 0]
//       CHECK:     linalg.generic
//  CHECK-SAME:       ins(%[[A_CUR_SLOT]], %[[B_CUR_SLOT]] :
//  CHECK-SAME:       outs(%[[C_CUR_SLOT]] :
//       CHECK:     %[[C_CUR:.+]] = memref.subview %[[C]][%[[T]], 0] [128, 512] [1, 1]
//       CHECK:     memref.dma_start %[[C_CUR_SLOT]][%[[C0]], %[[C0]]], %[[C_CUR]][%[[C0]], %[[C0]]], %[[N]], %[[TAG_C]][%[[CUR_SLOT]]]
// Tail: wait for the store of the previous tile.
//       CHECK:     %[[HAS_PREV:.+]] = arith.cmpi sgt, %[[T]], %[[C0]]
//       CHECK:     scf.if %[[HAS_PREV]] {
//       CHECK:       %[[PREV:.+]] = arith.subi %[[T]], %[[C128]]
//       CHECK:       %[[PREV_SLOT:.+]] = affine.apply #[[$SLOT]](%[[PREV]])
//       CHECK:       memref.dma_wait %[[TAG_C]][%[[PREV_SLOT]]], %[[N]]
//       CHECK:     }
//       CHECK:   } {llvm.loop_annotation = #[[$ANNOTATION]]}
// Epilogue: the store of the last tile (slot 1).
//       CHECK:   memref.dma_wait %[[TAG_C]][%[[C1]]], %[[N]]
//   CHECK-NOT:   linalg.generic
//   CHECK-NOT:   memref.copy
//   CHECK-NOT:   iree_hexagon.vtcm_tile_loop
func.func @loads_and_store(%a: memref<2048x512xf32>, %b: memref<2048x512xf32>, %c: memref<2048x512xf32>) {
  %c0 = arith.constant 0 : index
  %c128 = arith.constant 128 : index
  %c2048 = arith.constant 2048 : index
  %in0 = memref.alloc() : memref<128x512xf32, 1>
  %in1 = memref.alloc() : memref<128x512xf32, 1>
  %out = memref.alloc() : memref<128x512xf32, 1>
  scf.for %t = %c0 to %c2048 step %c128 {
    %sa = memref.subview %a[%t, 0] [128, 512] [1, 1] : memref<2048x512xf32> to memref<128x512xf32, strided<[512, 1], offset: ?>>
    %sb = memref.subview %b[%t, 0] [128, 512] [1, 1] : memref<2048x512xf32> to memref<128x512xf32, strided<[512, 1], offset: ?>>
    memref.copy %sa, %in0 : memref<128x512xf32, strided<[512, 1], offset: ?>> to memref<128x512xf32, 1>
    memref.copy %sb, %in1 : memref<128x512xf32, strided<[512, 1], offset: ?>> to memref<128x512xf32, 1>
    linalg.generic {indexing_maps = [#id, #id, #id], iterator_types = ["parallel", "parallel"]}
        ins(%in0, %in1 : memref<128x512xf32, 1>, memref<128x512xf32, 1>) outs(%out : memref<128x512xf32, 1>) {
    ^bb0(%x: f32, %y: f32, %o: f32):
      %s = arith.addf %x, %y : f32
      linalg.yield %s : f32
    }
    %sc = memref.subview %c[%t, 0] [128, 512] [1, 1] : memref<2048x512xf32> to memref<128x512xf32, strided<[512, 1], offset: ?>>
    memref.copy %out, %sc : memref<128x512xf32, 1> to memref<128x512xf32, strided<[512, 1], offset: ?>>
  } {iree_hexagon.vtcm_tile_loop}
  memref.dealloc %in0 : memref<128x512xf32, 1>
  memref.dealloc %in1 : memref<128x512xf32, 1>
  memref.dealloc %out : memref<128x512xf32, 1>
  return
}

// -----

// Tiles that are not contiguous in DDR become 2D DMAs: rows of 256 elements,
// 512 elements apart in DDR.

// CHECK-LABEL: func.func @strided_tiles(
//   CHECK-DAG:   %[[C256:.+]] = arith.constant 256 : index
//   CHECK-DAG:   %[[C512:.+]] = arith.constant 512 : index
//       CHECK:   memref.dma_start {{.+}}, %[[C512]], %[[C256]] : memref<256x256xf32, strided<[512, 1]>>, memref<256x256xf32, strided<[256, 1]>, 1>
//       CHECK:   scf.for
//       CHECK:     scf.if
//       CHECK:       memref.dma_start {{.+}}, %[[C512]], %[[C256]] :
//       CHECK:     memref.dma_wait
//       CHECK:     memref.dma_start {{.+}}, %[[C512]], %[[C256]] : memref<256x256xf32, strided<[256, 1], offset: ?>, 1>, memref<256x256xf32, strided<[512, 1], offset: ?>>
//   CHECK-NOT:   memref.copy
func.func @strided_tiles(%a: memref<2048x512xf32>, %c: memref<2048x512xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c16 = arith.constant 16 : index
  %c2 = arith.constant 2 : index
  %in = memref.alloc() : memref<256x256xf32, 1>
  %out = memref.alloc() : memref<256x256xf32, 1>
  scf.for %t = %c0 to %c16 step %c1 {
    %idx:2 = affine.delinearize_index %t into (8, 2) : index, index
    %row = affine.apply affine_map<(d0) -> (d0 * 256)>(%idx#0)
    %col = affine.apply affine_map<(d0) -> (d0 * 256)>(%idx#1)
    %sa = memref.subview %a[%row, %col] [256, 256] [1, 1] : memref<2048x512xf32> to memref<256x256xf32, strided<[512, 1], offset: ?>>
    memref.copy %sa, %in : memref<256x256xf32, strided<[512, 1], offset: ?>> to memref<256x256xf32, 1>
    linalg.copy ins(%in : memref<256x256xf32, 1>) outs(%out : memref<256x256xf32, 1>)
    %sc = memref.subview %c[%row, %col] [256, 256] [1, 1] : memref<2048x512xf32> to memref<256x256xf32, strided<[512, 1], offset: ?>>
    memref.copy %out, %sc : memref<256x256xf32, 1> to memref<256x256xf32, strided<[512, 1], offset: ?>>
  } {iree_hexagon.vtcm_tile_loop}
  return
}

// -----

// A buffer that is loaded and stored (the tile is updated in place) uses the
// same slot for both transfers, each with its own tags.

// CHECK-LABEL: func.func @in_out(
//   CHECK-DAG:   %[[LOAD_TAG:.+]] = memref.alloca() : memref<2xi32>
//   CHECK-DAG:   %[[STORE_TAG:.+]] = memref.alloca() : memref<2xi32>
//       CHECK:   %[[BUF:.+]] = memref.alloc() : memref<2x128x512xf32, 1>
//   CHECK-NOT:   memref.alloc()
//       CHECK:   scf.for
//       CHECK:     scf.if
//       CHECK:       memref.dma_start {{.+}}, %[[LOAD_TAG]]
//       CHECK:     }
//       CHECK:     %[[SLOT:.+]] = affine.apply
//       CHECK:     memref.dma_wait %[[LOAD_TAG]][%[[SLOT]]]
//       CHECK:     %[[VIEW:.+]] = memref.subview %[[BUF]][%[[SLOT]], 0, 0]
//       CHECK:     linalg.generic {{.+}} outs(%[[VIEW]] :
//       CHECK:     memref.dma_start %[[VIEW]]{{.+}}, %[[STORE_TAG]][%[[SLOT]]]
#id = affine_map<(d0, d1) -> (d0, d1)>
func.func @in_out(%c: memref<2048x512xf32>) {
  %c0 = arith.constant 0 : index
  %c128 = arith.constant 128 : index
  %c2048 = arith.constant 2048 : index
  %buf = memref.alloc() : memref<128x512xf32, 1>
  scf.for %t = %c0 to %c2048 step %c128 {
    %sc = memref.subview %c[%t, 0] [128, 512] [1, 1] : memref<2048x512xf32> to memref<128x512xf32, strided<[512, 1], offset: ?>>
    memref.copy %sc, %buf : memref<128x512xf32, strided<[512, 1], offset: ?>> to memref<128x512xf32, 1>
    linalg.generic {indexing_maps = [#id], iterator_types = ["parallel", "parallel"]} outs(%buf : memref<128x512xf32, 1>) {
    ^bb0(%o: f32):
      %s = arith.addf %o, %o : f32
      linalg.yield %s : f32
    }
    memref.copy %buf, %sc : memref<128x512xf32, 1> to memref<128x512xf32, strided<[512, 1], offset: ?>>
  } {iree_hexagon.vtcm_tile_loop}
  return
}

// -----

// A single tile has nothing to overlap.

// CHECK-LABEL: func.func @single_iteration(
//   CHECK-NOT:   memref.dma_start
//       CHECK:   memref.copy
//   CHECK-NOT:   iree_hexagon.vtcm_tile_loop
func.func @single_iteration(%a: memref<128x512xf32>) {
  %c0 = arith.constant 0 : index
  %c128 = arith.constant 128 : index
  %buf = memref.alloc() : memref<128x512xf32, 1>
  scf.for %t = %c0 to %c128 step %c128 {
    %sa = memref.subview %a[%t, 0] [128, 512] [1, 1] : memref<128x512xf32> to memref<128x512xf32, strided<[512, 1], offset: ?>>
    memref.copy %sa, %buf : memref<128x512xf32, strided<[512, 1], offset: ?>> to memref<128x512xf32, 1>
    linalg.copy ins(%buf : memref<128x512xf32, 1>) outs(%buf : memref<128x512xf32, 1>)
  } {iree_hexagon.vtcm_tile_loop}
  return
}

// -----

// A row stride of 128 KiB does not fit the 16 bits of a 2D DMA descriptor;
// the runtime would copy row by row, synchronously. The loop is left as is.

// CHECK-LABEL: func.func @descriptor_field_overflow(
//   CHECK-NOT:   memref.dma_start
//       CHECK:   memref.copy
func.func @descriptor_field_overflow(%a: memref<64x32768xf32>) {
  %c0 = arith.constant 0 : index
  %c8 = arith.constant 8 : index
  %c64 = arith.constant 64 : index
  %buf = memref.alloc() : memref<8x4096xf32, 1>
  scf.for %t = %c0 to %c64 step %c8 {
    %sa = memref.subview %a[%t, 0] [8, 4096] [1, 1] : memref<64x32768xf32> to memref<8x4096xf32, strided<[32768, 1], offset: ?>>
    memref.copy %sa, %buf : memref<8x4096xf32, strided<[32768, 1], offset: ?>> to memref<8x4096xf32, 1>
    linalg.copy ins(%buf : memref<8x4096xf32, 1>) outs(%buf : memref<8x4096xf32, 1>)
  } {iree_hexagon.vtcm_tile_loop}
  return
}

// -----

// Another DDR -> VTCM copy in the loop would stay synchronous and wait for the
// prefetch on the in-order DMA engine. The loop is left as is.

// CHECK-LABEL: func.func @remaining_synchronous_copy(
//   CHECK-NOT:   memref.dma_start
//       CHECK:   memref.copy
//       CHECK:   memref.copy
func.func @remaining_synchronous_copy(%a: memref<2048x512xf32>, %b: memref<128x512xf32>, %cond: i1) {
  %c0 = arith.constant 0 : index
  %c128 = arith.constant 128 : index
  %c2048 = arith.constant 2048 : index
  %buf = memref.alloc() : memref<128x512xf32, 1>
  %other = memref.alloc() : memref<128x512xf32, 1>
  scf.for %t = %c0 to %c2048 step %c128 {
    %sa = memref.subview %a[%t, 0] [128, 512] [1, 1] : memref<2048x512xf32> to memref<128x512xf32, strided<[512, 1], offset: ?>>
    memref.copy %sa, %buf : memref<128x512xf32, strided<[512, 1], offset: ?>> to memref<128x512xf32, 1>
    scf.if %cond {
      memref.copy %b, %other : memref<128x512xf32> to memref<128x512xf32, 1>
    }
    linalg.copy ins(%buf : memref<128x512xf32, 1>) outs(%other : memref<128x512xf32, 1>)
  } {iree_hexagon.vtcm_tile_loop}
  return
}

// -----

// DMA tokens are only valid on the issuing thread; a tile loop inside a thread
// loop is left as is.

// CHECK-LABEL: func.func @inside_thread_loop(
//   CHECK-NOT:   memref.dma_start
//       CHECK:   memref.copy
func.func @inside_thread_loop(%a: memref<2048x512xf32>) {
  %c0 = arith.constant 0 : index
  %c128 = arith.constant 128 : index
  %c1024 = arith.constant 1024 : index
  scf.forall (%chunk) in (2) {
    %buf = memref.alloc() : memref<128x512xf32, 1>
    scf.for %t = %c0 to %c1024 step %c128 {
      %sa = memref.subview %a[%t, 0] [128, 512] [1, 1] : memref<2048x512xf32> to memref<128x512xf32, strided<[512, 1], offset: ?>>
      memref.copy %sa, %buf : memref<128x512xf32, strided<[512, 1], offset: ?>> to memref<128x512xf32, 1>
      linalg.copy ins(%buf : memref<128x512xf32, 1>) outs(%buf : memref<128x512xf32, 1>)
    } {iree_hexagon.vtcm_tile_loop}
    memref.dealloc %buf : memref<128x512xf32, 1>
  } {mapping = [#iree_codegen.local_mapping<0>]}
  return
}

// -----

// Ragged tiles (81 rows in tiles of 40): the copies stage the valid rows
// through subviews of the buffers, and the number of rows of a DMA is only
// known at runtime. The waits give the size of the full buffer. The view of
// the output buffer that is hoisted out of the inner loop is moved back into
// it, so that it takes the slot of the iteration.

// CHECK-LABEL: func.func @ragged_tiles(
//  CHECK-SAME:     %[[A:[a-zA-Z0-9]+]]: memref<81x2048xf16>, %[[B:[a-zA-Z0-9]+]]: memref<81x6144xf16>, %[[C:[a-zA-Z0-9]+]]: memref<81x6144xf16>
//   CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//   CHECK-DAG:   %[[C192:.+]] = arith.constant 192 : index
//   CHECK-DAG:   %[[C6144:.+]] = arith.constant 6144 : index
//   CHECK-DAG:   %[[MAX:.+]] = arith.constant 7680 : index
//   CHECK-DAG:   %[[A_BUF:.+]] = memref.alloc() : memref<40x2048xf16, 1>
//   CHECK-DAG:   %[[B_BUF:.+]] = memref.alloc() : memref<2x40x192xf16, 1>
//   CHECK-DAG:   %[[C_BUF:.+]] = memref.alloc() : memref<2x40x192xf16, 1>
//       CHECK:   scf.for %[[M:.+]] = %[[C0]] to
//       CHECK:     %[[ROWS:.+]] = affine.min
//       CHECK:     memref.copy
//       CHECK:     %[[N_ELEMS:.+]] = arith.muli %[[ROWS]], %[[C192]]
//       CHECK:     memref.dma_start %{{.+}}[%[[C0]], %[[C0]]], %{{.+}}[%[[C0]], %[[C0]]], %[[N_ELEMS]], %{{.+}}[%[[C0]]], %[[C6144]], %[[C192]] : memref<?x192xf16, strided<[6144, 1], offset: ?>>, memref<?x192xf16, strided<[192, 1]>, 1>
//       CHECK:     scf.for %[[N:.+]] = %[[C0]] to %[[C6144]] step %[[C192]] {
//       CHECK:       scf.if
//       CHECK:         memref.subview %[[B]][%[[M]], %{{.+}}] [%[[ROWS]], 192]
//       CHECK:         memref.dma_start {{.+}}, %[[N_ELEMS]], {{.+}}, %[[C6144]], %[[C192]] :
//       CHECK:       }
//       CHECK:       %[[SLOT:.+]] = affine.apply
//       CHECK:       memref.dma_wait %{{.+}}[%[[SLOT]]], %[[MAX]]
//       CHECK:       %[[C_SLOT:.+]] = memref.subview %[[C_BUF]][%[[SLOT]], 0, 0] [1, 40, 192]
//       CHECK:       %[[C_TILE:.+]] = memref.subview %[[C_SLOT]][0, 0] [%[[ROWS]], 192]
//       CHECK:       linalg.copy {{.+}} outs(%[[C_TILE]] :
//       CHECK:       %[[C_DDR:.+]] = memref.subview %[[C]][%[[M]], %[[N]]] [%[[ROWS]], 192]
//       CHECK:       memref.dma_start %[[C_TILE]][%[[C0]], %[[C0]]], %[[C_DDR]][%[[C0]], %[[C0]]], %[[N_ELEMS]], %{{.+}}[%[[SLOT]]], %[[C6144]], %[[C192]]
//       CHECK:     } {llvm.loop_annotation
//       CHECK:     memref.dma_wait %{{.+}}, %[[MAX]]
//   CHECK-NOT:   memref.copy

// OUTER-LABEL: func.func @ragged_tiles(
//   OUTER-DAG:   %[[C0:.+]] = arith.constant 0 : index
//   OUTER-DAG:   %[[C2048:.+]] = arith.constant 2048 : index
//   OUTER-DAG:   %[[FIRST:.+]] = arith.constant 81920 : index
//   OUTER-DAG:   %[[A_BUF:.+]] = memref.alloc() : memref<2x40x2048xf16, 1>
//       OUTER:   memref.dma_start {{.+}}, %[[FIRST]], {{.+}} : memref<40x2048xf16, strided<[2048, 1]>>, memref<40x2048xf16, strided<[2048, 1]>, 1>
//       OUTER:   scf.for
//       OUTER:     scf.if
//       OUTER:       %[[NEXT_ROWS:.+]] = affine.min
//       OUTER:       %[[NEXT_ELEMS:.+]] = arith.muli %[[NEXT_ROWS]], %[[C2048]]
//       OUTER:       memref.dma_start {{.+}}, %[[NEXT_ELEMS]], {{.+}} : memref<?x2048xf16, strided<[2048, 1], offset: ?>>, memref<?x2048xf16, strided<[2048, 1], offset: ?>, 1>
//       OUTER:     }
//       OUTER:     memref.dma_wait %{{.+}}, %[[FIRST]]
//       OUTER:     scf.for
//       OUTER:     } {llvm.loop_annotation
//       OUTER:   } {llvm.loop_annotation
//   OUTER-NOT:   memref.copy
func.func @ragged_tiles(%a: memref<81x2048xf16>, %b: memref<81x6144xf16>, %c: memref<81x6144xf16>) {
  %c0 = arith.constant 0 : index
  %c40 = arith.constant 40 : index
  %c81 = arith.constant 81 : index
  %c192 = arith.constant 192 : index
  %c6144 = arith.constant 6144 : index
  %a_buf = memref.alloc() : memref<40x2048xf16, 1>
  %b_buf = memref.alloc() : memref<40x192xf16, 1>
  %c_buf = memref.alloc() : memref<40x192xf16, 1>
  scf.for %m = %c0 to %c81 step %c40 {
    %rows = affine.min affine_map<(d0) -> (-d0 + 81, 40)>(%m)
    %sa = memref.subview %a[%m, 0] [%rows, 2048] [1, 1] : memref<81x2048xf16> to memref<?x2048xf16, strided<[2048, 1], offset: ?>>
    %va = memref.subview %a_buf[0, 0] [%rows, 2048] [1, 1] : memref<40x2048xf16, 1> to memref<?x2048xf16, strided<[2048, 1]>, 1>
    memref.copy %sa, %va : memref<?x2048xf16, strided<[2048, 1], offset: ?>> to memref<?x2048xf16, strided<[2048, 1]>, 1>
    %vc = memref.subview %c_buf[0, 0] [%rows, 192] [1, 1] : memref<40x192xf16, 1> to memref<?x192xf16, strided<[192, 1]>, 1>
    scf.for %n = %c0 to %c6144 step %c192 {
      %sb = memref.subview %b[%m, %n] [%rows, 192] [1, 1] : memref<81x6144xf16> to memref<?x192xf16, strided<[6144, 1], offset: ?>>
      %vb = memref.subview %b_buf[0, 0] [%rows, 192] [1, 1] : memref<40x192xf16, 1> to memref<?x192xf16, strided<[192, 1]>, 1>
      memref.copy %sb, %vb : memref<?x192xf16, strided<[6144, 1], offset: ?>> to memref<?x192xf16, strided<[192, 1]>, 1>
      linalg.copy ins(%vb : memref<?x192xf16, strided<[192, 1]>, 1>) outs(%vc : memref<?x192xf16, strided<[192, 1]>, 1>)
      %sc = memref.subview %c[%m, %n] [%rows, 192] [1, 1] : memref<81x6144xf16> to memref<?x192xf16, strided<[6144, 1], offset: ?>>
      memref.copy %vc, %sc : memref<?x192xf16, strided<[192, 1]>, 1> to memref<?x192xf16, strided<[6144, 1], offset: ?>>
    } {iree_hexagon.vtcm_tile_loop}
    linalg.copy ins(%va : memref<?x2048xf16, strided<[2048, 1]>, 1>) outs(%va : memref<?x2048xf16, strided<[2048, 1]>, 1>)
  } {iree_hexagon.vtcm_tile_loop}
  memref.dealloc %a_buf : memref<40x2048xf16, 1>
  memref.dealloc %b_buf : memref<40x192xf16, 1>
  memref.dealloc %c_buf : memref<40x192xf16, 1>
  return
}

// -----

// Nested tile loops (an HMX matmul tiled in M and N): by default only the
// inner loop is double-buffered and the load of the outer loop stays a copy.
// With outer-tile-loops the outer loop is rotated as well, with the inner one
// as part of its compute.

// CHECK-LABEL: func.func @nested_tile_loops(
//       CHECK:   scf.for
//       CHECK:     memref.copy
//       CHECK:     scf.for
//       CHECK:       scf.if
//       CHECK:         memref.dma_start
//       CHECK:     } {llvm.loop_annotation
//   CHECK-NOT:   llvm.loop_annotation

// OUTER-LABEL: func.func @nested_tile_loops(
//   OUTER-NOT:   memref.copy
//       OUTER:   memref.dma_start
//       OUTER:   scf.for
//       OUTER:     scf.if
//       OUTER:       memref.dma_start
//       OUTER:     memref.dma_wait
//       OUTER:     linalg.copy
//       OUTER:     memref.dma_start
//       OUTER:     scf.for
//       OUTER:       scf.if
//       OUTER:         memref.dma_start
//       OUTER:     } {llvm.loop_annotation
//       OUTER:   } {llvm.loop_annotation
func.func @nested_tile_loops(%a: memref<1024x256xf32>, %b: memref<256x1024xf32>) {
  %c0 = arith.constant 0 : index
  %c128 = arith.constant 128 : index
  %c256 = arith.constant 256 : index
  %c1024 = arith.constant 1024 : index
  %a_buf = memref.alloc() : memref<128x256xf32, 1>
  %a_packed = memref.alloc() : memref<128x256xf32, 1>
  %b_buf = memref.alloc() : memref<256x128xf32, 1>
  scf.for %m = %c0 to %c1024 step %c128 {
    %sa = memref.subview %a[%m, 0] [128, 256] [1, 1] : memref<1024x256xf32> to memref<128x256xf32, strided<[256, 1], offset: ?>>
    memref.copy %sa, %a_buf : memref<128x256xf32, strided<[256, 1], offset: ?>> to memref<128x256xf32, 1>
    linalg.copy ins(%a_buf : memref<128x256xf32, 1>) outs(%a_packed : memref<128x256xf32, 1>)
    scf.for %n = %c0 to %c1024 step %c128 {
      %sb = memref.subview %b[0, %n] [256, 128] [1, 1] : memref<256x1024xf32> to memref<256x128xf32, strided<[1024, 1], offset: ?>>
      memref.copy %sb, %b_buf : memref<256x128xf32, strided<[1024, 1], offset: ?>> to memref<256x128xf32, 1>
      linalg.copy ins(%b_buf : memref<256x128xf32, 1>) outs(%b_buf : memref<256x128xf32, 1>)
    } {iree_hexagon.vtcm_tile_loop}
  } {iree_hexagon.vtcm_tile_loop}
  return
}

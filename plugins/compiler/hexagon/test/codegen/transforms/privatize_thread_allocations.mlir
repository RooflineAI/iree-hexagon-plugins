// RUN: iree-opt --pass-pipeline="builtin.module(func.func(iree-hexagon-privatize-thread-allocations, canonicalize, cse))" --split-input-file --verify-diagnostics %s | FileCheck %s

// Each of the four chunks gets its own slice of one allocation with a leading
// dimension of four, indexed by (iv - lb) / step.

// CHECK-LABEL: func.func @privatize(
//   CHECK-DAG:   %[[C16:.+]] = arith.constant 16 : index
//   CHECK-DAG:   %[[C64:.+]] = arith.constant 64 : index
//   CHECK-DAG:   %[[TAGS:.+]] = memref.alloca() : memref<4x1xi32>
//   CHECK-DAG:   %[[SCRATCH:.+]] = memref.alloc() alignment = 128 : memref<4x64xf32, 1>
//   CHECK-DAG:   %[[VTCM:.+]] = hexagonmem.alloc() <alignment = 2048> : memref<4x1024xf16, 1>
//       CHECK:   scf.forall (%[[IV:.+]]) = (16) to (272) step (64) {
//       CHECK:     %[[OFFSET:.+]] = arith.subi %[[IV]], %[[C16]] : index
//       CHECK:     %[[CHUNK:.+]] = arith.divui %[[OFFSET]], %[[C64]] : index
//   CHECK-DAG:     %[[TAG:.+]] = memref.subview %[[TAGS]][%[[CHUNK]], 0] [1, 1] [1, 1] : memref<4x1xi32> to memref<1xi32, strided<[1], offset: ?>>
//   CHECK-DAG:     %[[S:.+]] = memref.subview %[[SCRATCH]][%[[CHUNK]], 0] [1, 64] [1, 1] : memref<4x64xf32, 1> to memref<64xf32, strided<[1], offset: ?>, 1>
//   CHECK-DAG:     %[[V:.+]] = memref.subview %[[VTCM]][%[[CHUNK]], 0] [1, 1024] [1, 1] : memref<4x1024xf16, 1> to memref<1024xf16, strided<[1], offset: ?>, 1>
//       CHECK:     func.call @use(%[[TAG]], %[[S]], %[[V]])
//   CHECK-NOT:     dealloc
//       CHECK:   }
//   CHECK-DAG:   memref.dealloc %[[SCRATCH]]
//   CHECK-DAG:   hexagonmem.dealloc %[[VTCM]]
func.func private @use(memref<1xi32, strided<[1], offset: ?>>, memref<64xf32, strided<[1], offset: ?>, 1>, memref<1024xf16, strided<[1], offset: ?>, 1>)
func.func @privatize() {
  scf.forall (%iv) = (16) to (272) step (64) {
    %tag = memref.alloca() : memref<1xi32>
    %scratch = memref.alloc() alignment = 128 : memref<64xf32, 1>
    %vtcm = hexagonmem.alloc() <alignment = 2048> : memref<1024xf16, 1>
    %tag_cast = memref.cast %tag : memref<1xi32> to memref<1xi32, strided<[1], offset: ?>>
    %scratch_cast = memref.cast %scratch : memref<64xf32, 1> to memref<64xf32, strided<[1], offset: ?>, 1>
    %vtcm_cast = memref.cast %vtcm : memref<1024xf16, 1> to memref<1024xf16, strided<[1], offset: ?>, 1>
    func.call @use(%tag_cast, %scratch_cast, %vtcm_cast) : (memref<1xi32, strided<[1], offset: ?>>, memref<64xf32, strided<[1], offset: ?>, 1>, memref<1024xf16, strided<[1], offset: ?>, 1>) -> ()
    memref.dealloc %scratch : memref<64xf32, 1>
    hexagonmem.dealloc %vtcm : memref<1024xf16, 1>
  } {mapping = [#iree_codegen.local_mapping<0>]}
  return
}

// -----

// Foralls without the thread marker are left alone.

// CHECK-LABEL: func.func @unmarked(
//       CHECK:   scf.forall
//       CHECK:     memref.alloca() : memref<1xi32>
func.func private @use_tag(memref<1xi32>)
func.func @unmarked() {
  scf.forall (%iv) = (0) to (256) step (64) {
    %tag = memref.alloca() : memref<1xi32>
    func.call @use_tag(%tag) : (memref<1xi32>) -> ()
  }
  return
}

// -----

func.func @dynamic_alloc(%n: index) {
  scf.forall (%iv) = (0) to (256) step (64) {
    // expected-error @+1 {{inside a thread body must have a static shape and identity layout}}
    %a = memref.alloca(%n) : memref<?xi32>
  } {mapping = [#iree_codegen.local_mapping<0>]}
  return
}

// -----

// Slices that are not a multiple of the alignment are padded to it: the wide
// allocation is flat per chunk, viewed as chunks x T with a padded stride.

// CHECK-LABEL: func.func @pad(
//   CHECK-DAG:   %[[VTCM:.+]] = hexagonmem.alloc() <alignment = 2048> : memref<4x1024xf16, 1>
//   CHECK-DAG:   %[[VTCM_VIEW:.+]] = memref.reinterpret_cast %[[VTCM]] to offset: [0], sizes: [4, 32], strides: [1024, 1] : memref<4x1024xf16, 1> to memref<4x32xf16, strided<[1024, 1]>, 1>
//   CHECK-DAG:   %[[SCRATCH:.+]] = memref.alloc() alignment = 64 : memref<4x16xf32, 1>
//   CHECK-DAG:   %[[SCRATCH_VIEW:.+]] = memref.reinterpret_cast %[[SCRATCH]] to offset: [0], sizes: [4, 3, 5], strides: [16, 5, 1] : memref<4x16xf32, 1> to memref<4x3x5xf32, strided<[16, 5, 1]>, 1>
//   CHECK-DAG:   %[[SCALAR:.+]] = memref.alloc() alignment = 64 : memref<4x16xf32, 1>
//   CHECK-DAG:   %[[SCALAR_VIEW:.+]] = memref.reinterpret_cast %[[SCALAR]] to offset: [0], sizes: [4], strides: [16] : memref<4x16xf32, 1> to memref<4xf32, strided<[16]>, 1>
//       CHECK:   scf.forall (%[[IV:.+]]) = (0) to (256) step (64) {
//       CHECK:     %[[CHUNK:.+]] = arith.divui %[[IV]]
//   CHECK-DAG:     %[[V:.+]] = memref.subview %[[VTCM_VIEW]][%[[CHUNK]], 0] [1, 32] [1, 1] : memref<4x32xf16, strided<[1024, 1]>, 1> to memref<32xf16, strided<[1], offset: ?>, 1>
//   CHECK-DAG:     %[[S:.+]] = memref.subview %[[SCRATCH_VIEW]][%[[CHUNK]], 0, 0] [1, 3, 5] [1, 1, 1] : memref<4x3x5xf32, strided<[16, 5, 1]>, 1> to memref<3x5xf32, strided<[5, 1], offset: ?>, 1>
//   CHECK-DAG:     %[[C:.+]] = memref.subview %[[SCALAR_VIEW]][%[[CHUNK]]] [1] [1] : memref<4xf32, strided<[16]>, 1> to memref<f32, strided<[], offset: ?>, 1>
//       CHECK:     func.call @use_padded(%[[V]], %[[S]], %[[C]])
func.func private @use_padded(memref<32xf16, strided<[1], offset: ?>, 1>, memref<3x5xf32, strided<[5, 1], offset: ?>, 1>, memref<f32, strided<[], offset: ?>, 1>)
func.func @pad() {
  scf.forall (%iv) = (0) to (256) step (64) {
    %vtcm = hexagonmem.alloc() <alignment = 2048> : memref<32xf16, 1>
    %scratch = memref.alloc() alignment = 64 : memref<3x5xf32, 1>
    %scalar = memref.alloc() alignment = 64 : memref<f32, 1>
    %vtcm_cast = memref.cast %vtcm : memref<32xf16, 1> to memref<32xf16, strided<[1], offset: ?>, 1>
    %scratch_cast = memref.cast %scratch : memref<3x5xf32, 1> to memref<3x5xf32, strided<[5, 1], offset: ?>, 1>
    %scalar_cast = memref.cast %scalar : memref<f32, 1> to memref<f32, strided<[], offset: ?>, 1>
    func.call @use_padded(%vtcm_cast, %scratch_cast, %scalar_cast) : (memref<32xf16, strided<[1], offset: ?>, 1>, memref<3x5xf32, strided<[5, 1], offset: ?>, 1>, memref<f32, strided<[], offset: ?>, 1>) -> ()
  } {mapping = [#iree_codegen.local_mapping<0>]}
  return
}

// -----

func.func @dynamic_bounds(%ub: index) {
  // expected-error @+1 {{with allocations in its body must be one-dimensional with static bounds}}
  scf.forall (%iv) = (0) to (%ub) step (64) {
    %a = memref.alloca() : memref<4xi32>
  } {mapping = [#iree_codegen.local_mapping<0>]}
  return
}

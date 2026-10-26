// RUN: iree-opt --pass-pipeline="builtin.module(func.func(iree-hexagon-form-async-threads, cse))" --split-input-file --verify-diagnostics %s | FileCheck %s

// Every chunk becomes a task of one group, none runs inline; the await is
// where the dispatching thread joins the pool.

// CHECK-LABEL: func.func @four_chunks(
//  CHECK-SAME:     %[[BUF:.+]]: memref<256xf32>
//   CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//   CHECK-DAG:   %[[C4:.+]] = arith.constant 4 : index
//   CHECK-DAG:   %[[C64:.+]] = arith.constant 64 : index
//   CHECK-DAG:   %[[C256:.+]] = arith.constant 256 : index
//       CHECK:   %[[GROUP:.+]] = async.runtime.create_group %[[C4]] : !async.group
//       CHECK:   scf.for %[[IV:.+]] = %[[C0]] to %[[C256]] step %[[C64]] {
//       CHECK:     %[[TOKEN:.+]] = async.execute {
//       CHECK:       func.call @chunk(%[[BUF]], %[[IV]])
//       CHECK:       async.yield
//       CHECK:     async.add_to_group %[[TOKEN]], %[[GROUP]]
//       CHECK:   }
//   CHECK-NOT:   call @chunk
//       CHECK:   async.await_all %[[GROUP]]
//   CHECK-NOT:   scf.forall
func.func private @chunk(memref<256xf32>, index)
func.func @four_chunks(%buf: memref<256xf32>) {
  scf.forall (%iv) = (0) to (256) step (64) {
    func.call @chunk(%buf, %iv) : (memref<256xf32>, index) -> ()
  } {mapping = [#iree_codegen.local_mapping<0>]}
  return
}

// -----

// Dynamic bounds: the group size is the chunk count.

// CHECK-LABEL: func.func @dynamic(
//       CHECK:   %[[N:.+]] = arith.ceildivsi
//       CHECK:   %[[GROUP:.+]] = async.runtime.create_group %[[N]]
//       CHECK:   scf.for
//       CHECK:     async.execute
//       CHECK:       func.call @chunk(
//       CHECK:     async.add_to_group
//   CHECK-NOT:   call @chunk
//       CHECK:   async.await_all %[[GROUP]]
func.func private @chunk(memref<256xf32>, index)
func.func @dynamic(%buf: memref<256xf32>, %ub: index) {
  scf.forall (%iv) = (0) to (%ub) step (64) {
    func.call @chunk(%buf, %iv) : (memref<256xf32>, index) -> ()
  } {mapping = [#iree_codegen.local_mapping<0>]}
  return
}

// -----

// Foralls without the thread marker are left alone.

// CHECK-LABEL: func.func @unmarked(
//       CHECK:   scf.forall
//   CHECK-NOT:   async.
func.func private @chunk(memref<256xf32>, index)
func.func @unmarked(%buf: memref<256xf32>) {
  scf.forall (%iv) = (0) to (256) step (64) {
    func.call @chunk(%buf, %iv) : (memref<256xf32>, index) -> ()
  }
  return
}

// -----

func.func private @chunk(memref<256xf32>, index)
func.func @two_dims(%buf: memref<256xf32>) {
  // expected-error @+1 {{expected a one-dimensional thread forall}}
  scf.forall (%i, %j) = (0, 0) to (2, 2) step (1, 1) {
    func.call @chunk(%buf, %i) : (memref<256xf32>, index) -> ()
  } {mapping = [#iree_codegen.local_mapping<0>, #iree_codegen.local_mapping<1>]}
  return
}

// RUN: iree-opt --pass-pipeline="builtin.module(func.func(iree-hexagon-coalesce-vtcm-loops))" --split-input-file %s | FileCheck %s

// A perfectly nested band of VTCM tile loops becomes one marked loop over all
// tiles. The compute nested in the tile is left alone.

// CHECK-LABEL: func.func @band_2d(
//       CHECK:   scf.for %[[T:.+]] = %{{.+}} to %{{.+}} step %{{.+}} iter_args(%[[ACC:.+]] = %{{.+}}) -> (tensor<2048x512xf32>) {
//       CHECK:     affine.delinearize_index %[[T]] into (8, 2)
//       CHECK:     iree_hexagon.stage_to_vtcm
//       CHECK:     linalg.copy
//       CHECK:     tensor.insert_slice {{.+}} into %[[ACC]]
//   CHECK-NOT:     scf.for
//       CHECK:   } {iree_hexagon.vtcm_tile_loop}
func.func @band_2d(%a: tensor<2048x512xf32>) -> tensor<2048x512xf32> {
  %c0 = arith.constant 0 : index
  %c256 = arith.constant 256 : index
  %c512 = arith.constant 512 : index
  %c2048 = arith.constant 2048 : index
  %init = tensor.empty() : tensor<2048x512xf32>
  %out = iree_hexagon.vtcm_empty() : tensor<256x256xf32>
  %0 = scf.for %i = %c0 to %c2048 step %c256 iter_args(%acc0 = %init) -> (tensor<2048x512xf32>) {
    %1 = scf.for %j = %c0 to %c512 step %c256 iter_args(%acc1 = %acc0) -> (tensor<2048x512xf32>) {
      %slice = tensor.extract_slice %a[%i, %j] [256, 256] [1, 1] : tensor<2048x512xf32> to tensor<256x256xf32>
      %staged = iree_hexagon.stage_to_vtcm %slice : tensor<256x256xf32>
      %tile = linalg.copy ins(%staged : tensor<256x256xf32>) outs(%out : tensor<256x256xf32>) -> tensor<256x256xf32>
      %inserted = tensor.insert_slice %tile into %acc1[%i, %j] [256, 256] [1, 1] : tensor<256x256xf32> into tensor<2048x512xf32>
      scf.yield %inserted : tensor<2048x512xf32>
    }
    scf.yield %1 : tensor<2048x512xf32>
  }
  return %0 : tensor<2048x512xf32>
}

// -----

// A single tile loop is only marked.

// CHECK-LABEL: func.func @band_1d(
//   CHECK-NOT:   affine.delinearize_index
//       CHECK:   scf.for
//       CHECK:     iree_hexagon.stage_to_vtcm
//       CHECK:   } {iree_hexagon.vtcm_tile_loop}
func.func @band_1d(%a: tensor<2048x512xf32>) -> tensor<2048x512xf32> {
  %c0 = arith.constant 0 : index
  %c128 = arith.constant 128 : index
  %c2048 = arith.constant 2048 : index
  %init = tensor.empty() : tensor<2048x512xf32>
  %out = iree_hexagon.vtcm_empty() : tensor<128x512xf32>
  %0 = scf.for %i = %c0 to %c2048 step %c128 iter_args(%acc = %init) -> (tensor<2048x512xf32>) {
    %slice = tensor.extract_slice %a[%i, 0] [128, 512] [1, 1] : tensor<2048x512xf32> to tensor<128x512xf32>
    %staged = iree_hexagon.stage_to_vtcm %slice : tensor<128x512xf32>
    %tile = linalg.copy ins(%staged : tensor<128x512xf32>) outs(%out : tensor<128x512xf32>) -> tensor<128x512xf32>
    %inserted = tensor.insert_slice %tile into %acc[%i, 0] [128, 512] [1, 1] : tensor<128x512xf32> into tensor<2048x512xf32>
    scf.yield %inserted : tensor<2048x512xf32>
  }
  return %0 : tensor<2048x512xf32>
}

// -----

// Both loops stage tiles (an HMX matmul tiled in M and N): the band is not
// perfectly nested. Each loop is marked, none is coalesced. Staging hoisted out
// of all loops needs no double buffering.

// CHECK-LABEL: func.func @imperfect_band(
//   CHECK-NOT:   affine.delinearize_index
//       CHECK:   iree_hexagon.stage_to_vtcm
//       CHECK:   scf.for
//       CHECK:     iree_hexagon.stage_to_vtcm
//       CHECK:     scf.for
//       CHECK:       iree_hexagon.stage_to_vtcm
//       CHECK:     } {iree_hexagon.vtcm_tile_loop}
//       CHECK:   } {iree_hexagon.vtcm_tile_loop}
func.func @imperfect_band(%a: tensor<1024x256xf32>, %b: tensor<256x1024xf32>, %bias: tensor<128x128xf32>) -> tensor<1024x1024xf32> {
  %c0 = arith.constant 0 : index
  %c128 = arith.constant 128 : index
  %c1024 = arith.constant 1024 : index
  %init = tensor.empty() : tensor<1024x1024xf32>
  %out = iree_hexagon.vtcm_empty() : tensor<128x128xf32>
  %hoisted = iree_hexagon.stage_to_vtcm %bias : tensor<128x128xf32>
  %0 = scf.for %m = %c0 to %c1024 step %c128 iter_args(%acc0 = %init) -> (tensor<1024x1024xf32>) {
    %sa = tensor.extract_slice %a[%m, 0] [128, 256] [1, 1] : tensor<1024x256xf32> to tensor<128x256xf32>
    %staged_a = iree_hexagon.stage_to_vtcm %sa : tensor<128x256xf32>
    %1 = scf.for %n = %c0 to %c1024 step %c128 iter_args(%acc1 = %acc0) -> (tensor<1024x1024xf32>) {
      %sb = tensor.extract_slice %b[0, %n] [256, 128] [1, 1] : tensor<256x1024xf32> to tensor<256x128xf32>
      %staged_b = iree_hexagon.stage_to_vtcm %sb : tensor<256x128xf32>
      %mm = linalg.matmul ins(%staged_a, %staged_b : tensor<128x256xf32>, tensor<256x128xf32>) outs(%hoisted : tensor<128x128xf32>) -> tensor<128x128xf32>
      %tile = linalg.copy ins(%mm : tensor<128x128xf32>) outs(%out : tensor<128x128xf32>) -> tensor<128x128xf32>
      %inserted = tensor.insert_slice %tile into %acc1[%m, %n] [128, 128] [1, 1] : tensor<128x128xf32> into tensor<1024x1024xf32>
      scf.yield %inserted : tensor<1024x1024xf32>
    }
    scf.yield %1 : tensor<1024x1024xf32>
  }
  return %0 : tensor<1024x1024xf32>
}

// -----

// The loop carries a VTCM tile from one iteration to the next: double
// buffering it would split the carried data over two buffers. Not marked.

// CHECK-LABEL: func.func @vtcm_carried(
//       CHECK:   scf.for
//   CHECK-NOT:   iree_hexagon.vtcm_tile_loop
func.func @vtcm_carried(%a: tensor<2048x512xf32>) -> tensor<128x512xf32> {
  %c0 = arith.constant 0 : index
  %c128 = arith.constant 128 : index
  %c2048 = arith.constant 2048 : index
  %acc_init = iree_hexagon.vtcm_empty() : tensor<128x512xf32>
  %0 = scf.for %i = %c0 to %c2048 step %c128 iter_args(%acc = %acc_init) -> (tensor<128x512xf32>) {
    %slice = tensor.extract_slice %a[%i, 0] [128, 512] [1, 1] : tensor<2048x512xf32> to tensor<128x512xf32>
    %staged = iree_hexagon.stage_to_vtcm %slice : tensor<128x512xf32>
    %sum = linalg.copy ins(%staged : tensor<128x512xf32>) outs(%acc : tensor<128x512xf32>) -> tensor<128x512xf32>
    scf.yield %sum : tensor<128x512xf32>
  }
  return %0 : tensor<128x512xf32>
}

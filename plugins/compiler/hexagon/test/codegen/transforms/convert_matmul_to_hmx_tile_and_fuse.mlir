// RUN: iree-opt \
// RUN:   --pass-pipeline='builtin.module(func.func(iree-hexagon-convert-matmul-to-hmx,iree-llvmcpu-tile-and-fuse-producer-consumer{tiling-level=vector_common_parallel},canonicalize,cse))' \
// RUN:   %s | FileCheck %s

// The unpack lowering config makes it the two-dimensional tiling root and
// fuses the HMX matmul producer into the generated loop nest.

// CHECK-LABEL: func.func @tile_and_fuse_converted_matmul(
// CHECK:       iree_hexagon.hmx.tensor_pack
// CHECK:       iree_hexagon.hmx.tensor_pack
// CHECK:       scf.forall
// CHECK:         iree_hexagon.hmx.tensor_matmul
// CHECK:         iree_hexagon.hmx.tensor_unpack
// CHECK:         tensor.parallel_insert_slice
func.func @tile_and_fuse_converted_matmul(
    %lhs: tensor<64x96xf16>, %rhs: tensor<96x64xf16>)
    -> tensor<64x64xf32> {
  %c0 = arith.constant 0.0 : f32
  %lhs_vtcm = iree_hexagon.stage_to_vtcm %lhs : tensor<64x96xf16>
  %rhs_vtcm = iree_hexagon.stage_to_vtcm %rhs : tensor<96x64xf16>
  %init = iree_hexagon.vtcm_empty() : tensor<64x64xf32>
  %filled = linalg.fill ins(%c0 : f32) outs(%init : tensor<64x64xf32>)
      -> tensor<64x64xf32>
  %result = linalg.matmul
      ins(%lhs_vtcm, %rhs_vtcm : tensor<64x96xf16>, tensor<96x64xf16>)
      outs(%filled : tensor<64x64xf32>) -> tensor<64x64xf32>
  return %result : tensor<64x64xf32>
}

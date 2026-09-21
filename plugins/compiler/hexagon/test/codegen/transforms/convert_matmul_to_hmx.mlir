// RUN: iree-opt \
// RUN:   --pass-pipeline='builtin.module(func.func(iree-hexagon-convert-matmul-to-hmx))' \
// RUN:   --split-input-file %s | FileCheck %s

// A canonical f16 x f16 -> f32 matmul is converted, and its destination-style
// initializer remains the unpack destination.

// CHECK-LABEL: func.func @convert_f32_output(
// CHECK:       %[[LHS:.+]] = iree_hexagon.stage_to_vtcm
// CHECK:       %[[RHS:.+]] = iree_hexagon.stage_to_vtcm
// CHECK:       %[[INIT_EMPTY:.+]] = iree_hexagon.vtcm_empty() : tensor<32x32xf32>
// CHECK:       %[[INIT:.+]] = linalg.fill {{.*}} outs(%[[INIT_EMPTY]]
// CHECK:       %[[PACKED_LHS:.+]] = iree_hexagon.hmx.tensor_pack ins(%[[LHS]] : tensor<32x512xf16>) outs(%{{.+}} : tensor<1x16x16x32x2xf16>)
// CHECK:       %[[PACKED_RHS:.+]] = iree_hexagon.hmx.tensor_pack ins(%[[RHS]] : tensor<512x32xf16>) outs(%{{.+}} : tensor<16x1x16x32x2xf16>)
// CHECK:       %[[PRODUCT:.+]] = iree_hexagon.hmx.tensor_matmul ins(%[[PACKED_LHS]], %[[PACKED_RHS]]
// CHECK:       %[[RESULT:.+]] = iree_hexagon.hmx.tensor_unpack ins(%[[PRODUCT]] : tensor<1x1x16x32x2xf16>) outs(%[[INIT]] : tensor<32x32xf32>) {dim = 0 : i64, lowering_config = #{{.+}}}
// CHECK-NOT:   linalg.matmul
// CHECK:       return %[[RESULT]]
func.func @convert_f32_output(
    %lhs: tensor<32x512xf16>, %rhs: tensor<512x32xf16>)
    -> tensor<32x32xf32> {
  %c0 = arith.constant 0.0 : f32
  %lhs_vtcm = iree_hexagon.stage_to_vtcm %lhs : tensor<32x512xf16>
  %rhs_vtcm = iree_hexagon.stage_to_vtcm %rhs : tensor<512x32xf16>
  %init = iree_hexagon.vtcm_empty() : tensor<32x32xf32>
  %filled = linalg.fill ins(%c0 : f32) outs(%init : tensor<32x32xf32>)
      -> tensor<32x32xf32>
  %result = linalg.matmul
      ins(%lhs_vtcm, %rhs_vtcm : tensor<32x512xf16>, tensor<512x32xf16>)
      outs(%filled : tensor<32x32xf32>) -> tensor<32x32xf32>
  return %result : tensor<32x32xf32>
}

// -----

// f16 is the other supported output type. Keep a nonzero initializer to verify
// that destination-passing accumulation is preserved by the conversion.

// CHECK-LABEL: func.func @convert_f16_output(
// CHECK:       %[[INIT:.+]] = linalg.fill {{.*}} : tensor<32x32xf16>
// CHECK:       iree_hexagon.hmx.tensor_unpack {{.*}} outs(%[[INIT]] : tensor<32x32xf16>)
// CHECK-NOT:   linalg.matmul
// CHECK:       return
func.func @convert_f16_output(
    %lhs: tensor<32x64xf16>, %rhs: tensor<64x32xf16>)
    -> tensor<32x32xf16> {
  %c5 = arith.constant 5.0 : f16
  %lhs_vtcm = iree_hexagon.stage_to_vtcm %lhs : tensor<32x64xf16>
  %rhs_vtcm = iree_hexagon.stage_to_vtcm %rhs : tensor<64x32xf16>
  %init = iree_hexagon.vtcm_empty() : tensor<32x32xf16>
  %filled = linalg.fill ins(%c5 : f16) outs(%init : tensor<32x32xf16>)
      -> tensor<32x32xf16>
  %result = linalg.matmul
      ins(%lhs_vtcm, %rhs_vtcm : tensor<32x64xf16>, tensor<64x32xf16>)
      outs(%filled : tensor<32x32xf16>) -> tensor<32x32xf16>
  return %result : tensor<32x32xf16>
}

// -----

// A unit batch is rank-reduced before conversion. This also covers provenance
// through collapse_shape and a user-defined transpose-b map.

// CHECK-LABEL: func.func @convert_unit_batch_transpose_b(
// CHECK:       %[[LHS_2D:.+]] = tensor.collapse_shape
// CHECK:       %[[RHS_2D:.+]] = tensor.collapse_shape
// CHECK:       iree_hexagon.hmx.tensor_pack ins(%[[LHS_2D]] : tensor<32x64xf16>) {{.*}} {dim = 0 : i64}
// CHECK:       iree_hexagon.hmx.tensor_pack ins(%[[RHS_2D]] : tensor<32x64xf16>) {{.*}} {dim = 1 : i64}
// CHECK-NOT:   linalg.batch_matmul
// CHECK-NOT:   linalg.matmul
// CHECK:       tensor.expand_shape
func.func @convert_unit_batch_transpose_b(
    %lhs: tensor<1x32x64xf16>, %rhs: tensor<1x32x64xf16>)
    -> tensor<1x32x32xf32> {
  %c0 = arith.constant 0.0 : f32
  %lhs_vtcm = iree_hexagon.stage_to_vtcm %lhs : tensor<1x32x64xf16>
  %rhs_vtcm = iree_hexagon.stage_to_vtcm %rhs : tensor<1x32x64xf16>
  %init = iree_hexagon.vtcm_empty() : tensor<1x32x32xf32>
  %filled = linalg.fill ins(%c0 : f32) outs(%init : tensor<1x32x32xf32>)
      -> tensor<1x32x32xf32>
  %result = linalg.batch_matmul
      indexing_maps = [affine_map<(b, m, n, k) -> (b, m, k)>,
                       affine_map<(b, m, n, k) -> (b, n, k)>,
                       affine_map<(b, m, n, k) -> (b, m, n)>]
      ins(%lhs_vtcm, %rhs_vtcm : tensor<1x32x64xf16>, tensor<1x32x64xf16>)
      outs(%filled : tensor<1x32x32xf32>) -> tensor<1x32x32xf32>
  return %result : tensor<1x32x32xf32>
}

// -----

// Transpose-a and transpose-b may be combined. Each pack's dim attribute must
// describe where its interleaved logical row dimension appears in the source.

// CHECK-LABEL: func.func @convert_both_inputs_transposed(
// CHECK:       iree_hexagon.hmx.tensor_pack {{.*}} {dim = 1 : i64}
// CHECK:       iree_hexagon.hmx.tensor_pack {{.*}} {dim = 1 : i64}
// CHECK:       iree_hexagon.hmx.tensor_matmul {{.*}} tensor<2x3x16x32x2xf16>, tensor<3x4x16x32x2xf16>
// CHECK-NOT:   linalg.matmul
// CHECK:       return
func.func @convert_both_inputs_transposed(
    %lhs: tensor<96x64xf16>, %rhs: tensor<128x96xf16>)
    -> tensor<64x128xf32> {
  %c0 = arith.constant 0.0 : f32
  %lhs_vtcm = iree_hexagon.stage_to_vtcm %lhs : tensor<96x64xf16>
  %rhs_vtcm = iree_hexagon.stage_to_vtcm %rhs : tensor<128x96xf16>
  %init = iree_hexagon.vtcm_empty() : tensor<64x128xf32>
  %filled = linalg.fill ins(%c0 : f32) outs(%init : tensor<64x128xf32>)
      -> tensor<64x128xf32>
  %result = linalg.matmul
      indexing_maps = [affine_map<(m, n, k) -> (k, m)>,
                       affine_map<(m, n, k) -> (n, k)>,
                       affine_map<(m, n, k) -> (m, n)>]
      ins(%lhs_vtcm, %rhs_vtcm : tensor<96x64xf16>, tensor<128x96xf16>)
      outs(%filled : tensor<64x128xf32>) -> tensor<64x128xf32>
  return %result : tensor<64x128xf32>
}

// -----

// Ragged static dimensions round up independently to complete physical tiles.

// CHECK-LABEL: func.func @convert_ragged(
// CHECK:       iree_hexagon.hmx.tensor_pack {{.*}} tensor<49x9x16x32x2xf16>
// CHECK:       iree_hexagon.hmx.tensor_pack {{.*}} tensor<9x49x16x32x2xf16>
// CHECK:       iree_hexagon.hmx.tensor_matmul {{.*}} tensor<49x49x16x32x2xf16>
// CHECK:       iree_hexagon.hmx.tensor_unpack {{.*}} tensor<1537x1539xf32>
func.func @convert_ragged(
    %lhs: tensor<1537x257xf16>, %rhs: tensor<257x1539xf16>)
    -> tensor<1537x1539xf32> {
  %c0 = arith.constant 0.0 : f32
  %lhs_vtcm = iree_hexagon.stage_to_vtcm %lhs : tensor<1537x257xf16>
  %rhs_vtcm = iree_hexagon.stage_to_vtcm %rhs : tensor<257x1539xf16>
  %init = iree_hexagon.vtcm_empty() : tensor<1537x1539xf32>
  %filled = linalg.fill ins(%c0 : f32) outs(%init : tensor<1537x1539xf32>)
      -> tensor<1537x1539xf32>
  %result = linalg.matmul
      ins(%lhs_vtcm, %rhs_vtcm : tensor<1537x257xf16>, tensor<257x1539xf16>)
      outs(%filled : tensor<1537x1539xf32>) -> tensor<1537x1539xf32>
  return %result : tensor<1537x1539xf32>
}

// -----

// A dynamic boundary tile is accepted when value-bounds analysis can prove its
// maximum extent. This mirrors the affine.min shape produced by VTCM tiling.

// CHECK-LABEL: func.func @convert_bounded_dynamic_boundary(
// CHECK:       iree_hexagon.hmx.tensor_pack {{.*}} tensor<49x9x16x32x2xf16>
// CHECK:       iree_hexagon.hmx.tensor_pack {{.*}} tensor<9x6x16x32x2xf16>
// CHECK:       iree_hexagon.hmx.tensor_matmul {{.*}} tensor<49x6x16x32x2xf16>
// CHECK:       iree_hexagon.hmx.tensor_unpack {{.*}} tensor<1537x?xf32>
func.func @convert_bounded_dynamic_boundary(
    %lhs: tensor<1537x257xf16>, %rhs: tensor<257x1539xf16>, %offset: index)
    -> tensor<1537x?xf32> {
  %c0 = arith.constant 0.0 : f32
  %n = affine.min affine_map<(d0) -> (-d0 + 1539, 192)>(%offset)
  %rhs_slice = tensor.extract_slice %rhs[0, %offset] [257, %n] [1, 1]
      : tensor<257x1539xf16> to tensor<257x?xf16>
  %lhs_vtcm = iree_hexagon.stage_to_vtcm %lhs : tensor<1537x257xf16>
  %rhs_vtcm = iree_hexagon.stage_to_vtcm %rhs_slice : tensor<257x?xf16>
  %init = iree_hexagon.vtcm_empty(%n) : tensor<1537x?xf32>
  %filled = linalg.fill ins(%c0 : f32) outs(%init : tensor<1537x?xf32>)
      -> tensor<1537x?xf32>
  %result = linalg.matmul
      ins(%lhs_vtcm, %rhs_vtcm : tensor<1537x257xf16>, tensor<257x?xf16>)
      outs(%filled : tensor<1537x?xf32>) -> tensor<1537x?xf32>
  return %result : tensor<1537x?xf32>
}

// -----

// This is the minimized tensor-level carrier chain observed in
// batched_4x1024x128x1024: a VTCM destination is filled, carried through an
// scf.for iter_arg, rank-reduced with extract_slice, and updated through the
// yielded DPS recurrence.

// CHECK-LABEL: func.func @convert_scf_for_carried_batch(
// CHECK:       scf.for
// CHECK:         tensor.extract_slice
// CHECK:         iree_hexagon.hmx.tensor_pack
// CHECK:         iree_hexagon.hmx.tensor_pack
// CHECK:         iree_hexagon.hmx.tensor_matmul
// CHECK:         iree_hexagon.hmx.tensor_unpack
// CHECK-NOT:     linalg.matmul
func.func @convert_scf_for_carried_batch(
    %lhs: tensor<4x32x64xf16>, %rhs: tensor<4x64x32xf16>)
    -> tensor<4x32x32xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %zero = arith.constant 0.0 : f32
  %lhs_vtcm = iree_hexagon.stage_to_vtcm %lhs : tensor<4x32x64xf16>
  %rhs_vtcm = iree_hexagon.stage_to_vtcm %rhs : tensor<4x64x32xf16>
  %empty = iree_hexagon.vtcm_empty() : tensor<4x32x32xf32>
  %filled = linalg.fill ins(%zero : f32) outs(%empty : tensor<4x32x32xf32>)
      -> tensor<4x32x32xf32>
  %result = scf.for %b = %c0 to %c4 step %c1
      iter_args(%iter = %filled) -> tensor<4x32x32xf32> {
    %lhs_tile = tensor.extract_slice %lhs_vtcm[%b, 0, 0] [1, 32, 64]
        [1, 1, 1] : tensor<4x32x64xf16> to tensor<32x64xf16>
    %rhs_tile = tensor.extract_slice %rhs_vtcm[%b, 0, 0] [1, 64, 32]
        [1, 1, 1] : tensor<4x64x32xf16> to tensor<64x32xf16>
    %out_tile = tensor.extract_slice %iter[%b, 0, 0] [1, 32, 32]
        [1, 1, 1] : tensor<4x32x32xf32> to tensor<32x32xf32>
    %product = linalg.matmul
        ins(%lhs_tile, %rhs_tile : tensor<32x64xf16>, tensor<64x32xf16>)
        outs(%out_tile : tensor<32x32xf32>) -> tensor<32x32xf32>
    %updated = tensor.insert_slice %product into %iter[%b, 0, 0]
        [1, 32, 32] [1, 1, 1]
        : tensor<32x32xf32> into tensor<4x32x32xf32>
    scf.yield %updated : tensor<4x32x32xf32>
  }
  return %result : tensor<4x32x32xf32>
}

// -----

// CHECK-LABEL: func.func @convert_scf_forall_shared_output(
// CHECK:       scf.forall
// CHECK:         iree_hexagon.hmx.tensor_matmul
// CHECK:         iree_hexagon.hmx.tensor_unpack
// CHECK-NOT:     linalg.matmul
func.func @convert_scf_forall_shared_output(
    %lhs: tensor<4x32x64xf16>, %rhs: tensor<4x64x32xf16>)
    -> tensor<4x32x32xf32> {
  %zero = arith.constant 0.0 : f32
  %lhs_vtcm = iree_hexagon.stage_to_vtcm %lhs : tensor<4x32x64xf16>
  %rhs_vtcm = iree_hexagon.stage_to_vtcm %rhs : tensor<4x64x32xf16>
  %empty = iree_hexagon.vtcm_empty() : tensor<4x32x32xf32>
  %filled = linalg.fill ins(%zero : f32) outs(%empty : tensor<4x32x32xf32>)
      -> tensor<4x32x32xf32>
  %result = scf.forall (%b) in (4) shared_outs(%out = %filled)
      -> tensor<4x32x32xf32> {
    %lhs_tile = tensor.extract_slice %lhs_vtcm[%b, 0, 0] [1, 32, 64]
        [1, 1, 1] : tensor<4x32x64xf16> to tensor<32x64xf16>
    %rhs_tile = tensor.extract_slice %rhs_vtcm[%b, 0, 0] [1, 64, 32]
        [1, 1, 1] : tensor<4x64x32xf16> to tensor<64x32xf16>
    %out_tile = tensor.extract_slice %out[%b, 0, 0] [1, 32, 32]
        [1, 1, 1] : tensor<4x32x32xf32> to tensor<32x32xf32>
    %product = linalg.matmul
        ins(%lhs_tile, %rhs_tile : tensor<32x64xf16>, tensor<64x32xf16>)
        outs(%out_tile : tensor<32x32xf32>) -> tensor<32x32xf32>
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %product into %out[%b, 0, 0]
          [1, 32, 32] [1, 1, 1]
          : tensor<32x32xf32> into tensor<4x32x32xf32>
    }
  }
  return %result : tensor<4x32x32xf32>
}

// -----

// linalg.generic contraction support is deferred. Matching only
// maps and iterators would not establish that an arbitrary region is a matmul.

// CHECK-LABEL: func.func @do_not_convert_generic_contraction(
// CHECK:       linalg.generic
// CHECK-NOT:   iree_hexagon.hmx
// CHECK:       return
func.func @do_not_convert_generic_contraction(
    %lhs: tensor<32x64xf16>, %rhs: tensor<64x32xf16>)
    -> tensor<32x32xf32> {
  %c0 = arith.constant 0.0 : f32
  %lhs_vtcm = iree_hexagon.stage_to_vtcm %lhs : tensor<32x64xf16>
  %rhs_vtcm = iree_hexagon.stage_to_vtcm %rhs : tensor<64x32xf16>
  %init = iree_hexagon.vtcm_empty() : tensor<32x32xf32>
  %filled = linalg.fill ins(%c0 : f32) outs(%init : tensor<32x32xf32>)
      -> tensor<32x32xf32>
  %result = linalg.generic
      {indexing_maps = [affine_map<(m, n, k) -> (m, k)>,
                        affine_map<(m, n, k) -> (k, n)>,
                        affine_map<(m, n, k) -> (m, n)>],
       iterator_types = ["parallel", "parallel", "reduction"]}
      ins(%lhs_vtcm, %rhs_vtcm : tensor<32x64xf16>, tensor<64x32xf16>)
      outs(%filled : tensor<32x32xf32>) {
    ^bb0(%a: f16, %b: f16, %c: f32):
      %a32 = arith.extf %a : f16 to f32
      %b32 = arith.extf %b : f16 to f32
      %product = arith.mulf %a32, %b32 : f32
      %sum = arith.addf %c, %product : f32
      linalg.yield %sum : f32
  } -> tensor<32x32xf32>
  return %result : tensor<32x32xf32>
}

// -----

// Unsupported input and output element types remain ordinary matmuls.

// CHECK-LABEL: func.func @do_not_convert_f32_inputs(
// CHECK:       linalg.matmul
// CHECK-NOT:   iree_hexagon.hmx
// CHECK:       return
func.func @do_not_convert_f32_inputs(
    %lhs: tensor<32x64xf32>, %rhs: tensor<64x32xf32>)
    -> tensor<32x32xf32> {
  %init = tensor.empty() : tensor<32x32xf32>
  %result = linalg.matmul
      ins(%lhs, %rhs : tensor<32x64xf32>, tensor<64x32xf32>)
      outs(%init : tensor<32x32xf32>) -> tensor<32x32xf32>
  return %result : tensor<32x32xf32>
}

// -----

// CHECK-LABEL: func.func @do_not_convert_f64_output(
// CHECK:       linalg.matmul
// CHECK-NOT:   iree_hexagon.hmx
// CHECK:       return
func.func @do_not_convert_f64_output(
    %lhs: tensor<32x64xf16>, %rhs: tensor<64x32xf16>)
    -> tensor<32x32xf64> {
  %lhs_vtcm = iree_hexagon.stage_to_vtcm %lhs : tensor<32x64xf16>
  %rhs_vtcm = iree_hexagon.stage_to_vtcm %rhs : tensor<64x32xf16>
  %init = iree_hexagon.vtcm_empty() : tensor<32x32xf64>
  %result = linalg.matmul
      ins(%lhs_vtcm, %rhs_vtcm : tensor<32x64xf16>, tensor<64x32xf16>)
      outs(%init : tensor<32x32xf64>) -> tensor<32x32xf64>
  return %result : tensor<32x32xf64>
}

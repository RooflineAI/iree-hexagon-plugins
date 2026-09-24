// RUN: iree-opt \
// RUN:   --pass-pipeline='builtin.module(func.func(iree-hexagon-convert-matmul-to-hmx))' \
// RUN:   --split-input-file --verify-diagnostics %s

// Otherwise eligible matmuls are rejected when an input bypasses the VTCM
// staging contract.
// expected-note @+1 {{provenance stopped at unsupported carrier operation 'func.func'}}
func.func @reject_unstaged_input(
    %lhs: tensor<32x64xf16>, %rhs: tensor<64x32xf16>)
    -> tensor<32x32xf32> {
  %rhs_vtcm = iree_hexagon.stage_to_vtcm %rhs : tensor<64x32xf16>
  %init = iree_hexagon.vtcm_empty() : tensor<32x32xf32>
  // expected-error @+1 {{could not prove lhs has the required VTCM provenance}}
  %result = linalg.matmul
      ins(%lhs, %rhs_vtcm : tensor<32x64xf16>, tensor<64x32xf16>)
      outs(%init : tensor<32x32xf32>) -> tensor<32x32xf32>
  return %result : tensor<32x32xf32>
}

// -----

// Destination-style producers may be traversed, but their init must ultimately
// be a vtcm_empty rather than an ordinary tensor.empty.
func.func @reject_non_vtcm_output(
    %lhs: tensor<32x64xf16>, %rhs: tensor<64x32xf16>)
    -> tensor<32x32xf32> {
  %c0 = arith.constant 0.0 : f32
  %lhs_vtcm = iree_hexagon.stage_to_vtcm %lhs : tensor<32x64xf16>
  %rhs_vtcm = iree_hexagon.stage_to_vtcm %rhs : tensor<64x32xf16>
  // expected-note @+1 {{provenance stopped at unsupported carrier operation 'tensor.empty'}}
  %init = tensor.empty() : tensor<32x32xf32>
  %filled = linalg.fill ins(%c0 : f32) outs(%init : tensor<32x32xf32>)
      -> tensor<32x32xf32>
  // expected-error @+1 {{could not prove output initializer has the required VTCM provenance}}
  %result = linalg.matmul
      ins(%lhs_vtcm, %rhs_vtcm : tensor<32x64xf16>, tensor<64x32xf16>)
      outs(%filled : tensor<32x32xf32>) -> tensor<32x32xf32>
  return %result : tensor<32x32xf32>
}

// -----

// Both the loop initialization and its yielded recurrence participate in the
// proof. Yielding an unrelated tensor invalidates the carried VTCM marker.
func.func @reject_conflicting_loop_yield(
    %lhs: tensor<32x64xf16>, %rhs: tensor<64x32xf16>)
    -> tensor<2x32x32xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %lhs_vtcm = iree_hexagon.stage_to_vtcm %lhs : tensor<32x64xf16>
  %rhs_vtcm = iree_hexagon.stage_to_vtcm %rhs : tensor<64x32xf16>
  %init = iree_hexagon.vtcm_empty() : tensor<2x32x32xf32>
  %result = scf.for %i = %c0 to %c1 step %c1
      iter_args(%iter = %init) -> tensor<2x32x32xf32> {
    %out_tile = tensor.extract_slice %iter[0, 0, 0] [1, 32, 32]
        [1, 1, 1] : tensor<2x32x32xf32> to tensor<32x32xf32>
    // expected-error @+1 {{could not prove output initializer has the required VTCM provenance}}
    %product = linalg.matmul
        ins(%lhs_vtcm, %rhs_vtcm : tensor<32x64xf16>, tensor<64x32xf16>)
        outs(%out_tile : tensor<32x32xf32>) -> tensor<32x32xf32>
    // expected-note @+1 {{provenance stopped at unsupported carrier operation 'tensor.empty'}}
    %conflict = tensor.empty() : tensor<2x32x32xf32>
    %updated = tensor.insert_slice %product into %conflict[0, 0, 0]
        [1, 32, 32] [1, 1, 1]
        : tensor<32x32xf32> into tensor<2x32x32xf32>
    scf.yield %updated : tensor<2x32x32xf32>
  }
  return %result : tensor<2x32x32xf32>
}

// -----

// Dynamic shapes without a provable finite upper bound cannot size the static
// packed HMX tile grid.
func.func @reject_unbounded_dynamic_shape(
    %lhs: tensor<?x64xf16>, %rhs: tensor<64x?xf16>, %m: index, %n: index)
    -> tensor<?x?xf32> {
  %lhs_vtcm = iree_hexagon.stage_to_vtcm %lhs : tensor<?x64xf16>
  %rhs_vtcm = iree_hexagon.stage_to_vtcm %rhs : tensor<64x?xf16>
  %init = iree_hexagon.vtcm_empty(%m, %n) : tensor<?x?xf32>
  // expected-warning @+1 {{could not bound HMX matmul shapes for packing}}
  %result = linalg.matmul
      ins(%lhs_vtcm, %rhs_vtcm : tensor<?x64xf16>, tensor<64x?xf16>)
      outs(%init : tensor<?x?xf32>) -> tensor<?x?xf32>
  return %result : tensor<?x?xf32>
}

// -----

func.func @reject_zero_logical_extent(
    %lhs: tensor<0x64xf16>, %rhs: tensor<64x32xf16>)
    -> tensor<0x32xf32> {
  %lhs_vtcm = iree_hexagon.stage_to_vtcm %lhs : tensor<0x64xf16>
  %rhs_vtcm = iree_hexagon.stage_to_vtcm %rhs : tensor<64x32xf16>
  %init = iree_hexagon.vtcm_empty() : tensor<0x32xf32>
  // expected-error @+1 {{HMX matmul requires positive bounded M, N, and K}}
  %result = linalg.matmul
      ins(%lhs_vtcm, %rhs_vtcm : tensor<0x64xf16>, tensor<64x32xf16>)
      outs(%init : tensor<0x32xf32>) -> tensor<0x32xf32>
  return %result : tensor<0x32xf32>
}

// -----

// A transposed output is outside the runtime unpack contract.
func.func @reject_transposed_output(
    %lhs: tensor<32x64xf16>, %rhs: tensor<64x32xf16>)
    -> tensor<32x32xf32> {
  %lhs_vtcm = iree_hexagon.stage_to_vtcm %lhs : tensor<32x64xf16>
  %rhs_vtcm = iree_hexagon.stage_to_vtcm %rhs : tensor<64x32xf16>
  %init = iree_hexagon.vtcm_empty() : tensor<32x32xf32>
  // expected-warning @+1 {{unsupported HMX matmul operand layout}}
  %result = linalg.matmul
      indexing_maps = [affine_map<(m, n, k) -> (m, k)>,
                       affine_map<(m, n, k) -> (k, n)>,
                       affine_map<(m, n, k) -> (n, m)>]
      ins(%lhs_vtcm, %rhs_vtcm : tensor<32x64xf16>, tensor<64x32xf16>)
      outs(%init : tensor<32x32xf32>) -> tensor<32x32xf32>
  return %result : tensor<32x32xf32>
}

// -----

// A batch dimension larger than one shows that the required outer batch tiling
// did not run.
func.func @leave_untiled_batch_matmul(
    %lhs: tensor<4x32x64xf16>, %rhs: tensor<4x64x32xf16>)
    -> tensor<4x32x32xf32> {
  %init = tensor.empty() : tensor<4x32x32xf32>
  // expected-error @+1 {{HMX-eligible batch matmul still has a batch dimension}}
  %result = linalg.batch_matmul
      ins(%lhs, %rhs : tensor<4x32x64xf16>, tensor<4x64x32xf16>)
      outs(%init : tensor<4x32x32xf32>) -> tensor<4x32x32xf32>
  return %result : tensor<4x32x32xf32>
}

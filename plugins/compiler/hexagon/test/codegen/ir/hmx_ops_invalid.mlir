// RUN: iree-opt --iree-load-plugin=hexagon=$ROOF_HEXAGON_COMPILER_PLUGIN \
// RUN:   --split-input-file --verify-diagnostics %s

// expected-error @+1 {{expected the fixed HMX accumulator type 32x32xf32}}
func.func @reject_non_hardware_accumulator(%acc: !iree_hexagon.hmx.acc<16x16xf32>) {
  return
}

// -----

func.func @reject_missing_vtcm_empty_dynamic_size(%m: index) {
  // expected-error @+1 {{expected 2 dynamic size operands for the result type, got 1}}
  %empty = iree_hexagon.vtcm_empty(%m) : tensor<?x?xf32>
  return
}

// -----

func.func @reject_pack_dim(
    %source: tensor<32x32xf16>,
    %dest: tensor<1x1x16x32x2xf16>) {
  // expected-error @+1 {{dim must be 0 or 1}}
  %packed = iree_hexagon.hmx.tensor_pack
      ins(%source : tensor<32x32xf16>)
      outs(%dest : tensor<1x1x16x32x2xf16>) {dim = 2 : i64}
      -> tensor<1x1x16x32x2xf16>
  return
}

// -----

func.func @reject_dynamic_physical_suffix(
    %source: tensor<32x32xf16>,
    %dest: tensor<1x1x?x32x2xf16>) {
  // expected-error @+1 {{destination must have the static physical suffix [16, 32, 2]}}
  %packed = iree_hexagon.hmx.tensor_pack
      ins(%source : tensor<32x32xf16>)
      outs(%dest : tensor<1x1x?x32x2xf16>) {dim = 0 : i64}
      -> tensor<1x1x?x32x2xf16>
  return
}

// -----

func.func @reject_dynamic_physical_grid(
    %source: tensor<32x32xf16>,
    %dest: tensor<?x1x16x32x2xf16>) {
  // expected-error @+1 {{destination must have positive static tile-grid dimensions}}
  %packed = iree_hexagon.hmx.tensor_pack
      ins(%source : tensor<32x32xf16>)
      outs(%dest : tensor<?x1x16x32x2xf16>) {dim = 0 : i64}
      -> tensor<?x1x16x32x2xf16>
  return
}

// -----

func.func @reject_pack_source_rank(
    %source: tensor<1x32x32xf16>,
    %dest: tensor<1x1x16x32x2xf16>) {
  // expected-error @+1 {{source must be a rank-2 f16 shaped value}}
  %packed = iree_hexagon.hmx.tensor_pack
      ins(%source : tensor<1x32x32xf16>)
      outs(%dest : tensor<1x1x16x32x2xf16>) {dim = 0 : i64}
      -> tensor<1x1x16x32x2xf16>
  return
}

// -----

func.func @reject_pack_source_type(
    %source: tensor<32x32xf32>,
    %dest: tensor<1x1x16x32x2xf16>) {
  // expected-error @+1 {{source must be a rank-2 f16 shaped value}}
  %packed = iree_hexagon.hmx.tensor_pack
      ins(%source : tensor<32x32xf32>)
      outs(%dest : tensor<1x1x16x32x2xf16>) {dim = 0 : i64}
      -> tensor<1x1x16x32x2xf16>
  return
}

// -----

func.func @reject_pack_result_not_tied_to_init(
    %source: tensor<32x32xf16>,
    %dest: tensor<1x1x16x32x2xf16>) {
  // expected-error @+1 {{failed to verify that all of {dest, result} have same type}}
  %packed = iree_hexagon.hmx.tensor_pack
      ins(%source : tensor<32x32xf16>)
      outs(%dest : tensor<1x1x16x32x2xf16>) {dim = 0 : i64}
      -> tensor<2x1x16x32x2xf16>
  return
}

// -----

func.func @reject_undersized_pack_grid(
    %source: memref<64x32xf16, 1>,
    %dest: memref<1x1x16x32x2xf16, 1>) {
  // expected-error @+1 {{destination grid does not cover ceil(rows/32) logical tiles}}
  iree_hexagon.hmx.pack ins(%source : memref<64x32xf16, 1>)
      outs(%dest : memref<1x1x16x32x2xf16, 1>) {dim = 0 : i64}
  return
}

// -----

func.func @reject_mismatched_matmul_k(
    %lhs: tensor<1x2x16x32x2xf16>,
    %rhs: tensor<3x1x16x32x2xf16>,
    %acc: tensor<1x1x16x32x2xf16>) {
  // expected-error @+1 {{lhs and rhs K tile counts must match}}
  %result = iree_hexagon.hmx.tensor_matmul
      ins(%lhs, %rhs : tensor<1x2x16x32x2xf16>,
                       tensor<3x1x16x32x2xf16>)
      outs(%acc : tensor<1x1x16x32x2xf16>)
      -> tensor<1x1x16x32x2xf16>
  return
}

// -----

func.func @reject_mismatched_matmul_output_grid(
    %lhs: tensor<2x3x16x32x2xf16>,
    %rhs: tensor<3x4x16x32x2xf16>,
    %acc: tensor<1x4x16x32x2xf16>) {
  // expected-error @+1 {{accumulator grid must match lhs M and rhs N tile counts}}
  %result = iree_hexagon.hmx.tensor_matmul
      ins(%lhs, %rhs : tensor<2x3x16x32x2xf16>,
                       tensor<3x4x16x32x2xf16>)
      outs(%acc : tensor<1x4x16x32x2xf16>)
      -> tensor<1x4x16x32x2xf16>
  return
}

// -----

func.func @reject_rank3_accumulator_for_non_singleton_grid(
    %lhs: memref<2x3x16x32x2xf16, 1>,
    %rhs: memref<3x1x16x32x2xf16, 1>,
    %acc: memref<16x32x2xf16, 1>) {
  // expected-error @+1 {{a rank-3 accumulator requires singleton lhs M and rhs N grids}}
  iree_hexagon.hmx.matmul
      ins(%lhs, %rhs : memref<2x3x16x32x2xf16, 1>,
                       memref<3x1x16x32x2xf16, 1>)
      outs(%acc : memref<16x32x2xf16, 1>)
  return
}

// -----

func.func @reject_dynamic_matmul_k(
    %lhs: tensor<1x?x16x32x2xf16>,
    %rhs: tensor<?x1x16x32x2xf16>,
    %acc: tensor<1x1x16x32x2xf16>) {
  // expected-error @+1 {{lhs must have positive static tile-grid dimensions}}
  %result = iree_hexagon.hmx.tensor_matmul
      ins(%lhs, %rhs : tensor<1x?x16x32x2xf16>,
                       tensor<?x1x16x32x2xf16>)
      outs(%acc : tensor<1x1x16x32x2xf16>)
      -> tensor<1x1x16x32x2xf16>
  return
}

// -----

func.func @reject_large_single_tile_unpack(
    %source: tensor<16x32x2xf16>, %dest: tensor<33x32xf32>) {
  // expected-error @+1 {{a rank-3 source can unpack at most one logical 32x32 tile}}
  %result = iree_hexagon.hmx.tensor_unpack
      ins(%source : tensor<16x32x2xf16>)
      outs(%dest : tensor<33x32xf32>) {dim = 0 : i64}
      -> tensor<33x32xf32>
  return
}

// -----

func.func @reject_undersized_unpack_grid(
    %source: tensor<1x1x16x32x2xf16>, %dest: tensor<33x32xf32>) {
  // expected-error @+1 {{source grid does not cover ceil(rows/32) logical tiles}}
  %result = iree_hexagon.hmx.tensor_unpack
      ins(%source : tensor<1x1x16x32x2xf16>)
      outs(%dest : tensor<33x32xf32>) {dim = 0 : i64}
      -> tensor<33x32xf32>
  return
}

// -----

func.func @reject_unpack_dim(
    %source: tensor<1x1x16x32x2xf16>, %dest: tensor<32x32xf32>) {
  // expected-error @+1 {{dim must be 0}}
  %result = iree_hexagon.hmx.tensor_unpack
      ins(%source : tensor<1x1x16x32x2xf16>)
      outs(%dest : tensor<32x32xf32>) {dim = 1 : i64}
      -> tensor<32x32xf32>
  return
}

// -----

func.func @reject_unpack_destination_type(
    %source: tensor<1x1x16x32x2xf16>, %dest: tensor<32x32xi32>) {
  // expected-error @+1 {{destination must be a rank-2 f16 or f32 value}}
  %result = iree_hexagon.hmx.tensor_unpack
      ins(%source : tensor<1x1x16x32x2xf16>)
      outs(%dest : tensor<32x32xi32>) {dim = 0 : i64}
      -> tensor<32x32xi32>
  return
}

// -----

func.func @reject_short_hmx_config(%config: memref<128xi8, 1>) {
  // expected-error @+1 {{config must be a static rank-1 i8 memref containing at least 256 bytes}}
  iree_hexagon.hmx.acc.setup_read %config : memref<128xi8, 1>
  return
}

// -----

func.func @reject_mma_operand_shape(
    %lhs: memref<8x32x2xf16, 1>,
    %rhs: memref<16x32x2xf16, 1>) {
  %acc = iree_hexagon.hmx.acc.zero : !iree_hexagon.hmx.acc<32x32xf32>
  // expected-error @+1 {{lhs must be one f16 [16, 32, 2] HMX tile}}
  %next = iree_hexagon.hmx.mma %lhs, %rhs, %acc
      : memref<8x32x2xf16, 1>, memref<16x32x2xf16, 1>,
        !iree_hexagon.hmx.acc<32x32xf32> -> !iree_hexagon.hmx.acc<32x32xf32>
  return
}

// -----

func.func @reject_acc_read_destination(
    %dest: memref<16x31x2xf16, 1>) {
  %acc = iree_hexagon.hmx.acc.zero : !iree_hexagon.hmx.acc<32x32xf32>
  // expected-error @+1 {{destination must be one f16 [16, 32, 2] HMX tile}}
  iree_hexagon.hmx.acc.read %acc, %dest
      : !iree_hexagon.hmx.acc<32x32xf32>, memref<16x31x2xf16, 1>
  return
}

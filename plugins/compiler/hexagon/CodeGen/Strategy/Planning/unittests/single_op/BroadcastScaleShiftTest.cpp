// Copyright 2026 RooflineAI GmbH
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "TestUtils.h"

namespace mlir::iree_compiler::hexagon::codegen::planning {
namespace {

/// Per col affine transform out[i][j] = x[i][j] * scale[j] + shift[j]
/// scale and shift are Broadcasted
/// 2 loops (d0, d1), both parallel, 3 inputs with different indexing maps:
/// the main operand depends on both dims, scale/shift depend on d1 only.
REGISTER_ESTIMATION_TEST_SUITE(
    BroadcastScaleShift,
    R"mlir(
      func.func @broadcast_scale_shift(%x: tensor<3136x64xf32>, %scale: tensor<64xf32>,
                                        %shift: tensor<64xf32>, %out: tensor<3136x64xf32>)
                                        -> tensor<3136x64xf32> {
        %0 = linalg.generic {
          indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                            affine_map<(d0, d1) -> (d1)>,
                            affine_map<(d0, d1) -> (d1)>,
                            affine_map<(d0, d1) -> (d0, d1)>],
          iterator_types = ["parallel", "parallel"]
        } ins(%x, %scale, %shift : tensor<3136x64xf32>, tensor<64xf32>, tensor<64xf32>)
          outs(%out : tensor<3136x64xf32>) {
        ^bb0(%in0: f32, %in1: f32, %in2: f32, %init: f32):
          %mul = arith.mulf %in0, %in1 : f32
          %add = arith.addf %mul, %in2 : f32
          linalg.yield %add : f32
        } -> tensor<3136x64xf32>
        return %0 : tensor<3136x64xf32>
      }
    )mlir",
    DispatchConfig{
        "IreeChosenTiling",
        /*tileSizes=*/{1, 32}, // [d0, d1]
        /*expectedVector=*/3,
        /*expectedFailure=*/"",
    },
    DispatchConfig{
        "ExactRegisterMultiple",
        /*tileSizes=*/{8, 64}, // [d0, d1]
        /*expectedVector=*/20,
        /*expectedFailure=*/"",
        "Measured: 21"
        "the extra register is used for "
        "unaligned loads."
        "in theory, the compiler should not need to use this reg if "
        "everything is aligned. apparently, it fails to prove that here"});

} // namespace
} // namespace mlir::iree_compiler::hexagon::codegen::planning

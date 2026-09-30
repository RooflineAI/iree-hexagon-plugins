// Copyright 2026 RooflineAI GmbH
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "TestUtils.h"

namespace mlir::iree_compiler::hexagon::codegen::planning {
namespace {

/// linalg.batch_matmul, f32. Loops are [batch, M, N, K] = [d0 parallel,
/// d1 parallel, d2 parallel, d3 reduction]; lhs depends on (batch, M, K),
/// rhs on (batch, K, N), out on (batch, M, N).
REGISTER_ESTIMATION_TEST_SUITE(
    BatchMatmul,
    R"mlir(
      func.func @batch_matmul(%lhs: tensor<8x128x128xf32>, %rhs: tensor<8x128x128xf32>,
                               %out: tensor<8x128x128xf32>) -> tensor<8x128x128xf32> {
        %0 = linalg.batch_matmul ins(%lhs, %rhs : tensor<8x128x128xf32>, tensor<8x128x128xf32>)
                                  outs(%out : tensor<8x128x128xf32>) -> tensor<8x128x128xf32>
        return %0 : tensor<8x128x128xf32>
      }
    )mlir",
    DispatchConfig{"IreeChosenTiling",
                   /*tileSizes=*/{1, 8, 32, 8}, // [batch, M, N, K]
                   /*expectedVector=*/10,
                   /*expectedFailure=*/"",
                   "measured HvxVR=11; extra register needed for possible "
                   "unaligned loads"},
    DispatchConfig{
        "BatchMultipliesIntoEveryOperand",
        /*tileSizes=*/{2, 8, 32, 8}, // [batch, M, N, K]
        /*expectedVector=*/19,
        /*expectedFailure=*/"",
    });

/// linalg.batch_matmul with both operands transposed

REGISTER_ESTIMATION_TEST_SUITE(
    BatchMatmulTransposed,
    R"mlir(
    func.func @transposed_batch_matmul(%lhs: tensor<8x128x128xf32>,
                                       %rhs: tensor<8x128x128xf32>,
                                       %out: tensor<8x128x128xf32>) -> tensor<8x128x128xf32> {
      %0 = linalg.batch_matmul
             indexing_maps = [affine_map<(batch, m, n, k) -> (batch, k, m)>,
                              affine_map<(batch, m, n, k) -> (batch, n, k)>,
                              affine_map<(batch, m, n, k) -> (batch, m, n)>]
             ins(%lhs, %rhs : tensor<8x128x128xf32>, tensor<8x128x128xf32>)
             outs(%out : tensor<8x128x128xf32>) -> tensor<8x128x128xf32>
      return %0 : tensor<8x128x128xf32>
    }
  )mlir",
    DispatchConfig{"Trsnaposed1",
                   /*tileSizes=*/{1, 8, 32, 8}, // [batch, M, N, K]
                   // 1 lhs, 8 rhs, 8 accumulator/output
                   // + 8 for the rhs tile's transposed copy, written while the
                   // gathered source is read
                   /*expectedVector=*/25,
                   /*expectedFailure=*/"",
                   "measured HvxVR=39; see Matmul case"},
    DispatchConfig{"Transposed2",
                   /*tileSizes=*/{2, 8, 32, 8}, // [batch, M, N, K]
                   // 2 lhs, 16 rhs, 16 accumulator/output
                   // + 16 for the rhs tile's transposed copy, written while the
                   // gathered source is read
                   /*expectedVector=*/50,
                   /*expectedFailure=*/"",
                   "measured HvxVR=44; see Matmul case"});

} // namespace
} // namespace mlir::iree_compiler::hexagon::codegen::planning

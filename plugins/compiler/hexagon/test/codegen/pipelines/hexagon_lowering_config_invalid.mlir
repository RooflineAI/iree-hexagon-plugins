// Copyright 2026 RooflineAI GmbH
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RUN: iree-opt --pass-pipeline='builtin.module(func.func(iree-hexagon-lower-executable-target))' --verify-diagnostics %s

// Stage ranks agree with one another, but must also match the operation's
// iteration space before any pipeline tiling runs.
func.func @wrong_iteration_rank(%input: tensor<16xf32>, %init: tensor<16xf32>)
    -> tensor<16xf32> attributes {
  hal.executable.target = #hal.executable.target<"hexagon", "embedded-elf-hexagon">,
  translation_info = #iree_codegen.translation_info<pipeline = #iree_hexagon.pipeline<MultiTilingExpert>>
} {
  // expected-error @+1 {{Hexagon lowering config rank must match iteration rank 1}}
  %result = linalg.generic {
    indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
    iterator_types = ["parallel"]}
    ins(%input : tensor<16xf32>) outs(%init : tensor<16xf32>)
    attrs = {lowering_config = #iree_hexagon.lowering_config<
      distribution = [0, 0], vector_common_parallel = [4, 0]>} {
  ^bb0(%in: f32, %out: f32):
    %sum = arith.addf %in, %out : f32
    linalg.yield %sum : f32
  } -> tensor<16xf32>
  return %result : tensor<16xf32>
}

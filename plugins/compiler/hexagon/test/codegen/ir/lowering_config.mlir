// Copyright 2026 RooflineAI GmbH
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RUN: iree-opt --split-input-file --verify-diagnostics %s | FileCheck %s

// Roundtrip every stage, including the all-zero distribution root marker.
// CHECK: #[[CONFIG:.+]] = #iree_hexagon.lowering_config<cache_parallel = [64, 64, 0], cache_reduction = [0, 0, 32], distribution = [0, 0, 0], hmx = [1, 1, 0], vector_common_parallel = [8, 32, 0], vector_inner_parallel = [0, 0, 0], vector_reduction = [0, 0, 8], vtcm = [128, 128, 64]>
// CHECK: module attributes {iree_hexagon.config = #[[CONFIG]]}
module attributes {iree_hexagon.config = #iree_hexagon.lowering_config<
    distribution = [0, 0, 0], cache_parallel = [64, 64, 0],
    cache_reduction = [0, 0, 32], vector_common_parallel = [8, 32, 0],
    vector_reduction = [0, 0, 8], vector_inner_parallel = [0, 0, 0],
    vtcm = [128, 128, 64], hmx = [1, 1, 0]>} {}

// -----

// expected-error @+1 {{unknown Hexagon tiling stage "vector_parallel"}}
module attributes {iree_hexagon.config = #iree_hexagon.lowering_config<vector_parallel = [8]>} {}

// -----

// expected-error @+1 {{expected nonnegative tile sizes for "vector_reduction"}}
module attributes {iree_hexagon.config = #iree_hexagon.lowering_config<vector_reduction = [-1]>} {}

// -----

// expected-error @+1 {{expected all Hexagon tiling stages to have the same rank}}
module attributes {iree_hexagon.config = #iree_hexagon.lowering_config<distribution = [0, 0], vector_reduction = [8]>} {}

// -----

// expected-error @+1 {{duplicate Hexagon tiling stage distribution}}
module attributes {iree_hexagon.config = #iree_hexagon.lowering_config<distribution = [0], distribution = [8]>} {}

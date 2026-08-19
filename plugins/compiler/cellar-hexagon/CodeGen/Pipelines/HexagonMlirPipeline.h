// Copyright 2025 RooflineAI GmbH
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// This header declares the experimental hexagon-mlir-inspired lowering route.

#ifndef ROOF_HEXAGON_CODEGEN_HEXAGONMLIRPIPELINE_H_
#define ROOF_HEXAGON_CODEGEN_HEXAGONMLIRPIPELINE_H_

#include "cellar-hexagon/CodeGen/Passes.h"

namespace mlir::iree_compiler::cellar_hexagon::codegen {

void buildHexagonMlirTranslationRoute(
    mlir::OpPassManager &variantPassManager,
    const HexagonPipelineOptions &pipelineOpt);

void addHexagonMlirLowerToLLVMPasses(mlir::OpPassManager &variantPassManager);

} // namespace mlir::iree_compiler::cellar_hexagon::codegen

#endif // ROOF_HEXAGON_CODEGEN_HEXAGONMLIRPIPELINE_H_

// Copyright 2025 RooflineAI GmbH
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// This header declares the translation-stage Hexagon pipeline builder and
// route-shaping policy accessors owned by TranslationPipeline.cpp.

#ifndef ROOF_HEXAGON_CODEGEN_TRANSLATIONPIPELINE_H_
#define ROOF_HEXAGON_CODEGEN_TRANSLATIONPIPELINE_H_

#include "cellar-hexagon/CodeGen/Passes.h"

namespace mlir::iree_compiler::cellar_hexagon::codegen {

void buildHexagonTranslationPassPipeline(
    mlir::OpPassManager &passManager,
    const HexagonPipelineOptions &pipelineOptions = HexagonPipelineOptions{});

bool isHexKLMatmulLoweringEnabled();
bool isHexagonVTCMTilingEnabled();
bool isHexagonMlirLinalgLoweringEnabled();

} // namespace mlir::iree_compiler::cellar_hexagon::codegen

#endif // ROOF_HEXAGON_CODEGEN_TRANSLATIONPIPELINE_H_

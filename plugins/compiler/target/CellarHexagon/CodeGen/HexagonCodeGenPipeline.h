// Copyright 2020 The IREE Authors
//
// Copyright 2025 RooflineAI GmbH
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https: // llvm.org/LICENSE.txt for license information.
// SPDX - License - Identifier : Apache - 2.0 WITH LLVM - exception

#ifndef ROOF_HEXAGON_CODEGEN_PIPELINE_H_
#define ROOF_HEXAGON_CODEGEN_PIPELINE_H_

#include "mlir/Pass/PassManager.h"

namespace cellar::target::hexagon {

// TODO: Remove or add passes as needed.
// The placeholders are copied from LLVMCPUTarget

// Register Hexagon-specific codegen passes
void registerHexagonCodeGenPasses();

// Placeholder pass pipeline for configuration
void buildHexagonConfigurationPassPipeline(mlir::OpPassManager &passManager);

// Placeholder pass pipeline for translation
void buildHexagonTranslationPassPipeline(mlir::OpPassManager &passManager);

// Placeholder pass pipeline for linking
void buildHexagonLinkingPassPipeline(mlir::OpPassManager &passManager);

} // namespace cellar::target::hexagon

#endif // ROOF_HEXAGON_CODEGEN_PIPELINE_H_

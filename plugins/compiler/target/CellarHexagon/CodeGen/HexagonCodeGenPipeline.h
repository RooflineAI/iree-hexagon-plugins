// Copyright 2025 RooflineAI GmbH
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https: // llvm.org/LICENSE.txt for license information.
// SPDX - License - Identifier : Apache - 2.0 WITH LLVM - exception

#ifndef ROOF_HEXAGON_CODEGEN_PIPELINE_H_
#define ROOF_HEXAGON_CODEGEN_PIPELINE_H_

#include "mlir/Pass/PassManager.h"

namespace cellar::target::hexagon {

// Register Hexagon-specific codegen passes
void registerHexagonCodeGenPasses();

// Builder for the configuration pass pipeline
void buildHexagonConfigurationPassPipeline(mlir::OpPassManager &passManager);

// Builder for the translation pass pipeline
void buildHexagonTranslationPassPipeline(mlir::OpPassManager &passManager);

// Builder for the linker pass pipeline
void buildHexagonLinkingPassPipeline(
    mlir::OpPassManager &passManager,
    std::optional<std::string> target = std::nullopt);

} // namespace cellar::target::hexagon

#endif // ROOF_HEXAGON_CODEGEN_PIPELINE_H_

// Copyright 2025 RooflineAI GmbH
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https: // llvm.org/LICENSE.txt for license information.
// SPDX - License - Identifier : Apache - 2.0 WITH LLVM - exception

#include "HexagonCodeGenPipeline.h"

#include "iree/compiler/Codegen/LLVMCPU/Passes.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/Passes.h"

using namespace mlir;
using namespace mlir::iree_compiler;

// TODO: I do not understand the convention for this namespace, everyone does
// something different...
namespace cellar::target::hexagon {

namespace {

void registerPassPipelines() {
  // Pipelines called in this order by the HAL
  static PassPipelineRegistration<> configurationPipeline(
      "iree-hexagon-configuration-pipeline",
      "Runs the Hexagon HAL configuration pipeline.",
      [](OpPassManager &passManager) {
        buildHexagonConfigurationPassPipeline(passManager);
      });

  static PassPipelineRegistration<> translationPipeline(
      "iree-hexagon-translation-pipeline",
      "Runs the Hexagon HAL translation pipeline.",
      [](OpPassManager &passManager) {
        buildHexagonTranslationPassPipeline(passManager);
      });

  static PassPipelineRegistration<> linkingPipeline(
      "iree-hexagon-linking-pipeline", "Runs the Hexagon HAL linking pipeline.",
      [](OpPassManager &passManager) {
        buildHexagonLinkingPassPipeline(passManager);
      });
}

} // namespace

void registerHexagonCodeGenPasses() { registerPassPipelines(); }

// TODO: The final goal is to extract the minimal subset of necessary passes,
// but I will reuse the same ones that are used in the LLVMCPUTarget for now
void buildHexagonConfigurationPassPipeline(OpPassManager &passManager) {
  buildLLVMCPUCodegenConfigurationPassPipeline(passManager);
}

void buildHexagonTranslationPassPipeline(OpPassManager &passManager) {
  buildLLVMCPUCodegenPassPipeline(passManager);
}

void buildHexagonLinkingPassPipeline(OpPassManager &passManager) {
  buildLLVMCPULinkingPassPipeline(passManager);
}

} // namespace cellar::target::hexagon

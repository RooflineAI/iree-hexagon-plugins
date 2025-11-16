// Copyright 2020 The IREE Authors
//
// Copyright 2025 RooflineAI GmbH
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https: // llvm.org/LICENSE.txt for license information.
// SPDX - License - Identifier : Apache - 2.0 WITH LLVM - exception

#include "HexagonCodeGenPipeline.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/Passes.h"

using namespace mlir;

namespace cellar::target::hexagon {

namespace {

void registerPassPipelines() {
  // TODO: Placeholder pipelines, remove or add passes as needed.
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

void buildHexagonConfigurationPassPipeline(OpPassManager &passManager) {
  passManager.addPass(createCanonicalizerPass());
  passManager.addPass(createCSEPass());
}

void buildHexagonTranslationPassPipeline(OpPassManager &passManager) {
  passManager.addNestedPass<func::FuncOp>(createCanonicalizerPass());
}

void buildHexagonLinkingPassPipeline(OpPassManager &passManager) {
  passManager.addPass(createCanonicalizerPass());
}

} // namespace cellar::target::hexagon

// Copyright 2025 RooflineAI GmbH
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// This file only owns pass/pipeline registration.

#include "hexagon/CodeGen/Passes.h"

#include "hexagon/CodeGen/Pipelines/ConfigurationPipeline.h"
#include "hexagon/CodeGen/Pipelines/LinkingPipeline.h"
#include "hexagon/CodeGen/Pipelines/TranslationPipeline.h"

#include "hexagon/Conversion/LinalgToLLVM/LinalgToLLVM.h"
#include "iree/compiler/Codegen/LLVMCPU/Passes.h"
#include "mlir/Pass/PassRegistry.h"

namespace mlir::iree_compiler::hexagon::codegen {

namespace {
#define GEN_PASS_REGISTRATION
#include "hexagon/CodeGen/Passes.h.inc" // IWYU pragma: keep
} // namespace

void registerHexagonPasses() {
  // Only register the Hexagon-owned passes.
  // LLVMCPU passes reused are already registered via
  // registerCodegenPasses() at global init. Re-registering them here would call
  // registerCPUPipelineBuilder() a second time, which asserts single
  // registration.
  registerPasses();
}

void registerHexagonCodeGenPasses() {
  registerHexagonPasses();

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

} // namespace mlir::iree_compiler::hexagon::codegen

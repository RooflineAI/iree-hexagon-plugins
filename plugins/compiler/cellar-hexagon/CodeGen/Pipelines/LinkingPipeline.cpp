// Copyright 2025 RooflineAI GmbH
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// This file owns HAL executable linking pipeline assembly for Hexagon targets.

#include "cellar-hexagon/CodeGen/Pipelines/LinkingPipeline.h"

#include "iree/compiler/Codegen/LLVMCPU/Passes.h"
#include "mlir/Transforms/Passes.h"

namespace mlir::iree_compiler::cellar_hexagon::codegen {
namespace IREE = mlir::iree_compiler::IREE;

// NOTE: this runs on the top-level program module containing all
// hal.executable ops.
void buildHexagonLinkingPassPipeline(OpPassManager &modulePassManager,
                                     std::optional<std::string> target) {
  // Link together executables. This may produce some IR duplication.
  LLVMCPULinkExecutablesPassOptions linkOptions;
  linkOptions.target = target.value_or("");
  modulePassManager.addPass(createLLVMCPULinkExecutablesPass(linkOptions));

  // Cleanup IR duplication.
  modulePassManager.addNestedPass<IREE::HAL::ExecutableOp>(
      mlir::createCanonicalizerPass());

  // Assign final executable constant and import ordinals.
  auto &variantPassManager = modulePassManager.nest<IREE::HAL::ExecutableOp>()
                                 .nest<IREE::HAL::ExecutableVariantOp>();
  variantPassManager.addPass(createLLVMCPUAssignConstantOrdinalsPass());
  variantPassManager.addPass(createLLVMCPUAssignImportOrdinalsPass());
}

} // namespace mlir::iree_compiler::cellar_hexagon::codegen

// Copyright 2025 RooflineAI GmbH
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// This file owns HAL executable linking pipeline assembly for Hexagon targets.

#include "hexagon/CodeGen/Pipelines/LinkingPipeline.h"

#include "hexagon/CodeGen/Passes.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "mlir/Transforms/Passes.h"

namespace mlir::iree_compiler::hexagon::codegen {
namespace IREE = mlir::iree_compiler::IREE;

void buildHexagonLinkingPassPipeline(OpPassManager &modulePassManager,
                                     std::optional<std::string> target) {
  // Link together executables. This may produce some IR duplication.
  HexagonLinkExecutablesPassOptions linkOptions;
  linkOptions.target = target.value_or("");
  modulePassManager.addPass(createHexagonLinkExecutablesPass(linkOptions));

  // Cleanup IR duplication.
  modulePassManager.addNestedPass<IREE::HAL::ExecutableOp>(
      mlir::createCanonicalizerPass());

  // Assign final executable constant ordinals. Hexagon does not support HAL
  // dynamic imports; external DSP runtime calls remain direct static links.
  auto &variantPassManager = modulePassManager.nest<IREE::HAL::ExecutableOp>()
                                 .nest<IREE::HAL::ExecutableVariantOp>();
  variantPassManager.addPass(createHexagonAssignConstantOrdinalsPass());
}

} // namespace mlir::iree_compiler::hexagon::codegen

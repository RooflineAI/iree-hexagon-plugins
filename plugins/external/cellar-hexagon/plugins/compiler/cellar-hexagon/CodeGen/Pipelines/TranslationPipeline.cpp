// Copyright 2025 RooflineAI GmbH
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// This file owns top-level translation orchestration and route selection
// between the IREE-oriented and hexagon-mlir-oriented lowering paths.

#include "cellar-hexagon/CodeGen/Pipelines/TranslationPipeline.h"

#include "cellar-hexagon/CodeGen/Pipelines/HexagonMlirPipeline.h"
#include "cellar-hexagon/CodeGen/Pipelines/IreeLoweringPipelines.h"
#include "iree/compiler/Codegen/Common/Passes.h"
#include "iree/compiler/Dialect/Util/Transforms/Passes.h"
#include "iree/compiler/Utils/PassUtils.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "iree-hexagon-pass-pipelines"

namespace mlir::iree_compiler::cellar_hexagon::codegen {
namespace IREE = mlir::iree_compiler::IREE;

enum class TranslationRoute {
  IreePath,
  HexagonMlirPath,
};

static llvm::cl::opt<bool> clHexagonEnableHexKLMatmulLowering(
    "iree-hexagon-enable-hexkl-matmul-lowering",
    llvm::cl::desc("Enable matmul lowering through the HexKL pipeline "
                   "(linalg.matmul -> hexkl.matmul -> LLVM calls)."),
    llvm::cl::init(false));

static llvm::cl::opt<bool> clHexagonEnableVTCMTiling(
    "iree-hexagon-enable-vtcm-tiling",
    llvm::cl::desc("Enable Hexagon VTCM tiling before bufferization in the "
                   "pipelines using it."),
    llvm::cl::init(false));

static llvm::cl::opt<bool> clHexagonUseHexagonMlirLinalgLowering(
    "iree-hexagon-use-hexagon-mlir-linalg-lowering",
    llvm::cl::desc("Replace IREE's lowering by hexagon mlir's completely."),
    llvm::cl::init(false));

bool isHexKLMatmulLoweringEnabled() {
  return clHexagonEnableHexKLMatmulLowering;
}

bool isHexagonVTCMTilingEnabled() { return clHexagonEnableVTCMTiling; }

bool isHexagonMlirLinalgLoweringEnabled() {
  return clHexagonUseHexagonMlirLinalgLowering;
}

TranslationRoute getTranslationRoute() {
  return isHexagonMlirLinalgLoweringEnabled()
             ? TranslationRoute::HexagonMlirPath
             : TranslationRoute::IreePath;
}

void buildHexagonTranslationPassPipeline(
    OpPassManager &variantPassManager,
    const HexagonPipelineOptions &pipelineOptions) {
  TranslationRoute route = getTranslationRoute();
  switch (route) {
  case TranslationRoute::IreePath:
    buildHexagonIreeTranslationRoute(variantPassManager, pipelineOptions);
    break;
  case TranslationRoute::HexagonMlirPath:
    buildHexagonMlirTranslationRoute(variantPassManager, pipelineOptions);
    break;
  }

  addHexagonVariantFinalizationPasses(variantPassManager);

  switch (route) {
  case TranslationRoute::IreePath: {
    OpPassManager &modulePassManager = variantPassManager.nest<ModuleOp>();
    addHexagonLowerToLLVMPasses(modulePassManager);
    break;
  }
  case TranslationRoute::HexagonMlirPath:
    addHexagonMlirLowerToLLVMPasses(variantPassManager);
    break;
  }

  LLVM_DEBUG({
    llvm::dbgs() << "Hexagon codegen pass pipeline:\n";
    variantPassManager.printAsTextualPipeline(llvm::dbgs());
    llvm::dbgs() << "\n";
  });
}

} // namespace mlir::iree_compiler::cellar_hexagon::codegen

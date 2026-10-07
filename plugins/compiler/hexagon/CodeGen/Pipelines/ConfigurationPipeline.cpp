// Copyright 2025 RooflineAI GmbH
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// This file owns configuration-stage preprocessing and lowering-strategy
// selection for Hexagon executables.

#include "hexagon/CodeGen/Pipelines/ConfigurationPipeline.h"

#include "hexagon/CodeGen/Passes.h"
#include "hexagon/CodeGen/Pipelines/TranslationPipeline.h"

#include "hexagon/Transforms/Transforms.h"
#include "iree/compiler/Codegen/Common/CPU/Passes.h"
#include "iree/compiler/Codegen/Common/Passes.h"
#include "iree/compiler/Utils/PassUtils.h"
#include "mlir/Pass/PassManager.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "iree-hexagon-pass-pipelines"

namespace mlir::iree_compiler::hexagon::codegen {

static llvm::cl::opt<bool> clHexagonUseSoftmaxInterFusion(
    "iree-hexagon-use-decompose-softmax-fuse",
    llvm::cl::desc("Enables inter-pass fusion for the DecomposeSoftmax pass."),
    llvm::cl::init(true));

namespace {

static void
buildHexagonCodegenConfigurationPassPipeline(OpPassManager &modulePassManager) {
  {
    FunctionLikeNest funcPassManager(modulePassManager);
    addCommonTargetExecutablePreprocessingPasses(
        funcPassManager, clHexagonUseSoftmaxInterFusion);
  }

  modulePassManager.addPass(createMaterializeUserConfigsPass());

  FunctionLikeNest(modulePassManager)
      // Without data-tiling encodings, MaterializeDeviceEncoding and
      // CPUPropagateDataLayout are no-ops. They will be restructured in the
      // future.
      .addPass(createMaterializeDeviceEncodingPass)
      .addPass(createCPUPropagateDataLayoutPass)
      .addPass(createRematerializeParallelOpsPass)
      .addPass(createConvertAccGEMMToGEMMPass)
      .addPass(createEraseHALDescriptorTypeFromMemRefPass);

  modulePassManager.addPass(createHexagonSelectLoweringStrategyPass());
  LLVM_DEBUG({
    llvm::dbgs() << "Hexagon codegen configuration pass pipeline:\n";
    modulePassManager.printAsTextualPipeline(llvm::dbgs());
    llvm::dbgs() << "\n";
  });
}

} // namespace

void buildHexagonConfigurationPassPipeline(OpPassManager &variantPassManager) {
  buildCodegenConfigurationPreProcessingPassPipeline(variantPassManager);
  OpPassManager &modulePassManager = variantPassManager.nest<ModuleOp>();
  buildHexagonCodegenConfigurationPassPipeline(modulePassManager);
}

} // namespace mlir::iree_compiler::hexagon::codegen

// Copyright 2025 RooflineAI GmbH
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// This file owns configuration-stage preprocessing and lowering-strategy
// selection for Hexagon executables.

#include "cellar-hexagon/CodeGen/Pipelines/ConfigurationPipeline.h"

#include "cellar-hexagon/CodeGen/Passes.h"
#include "cellar-hexagon/CodeGen/Pipelines/TranslationPipeline.h"

#include "hexagon/Transforms/Transforms.h"
#include "iree/compiler/Codegen/Common/CPU/Passes.h"
#include "iree/compiler/Codegen/Common/Passes.h"
#include "iree/compiler/Codegen/LLVMCPU/Passes.h"
#include "iree/compiler/Utils/PassUtils.h"
#include "mlir/Pass/PassManager.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "iree-hexagon-pass-pipelines"

namespace mlir::iree_compiler::cellar_hexagon::codegen {

enum class LaunchConfigSelector {
  LLVMCPU,
  Hexagon,
};

static llvm::cl::opt<bool> clHexagonUseSoftmaxInterFusion(
    "iree-hexagon-use-decompose-softmax-fuse",
    llvm::cl::desc("Enables inter-pass fusion for the DecomposeSoftmax pass."),
    llvm::cl::init(true));

static llvm::cl::opt<LaunchConfigSelector> clHexagonLaunchConfigSelector(
    "iree-hexagon-launch-config-selector",
    llvm::cl::desc("Select which pass provides the Hexagon launch-config "
                   "strategy."),
    llvm::cl::values(
        clEnumValN(LaunchConfigSelector::LLVMCPU, "llvmcpu",
                   "Use the upstream LLVMCPU lowering-strategy selector."),
        clEnumValN(LaunchConfigSelector::Hexagon, "hexagon",
                   "Use the Hexagon lowering-strategy selector.")),
    llvm::cl::init(LaunchConfigSelector::LLVMCPU));

namespace {

void addLaunchConfigSelectionPass(OpPassManager &modulePassManager) {
  // Hexagon now owns the launch-config policy selection step. The pass still
  // emits the standard IREE CPU lowering attrs/pipeline enums so the rest of
  // the Hexagon lowering stack can remain unchanged while we compare Hexagon
  // policy decisions against the original LLVMCPU heuristics.
  // modulePassManager.addPass(createHexagonSelectLoweringStrategyPass());
  switch (clHexagonLaunchConfigSelector) {
  case LaunchConfigSelector::Hexagon:
    modulePassManager.addPass(createHexagonSelectLoweringStrategyPass());
    return;
  case LaunchConfigSelector::LLVMCPU:
    modulePassManager.addPass(createLLVMCPUSelectLoweringStrategyPass());
    return;
  }
}

static void
buildHexagonCodegenConfigurationPassPipeline(OpPassManager &modulePassManager) {
  {
    FunctionLikeNest funcPassManager(modulePassManager);
    addCommonTargetExecutablePreprocessingPasses(
        funcPassManager, clHexagonUseSoftmaxInterFusion);
  }
  // TODO: Data tiling is completely removed, so this might potentially be
  // removed, along with the corresponding passes
  modulePassManager.addPass(createMaterializeUserConfigsPass());

  // If reusing hexagon-mlir's lowering pipeline, this is done later down the
  // pipeline. Otherwise have to do it this early to avoid LLVMCPU copied
  // functionality start taking decisions based on the operation types.
  if (isHexKLMatmulLoweringEnabled() && !isHexagonMlirLinalgLoweringEnabled()) {
    // Here we reduced batched_matmuls with dimensionality 1 to matmul ops,
    // before converting matmul ops to hexkl calls
    modulePassManager.addNestedPass<func::FuncOp>(
        mlir::hexagon::createReduceContractionRankPass());
    modulePassManager.addNestedPass<func::FuncOp>(
        mlir::hexagon::createMatmulToHexKLPass());
    // Must disable data tiling, or remove the encoding to ensure there are
    // no conflicts in the pipeline when creating hexkl.matmul calls.
  }

  FunctionLikeNest(modulePassManager)
      .addPass(createMaterializeDeviceEncodingPass)
      .addPass(createCPUPropagateDataLayoutPass)
      .addPass(createRematerializeParallelOpsPass)
      // This pass is removed for hexagon
      // .addPass(createExpandF16OpToF32Pass)
      .addPass(createConvertAccGEMMToGEMMPass)
      .addPass(createEraseHALDescriptorTypeFromMemRefPass);

  addLaunchConfigSelectionPass(modulePassManager);
  LLVM_DEBUG({
    llvm::dbgs() << "Hexagon codegen configuration pass pipeline:\n";
    modulePassManager.printAsTextualPipeline(llvm::dbgs());
    llvm::dbgs() << "\n";
  });
}

} // namespace

void buildHexagonConfigurationPassPipeline(OpPassManager &variantPassManager) {
  variantPassManager.addPass(createSpecializeExportsPass());
  OpPassManager &modulePassManager = variantPassManager.nest<ModuleOp>();
  buildHexagonCodegenConfigurationPassPipeline(modulePassManager);
}

} // namespace mlir::iree_compiler::cellar_hexagon::codegen

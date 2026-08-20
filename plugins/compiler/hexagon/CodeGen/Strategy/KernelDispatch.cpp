// Copyright 2026 RooflineAI GmbH
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "KernelDispatch.h"
#include "KernelDispatchEncoding.h"
#include "KernelDispatchHeuristics.h"
#include "KernelDispatchPropagation.h"

#include "iree/compiler/Codegen/Utils/CPUUtils.h"
#include "iree/compiler/Codegen/Utils/Utils.h"
#include "mlir/Dialect/MemRef/Transforms/Transforms.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/STLExtras.h"

#define DEBUG_TYPE "iree-hexagon-kernel-dispatch"

namespace mlir::iree_compiler::hexagon::codegen {
namespace {

using mlir::failure;
using mlir::FunctionOpInterface;
using mlir::LogicalResult;
using mlir::Operation;
using mlir::iree_compiler::getComputeOps;
using mlir::iree_compiler::getLoweringConfig;
using mlir::iree_compiler::getRootOperation;
using mlir::iree_compiler::getTranslationInfo;
using mlir::iree_compiler::setTranslationInfo;

static LogicalResult
lowerUsingDefaultPipeline(FunctionOpInterface entryPointFn) {
  if (getTranslationInfo(entryPointFn)) {
    return mlir::success();
  }
  MLIRContext *ctx = entryPointFn.getContext();
  auto translationInfo =
      mlir::iree_compiler::IREE::Codegen::TranslationInfoAttr::get(
          ctx,
          mlir::iree_compiler::IREE::CPU::PipelineAttr::get(
              ctx, mlir::iree_compiler::IREE::CPU::LoweringPipeline::Default),
          mlir::SymbolRefAttr(), /*workgroupSize=*/{}, /*subgroupSize=*/0,
          /*configuration=*/mlir::DictionaryAttr());
  return setTranslationInfo(entryPointFn, translationInfo);
}

// This function is made in the image of LLVMCPU's equivalent.
// The idea is to select a root operation, define a configuration for it and
// then propagate to other operations in the dispatch based on it.
// Hexagon reuses the same pattern and tries to simplify and adapt it to the
// architecture.
// Contrary to LLVMCPU though, Hexagon splits the responsibilities from the
// equivalent kernelDispatch.cpp file.
// In LLVMCPU, propagation may influence the root operation initially chose.
// In Hexagon's current state, only the root operation influences others during
// propagation.
static LogicalResult
runRootSelectionAndPropagation(FunctionOpInterface entryPointFn,
                               llvm::ArrayRef<Operation *> computeOps) {
  for (Operation *computeOp : computeOps) {
    if (getLoweringConfig(computeOp)) {
      return failure();
    }
  }

  auto rootOp = getRootOperation(computeOps);
  if (mlir::failed(rootOp)) {
    return failure();
  }
  Operation *rootOperation = rootOp.value();
  if (!rootOperation) {
    return lowerUsingDefaultPipeline(entryPointFn);
  }

  RootLoweringPlan rootPlan =
      selectRootLoweringPlan(entryPointFn, rootOperation);

  if (!rootPlan.opPlan) {
    return lowerUsingDefaultPipeline(entryPointFn);
  }

  if (mlir::failed(
          applyRootLoweringPlan(entryPointFn, rootOperation, rootPlan))) {
    return failure();
  }

  NonRootPlanPropagator propagator(entryPointFn, rootOperation, computeOps,
                                   rootPlan);
  for (auto &[op, opPlan] : propagator.inferPlans()) {
    applyOpLoweringPlan(op, opPlan);
  }
  return mlir::success();
}

} // namespace

LogicalResult initHexagonLaunchConfig(FunctionOpInterface funcOp) {
  if (getTranslationInfo(funcOp)) {
    return mlir::success();
  }

  if (funcOp.empty() || !llvm::hasSingleElement(funcOp.getFunctionBody())) {
    return lowerUsingDefaultPipeline(funcOp);
  }

  llvm::SmallVector<Operation *> computeOps = getComputeOps(funcOp);
  if (computeOps.empty()) {
    return lowerUsingDefaultPipeline(funcOp);
  }
  if (mlir::failed(runRootSelectionAndPropagation(funcOp, computeOps))) {
    return failure();
  }

  mlir::RewritePatternSet patterns(funcOp.getContext());
  mlir::memref::populateResolveRankedShapedTypeResultDimsPatterns(patterns);
  return mlir::applyPatternsGreedily(funcOp, std::move(patterns));
}

} // namespace mlir::iree_compiler::hexagon::codegen

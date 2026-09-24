// Copyright 2026 RooflineAI GmbH
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "DispatchPlanner.h"

#include "ComputeTileSelection.h"
#include "DecisionTrace.h"
#include "DispatchAnalysis.h"
#include "PipelineContract.h"
#include "PlanEncoding.h"
#include "PlanVerification.h"
#include "RootTilePropagation.h"
#include "StrategySelection.h"
#include "VTCMPlanning.h"

#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "iree/compiler/Codegen/LLVMCPU/Utils.h"
#include "iree/compiler/Codegen/Utils/Utils.h"
#include "iree/compiler/Dialect/HAL/IR/HALTypes.h"

namespace mlir::iree_compiler::hexagon::codegen::planning {
namespace {

TargetInfo getTargetInfo(FunctionOpInterface entryPoint) {
  TargetInfo target;
  auto targetAttr = IREE::HAL::ExecutableTargetAttr::lookup(entryPoint);
  if (!targetAttr)
    return target;
  DictionaryAttr configuration = targetAttr.getConfiguration();
  if (std::optional<int64_t> nativeVector =
          getConfigNativeVectorSize(configuration))
    target.nativeVectorBytes = std::max<int64_t>(1, *nativeVector);
  if (hasFeature(configuration, "+hvx-length128b"))
    target.nativeVectorBytes = 128;
  else if (hasFeature(configuration, "+hvx-length64b"))
    target.nativeVectorBytes = 64;
  return target;
}

LogicalResult configureDefaultPipeline(FunctionOpInterface entryPoint) {
  EncodedDispatchPlan encoded;
  encoded.entryPoint = entryPoint;
  MLIRContext *context = entryPoint.getContext();
  encoded.translationInfo = IREE::Codegen::TranslationInfoAttr::get(
      context,
      IREE::CPU::PipelineAttr::get(context,
                                   IREE::CPU::LoweringPipeline::Default),
      SymbolRefAttr(), /*workgroupSize=*/{}, /*subgroupSize=*/0,
      /*configuration=*/DictionaryAttr());
  return applyEncodedDispatchPlan(encoded);
}

} // namespace

// The orchestration order and stage contracts are documented in README.md.
// Update its Planning flow table when adding, removing, or reordering stages.
LogicalResult configureDispatch(FunctionOpInterface entryPoint,
                                const PlanningOptions &options) {
  if (getTranslationInfo(entryPoint))
    return success();
  if (entryPoint.empty() ||
      !llvm::hasSingleElement(entryPoint.getFunctionBody()))
    return configureDefaultPipeline(entryPoint);

  SmallVector<Operation *> computeOps = getComputeOps(entryPoint);
  if (computeOps.empty())
    return configureDefaultPipeline(entryPoint);
  for (Operation *op : computeOps) {
    if (getLoweringConfig(op) ||
        op->hasAttr(kHexagonVTCMTilingConfigAttrName)) {
      op->emitError("expected an unconfigured operation before Hexagon "
                    "dispatch planning");
      return failure();
    }
  }

  DecisionTrace trace;
  PlanningContext context{entryPoint, getTargetInfo(entryPoint), options,
                          trace};
  FailureOr<DispatchShape> shape = analyzeDispatch(context, computeOps);
  if (failed(shape))
    return failure();
  if (!shape->root)
    return configureDefaultPipeline(entryPoint);

  FailureOr<DispatchStrategy> strategy =
      selectDispatchStrategy(context, *shape);
  if (failed(strategy))
    return failure();

  FailureOr<PipelineContract> contract =
      getPipelineContract(strategy->pipeline);
  if (failed(contract)) {
    entryPoint.emitError("selected Hexagon pipeline has no planning contract");
    return failure();
  }

  if (failed(planVTCMTiling(context, *shape, *contract, *strategy)))
    return failure();

  SmallVector<OpComputeTilePlan> nonRootComputeTilePlans;
  for (const OpShape &opShape : shape->operations) {
    if (opShape.op == shape->root)
      continue;

    FailureOr<std::optional<OpComputeTilePlan>> selected =
        selectNonRootComputeTile(context, *shape, *strategy, *contract,
                                 opShape);
    if (failed(selected))
      return failure();
    if (*selected)
      nonRootComputeTilePlans.push_back(std::move(**selected));
  }

  DispatchPlan plan{std::move(*strategy), std::move(nonRootComputeTilePlans)};
  if (failed(reconcileNonRootComputeTiles(context, *shape, *contract, plan)))
    return failure();

  if (failed(verifyDispatchPlan(context, *shape, plan, *contract)))
    return failure();

  FailureOr<EncodedDispatchPlan> encoded =
      encodeDispatchPlan(context, *shape, plan, *contract);
  if (failed(encoded))
    return failure();

  return applyEncodedDispatchPlan(*encoded);
}

} // namespace mlir::iree_compiler::hexagon::codegen::planning

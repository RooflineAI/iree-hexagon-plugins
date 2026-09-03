// Copyright 2026 RooflineAI GmbH
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "StrategySelection.h"

#include "DecisionTrace.h"
#include "Strategies/Strategy.h"

#include "llvm/ADT/Twine.h"

namespace mlir::iree_compiler::hexagon::codegen::planning {

static const StrategyCandidate candidates[] = {
    {"buffer-copy", selectBufferCopyStrategy},
    {"convolution", selectConvolutionStrategy},
    {"hmx-contraction", selectHmxStrategy},
    {"contraction", selectContractionStrategy},
    {"unsupported-contraction", selectUnsupportedContractionFallback},
    {"generic", selectGenericStrategy},
    {"tiling-interface", selectDefaultTilingInterfaceStrategy},
};

FailureOr<DispatchStrategy>
selectDispatchStrategy(const PlanningContext &context,
                       const DispatchShape &dispatchShape) {
  const OpShape *rootShape = findOpShape(dispatchShape, dispatchShape.root);
  if (!rootShape) {
    context.trace.record(
        DecisionStage::Fallback, DecisionKind::Fallback,
        "selected CPUDefault because the dispatch has no analyzable root");
    return makeCPUDefaultStrategy(dispatchShape);
  }

  std::optional<DispatchStrategy> selectedStrategy;
  llvm::StringRef selectedCandidateName;
  for (const StrategyCandidate &candidate : candidates) {
    FailureOr<std::optional<DispatchStrategy>> selected =
        candidate.select(context, dispatchShape);
    if (failed(selected))
      return failure();
    if (!*selected) {
      context.trace.record(DecisionStage::Strategy, DecisionKind::Rejected,
                           Twine("candidate ") + candidate.name +
                               " did not match");
      continue;
    }

    selectedStrategy = std::move(**selected);
    selectedCandidateName = candidate.name;
    break;
  }

  if (!selectedStrategy) {
    context.trace.record(DecisionStage::Fallback, DecisionKind::Fallback,
                         "selected CPUDefault because no strategy matched");
    return makeCPUDefaultStrategy(dispatchShape);
  }

  context.trace.recordForOp(
      DecisionStage::Strategy, DecisionKind::Selected, rootShape->ordinal,
      dispatchShape.root->getName().getStringRef(),
      Twine("selected ") + selectedCandidateName + " strategy");
  context.trace.recordTilePlan(
      DecisionStage::Strategy, DecisionKind::Selected, rootShape->ordinal,
      dispatchShape.root->getName().getStringRef(), "root distribution tile",
      selectedStrategy->rootTiling.distributionTile);
  context.trace.recordTilePlan(
      DecisionStage::Strategy, DecisionKind::Selected, rootShape->ordinal,
      dispatchShape.root->getName().getStringRef(), "root cache tile",
      selectedStrategy->rootTiling.cacheTile);
  context.trace.recordTilePlan(
      DecisionStage::Strategy, DecisionKind::Selected, rootShape->ordinal,
      dispatchShape.root->getName().getStringRef(), "root compute tile",
      selectedStrategy->rootTiling.computeTile);
  return std::move(*selectedStrategy);
}

} // namespace mlir::iree_compiler::hexagon::codegen::planning

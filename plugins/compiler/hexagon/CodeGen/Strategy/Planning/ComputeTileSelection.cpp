// Copyright 2026 RooflineAI GmbH
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "ComputeTileSelection.h"

#include "DecisionTrace.h"
#include "Strategies/Strategy.h"
#include "Strategies/StrategySupport.h"

#include "llvm/ADT/Twine.h"

namespace mlir::iree_compiler::hexagon::codegen::planning {

static const TilePolicy policies[] = {
    {"buffer-copy", selectBufferCopyComputeTile},
    {"convolution", selectConvolutionComputeTile},
    {"contraction", selectContractionComputeTile},
    {"generic", selectGenericComputeTile},
    {"tiling-interface", selectDefaultTilingInterfaceComputeTile},
};

FailureOr<std::optional<OpComputeTilePlan>> selectNonRootComputeTile(
    const PlanningContext &context, const DispatchShape &dispatchShape,
    const DispatchStrategy &strategy, const PipelineContract &pipelineContract,
    const OpShape &opShape) {
  if (opShape.op == dispatchShape.root)
    return std::optional<OpComputeTilePlan>{};
  if (!pipelineContract.supportsIndependentNonRootComputeTiles) {
    context.trace.recordForOp(
        DecisionStage::ComputeTile, DecisionKind::Rejected, opShape.ordinal,
        opShape.op->getName().getStringRef(),
        "selected pipeline cannot realize an independent non-root compute "
        "tile; "
        "omitting lowering config");
    return std::optional<OpComputeTilePlan>{};
  }

  for (const TilePolicy &policy : policies) {
    FailureOr<std::optional<OpComputeTilePlan>> selected = policy.select(
        context, dispatchShape, strategy, pipelineContract, opShape);
    if (failed(selected))
      return failure();
    if (!*selected)
      continue;
    context.trace.recordForOp(
        DecisionStage::ComputeTile, DecisionKind::Selected, opShape.ordinal,
        opShape.op->getName().getStringRef(),
        Twine("selected ") + policy.name + " compute tile");
    context.trace.recordTilePlan(
        DecisionStage::ComputeTile, DecisionKind::Selected, opShape.ordinal,
        opShape.op->getName().getStringRef(), "operation compute tile",
        (**selected).computeTile);
    return std::optional<OpComputeTilePlan>(std::move(**selected));
  }
  context.trace.recordForOp(DecisionStage::ComputeTile, DecisionKind::Rejected,
                            opShape.ordinal,
                            opShape.op->getName().getStringRef(),
                            "no independent compute-tile policy matched");
  return std::optional<OpComputeTilePlan>{};
}

} // namespace mlir::iree_compiler::hexagon::codegen::planning

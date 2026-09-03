// Copyright 2026 RooflineAI GmbH
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "VTCMPlanning.h"

#include "DecisionTrace.h"

#include "hexagon/Conversion/LinalgToLLVM/VTCMTilingOptions.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Twine.h"

namespace mlir::iree_compiler::hexagon::codegen::planning {
namespace {

LogicalResult handleUnavailableVTCM(const PlanningContext &context,
                                    const DispatchShape &dispatchShape,
                                    const PipelineContract &contract,
                                    const Twine &requiredDiagnostic,
                                    const Twine &traceMessage) {
  if (contract.vtcmRequirement == VTCMRequirement::Required) {
    dispatchShape.root->emitError(requiredDiagnostic);
    return failure();
  }
  context.trace.record(DecisionStage::Resource, DecisionKind::Rejected,
                       Twine("optional VTCM plan unavailable: ") +
                           traceMessage);
  return success();
}

LogicalResult suppressCacheTiling(const PlanningContext &context,
                                  const DispatchShape &dispatchShape,
                                  DispatchStrategy &strategy) {
  const OpShape *rootShape = findOpShape(dispatchShape, dispatchShape.root);
  if (!rootShape)
    return failure();
  for (auto [dimension, tile] :
       llvm::enumerate(strategy.rootTiling.cacheTile)) {
    if (tile.size == 0)
      continue;
    if (tile.hardwareFixed) {
      dispatchShape.root->emitError(
          "cannot suppress a hardware-fixed Hexagon cache tile for VTCM");
      return failure();
    }
    int64_t previousSize = tile.size;
    tile.size = 0;
    context.trace.recordAdjustment(
        DecisionStage::Resource, rootShape->ordinal,
        dispatchShape.root->getName().getStringRef(), "root cache tile",
        dimension, previousSize, 0,
        "successful VTCM staging suppresses cache tiling");
  }
  return success();
}

} // namespace

LogicalResult planVTCMTiling(const PlanningContext &context,
                             const DispatchShape &dispatchShape,
                             const PipelineContract &contract,
                             DispatchStrategy &strategy) {
  if (contract.vtcmRequirement == VTCMRequirement::Unsupported)
    return success();
  if (!context.options.enableVTCM) {
    return handleUnavailableVTCM(
        context, dispatchShape, contract,
        "selected Hexagon pipeline requires VTCM tiling",
        "VTCM tiling is disabled");
  }

  auto linalgOp = dyn_cast_or_null<linalg::LinalgOp>(dispatchShape.root);
  if (!linalgOp || !linalgOp.hasPureTensorSemantics()) {
    return handleUnavailableVTCM(
        context, dispatchShape, contract,
        "selected Hexagon pipeline requires a tensor-semantics root for VTCM "
        "tiling",
        "root is not a tensor-semantics Linalg operation");
  }
  // VTCM capacity and layout policy remain owned by the Hexagon lowering
  // utility; this planning stage records its result in the dispatch plan.
  std::optional<SmallVector<int64_t>> tileSizes =
      hexagon::determineTileSizes(linalgOp);
  if (!tileSizes) {
    return handleUnavailableVTCM(context, dispatchShape, contract,
                                 "failed to derive required VTCM tile sizes",
                                 "failed to derive VTCM tile sizes");
  }

  VTCMPlan vtcm;
  for (int64_t tile : *tileSizes)
    vtcm.tileSizes.push_back(TileDecision{tile});
  strategy.rootTiling.vtcm = std::move(vtcm);

  if (contract.cacheTilingWithVTCM == CacheTilingWithVTCM::Suppress &&
      failed(suppressCacheTiling(context, dispatchShape, strategy)))
    return failure();

  const OpShape *rootShape = findOpShape(dispatchShape, dispatchShape.root);
  if (!rootShape)
    return failure();
  context.trace.recordTilePlan(
      DecisionStage::Resource, DecisionKind::Selected, rootShape->ordinal,
      dispatchShape.root->getName().getStringRef(), "root VTCM tile",
      strategy.rootTiling.vtcm->tileSizes);
  return success();
}

} // namespace mlir::iree_compiler::hexagon::codegen::planning

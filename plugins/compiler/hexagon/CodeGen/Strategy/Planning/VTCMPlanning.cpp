// Copyright 2026 RooflineAI GmbH
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "VTCMPlanning.h"

#include "DecisionTrace.h"

#include "hexagon/Conversion/LinalgToLLVM/Common.h"
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
    context.trace.record(DecisionStage::Resource, DecisionKind::Rejected,
                         Twine("required VTCM plan unavailable: ") +
                             traceMessage);
    dispatchShape.root->emitError(requiredDiagnostic);
    return failure();
  }
  // The pipeline keeps its ordinary, non-VTCM strategy.
  context.trace.record(DecisionStage::Resource, DecisionKind::Fallback,
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

/// Why hexagon-mlir's footprint model cannot bound what HexagonVTCMTilingPass
/// stages for `op`, or std::nullopt if it can.
std::optional<StringRef> getUnboundedFootprintReason(linalg::LinalgOp op) {
  // The footprint is computed from the operands' element sizes.
  if (!llvm::all_of(op->getOperandTypes(), [](Type type) {
        return ::mlir::hexagon::getElementSizeInBytes(type).has_value();
      }))
    return StringRef("unsupported operand element type");
  // hexagon-mlir sizes a dynamic dimension as 256. That bounds the tile of a
  // parallel dimension, but a reduction is staged whole at its runtime extent.
  for (auto [range, iterator] :
       llvm::zip_equal(op.getStaticLoopRanges(), op.getIteratorTypesArray())) {
    if (linalg::isReductionIterator(iterator) && ShapedType::isDynamic(range))
      return StringRef("dynamic reduction dimensions are staged whole at their "
                       "runtime extent, which no static plan bounds");
  }
  return std::nullopt;
}

/// Whether `tiles` leaves every reduction dimension of `op` at its full static
/// extent.
bool keepsReductionsWhole(linalg::LinalgOp op, ArrayRef<int64_t> tiles) {
  for (auto [tile, range, iterator] : llvm::zip_equal(
           tiles, op.getStaticLoopRanges(), op.getIteratorTypesArray())) {
    if (linalg::isReductionIterator(iterator) && tile != range)
      return false;
  }
  return true;
}

/// Returns VTCM tile sizes for `op` that HexagonVTCMTilingPass can realize, or
/// std::nullopt if hexagon-mlir finds none that fit in VTCM.
///
/// The pass distributes only the parallel loops over its scf.forall and stages
/// every reduction dimension whole. A tile is therefore only valid if it keeps
/// each reduction at its full extent: only then is the footprint that
/// hexagon-mlir checks against VTCM capacity the one that is actually staged.
///
/// 1. The unrestricted search runs first, and its result is kept if it keeps
///    the reductions whole. This is hexagon-mlir's own priority order (each
///    dimension ranked by how much halving it shrinks the footprint, parallel
///    dimensions before reductions), so every root that fits without tiling a
///    reduction keeps the plan it always had.
/// 2. Otherwise the search is rerun over the parallel dimensions only, so it
///    cannot shrink a reduction. hexagon-mlir visits an explicit dimension
///    list in the order given, not in its ranked order; that is why this is
///    the fallback rather than the only search. An empty list would mean "all
///    dimensions" to hexagon-mlir, so a root with no parallel dimension stops
///    after step 1.
///
/// No fit means the reductions alone are too large for VTCM. The search is a
/// heuristic (it visits each dimension at most twice), so in rare cases it can
/// miss a fit that unit parallel tiles would give.
std::optional<SmallVector<int64_t>>
determineStagedTileSizes(const PlanningContext &context, linalg::LinalgOp op) {
  std::optional<SmallVector<int64_t>> tiles =
      ::mlir::hexagon::determineTileSizes(op);
  if (tiles && keepsReductionsWhole(op, *tiles))
    return tiles;

  SmallVector<unsigned> parallelDims;
  op.getParallelDims(parallelDims);
  if (parallelDims.empty())
    return std::nullopt;
  context.trace.record(DecisionStage::Resource, DecisionKind::Adjusted,
                       "VTCM tile search found no tile that keeps the "
                       "reduction dimensions whole; "
                       "retrying over parallel dimensions only");
  return ::mlir::hexagon::determineTileSizes(
      op, llvm::to_vector_of<int64_t>(parallelDims));
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
  if (std::optional<StringRef> reason = getUnboundedFootprintReason(linalgOp)) {
    return handleUnavailableVTCM(
        context, dispatchShape, contract,
        Twine("failed to derive required VTCM tile sizes: ") + *reason,
        *reason);
  }
  // VTCM capacity and layout policy remain owned by the Hexagon lowering
  // utility; this planning stage only restricts it to tiles the VTCM tiling
  // pass can realize, and records the result in the dispatch plan.
  std::optional<SmallVector<int64_t>> tileSizes =
      determineStagedTileSizes(context, linalgOp);
  if (!tileSizes) {
    return handleUnavailableVTCM(
        context, dispatchShape, contract,
        "failed to derive required VTCM tile sizes: the root does not fit in "
        "VTCM with its reduction dimensions staged whole",
        "root does not fit with its reduction dimensions staged whole");
  }

  VTCMPlan vtcm;
  for (int64_t tile : *tileSizes)
    vtcm.tileSizes.push_back(TileDecision{tile});
  strategy.rootTiling.vtcm = std::move(vtcm);

  if (failed(suppressCacheTiling(context, dispatchShape, strategy)))
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

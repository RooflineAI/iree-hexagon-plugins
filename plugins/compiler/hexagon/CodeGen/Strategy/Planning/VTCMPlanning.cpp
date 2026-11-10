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
#include "mlir/Interfaces/ValueBoundsOpInterface.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Twine.h"

#include <tuple>

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

/// The static extent of each loop of `op`, or for a dynamic loop the constant
/// upper bound of an operand dimension it indexes directly (e.g. the maximum
/// from a `util.assume.int`), or kDynamic if there is none.
SmallVector<int64_t> getLoopUpperBounds(linalg::LinalgOp op) {
  SmallVector<int64_t> bounds = op.getStaticLoopRanges();
  for (auto [loop, bound] : llvm::enumerate(bounds)) {
    if (!ShapedType::isDynamic(bound))
      continue; // already bounded
    for (OpOperand &operand : op->getOpOperands()) {
      if (!isa<ShapedType>(operand.get().getType()))
        continue;
      AffineMap map = op.getMatchingIndexingMap(&operand);
      std::optional<unsigned> dim =
          map.getResultPosition(getAffineDimExpr(loop, op->getContext()));
      if (!dim)
        continue;
      FailureOr<int64_t> upperBound =
          ValueBoundsConstraintSet::computeConstantBound(
              presburger::BoundType::UB, {operand.get(), int64_t(*dim)},
              /*stopCondition=*/nullptr, ValueBoundsOptions{/*closedUB=*/true});
      if (succeeded(upperBound)) {
        bound = *upperBound;
        break;
      }
    }
  }
  return bounds;
}

/// Why hexagon-mlir's footprint model cannot bound what HexagonVTCMTilingPass
/// stages for `op`, or std::nullopt if it can.
std::optional<StringRef>
getUnboundedFootprintReason(linalg::LinalgOp op,
                            ArrayRef<int64_t> loopUpperBounds) {
  // The footprint is computed from the operands' element sizes.
  if (!llvm::all_of(op->getOperandTypes(), [](Type type) {
        return ::mlir::hexagon::getElementSizeInBytes(type).has_value();
      }))
    return StringRef("unsupported operand element type");
  // A dynamic parallel dimension is tiled, which bounds what is staged of it,
  // but a reduction is staged whole at its runtime extent, which only its
  // upper bound bounds.
  for (auto [bound, iterator] :
       llvm::zip_equal(loopUpperBounds, op.getIteratorTypesArray())) {
    if (linalg::isReductionIterator(iterator) && ShapedType::isDynamic(bound))
      return StringRef("dynamic reduction dimension has no upper bound");
  }
  return std::nullopt;
}

/// Whether `tiles` leaves every reduction dimension of `op` whole, i.e. at
/// least its upper bound.
bool keepsReductionsWhole(linalg::LinalgOp op, ArrayRef<int64_t> tiles,
                          ArrayRef<int64_t> loopUpperBounds) {
  for (auto [tile, bound, iterator] :
       llvm::zip_equal(tiles, loopUpperBounds, op.getIteratorTypesArray())) {
    if (linalg::isReductionIterator(iterator) && tile < bound)
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
determineStagedTileSizes(const PlanningContext &context, linalg::LinalgOp op,
                         ArrayRef<int64_t> loopUpperBounds) {
  const int64_t budgetBytes = context.options.vtcmTileBudgetBytes;
  std::optional<SmallVector<int64_t>> tiles =
      ::mlir::hexagon::determineTileSizes(op, budgetBytes, /*tilingDims=*/{},
                                          loopUpperBounds);
  if (tiles && keepsReductionsWhole(op, *tiles, loopUpperBounds))
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
      op, budgetBytes, llvm::to_vector_of<int64_t>(parallelDims),
      loopUpperBounds);
}

/// Tile candidates for a dimension of static `extent` on an HMX unit with
/// `hmxTile`-element tiles: every divisor of `extent` that is a multiple of
/// `hmxTile`, or the whole extent (padded by HMX) if there is none.
SmallVector<int64_t> getHmxTileCandidates(int64_t extent, int64_t hmxTile) {
  SmallVector<int64_t> candidates;
  for (int64_t tile = hmxTile; tile <= extent; tile += hmxTile) {
    if (extent % tile == 0)
      candidates.push_back(tile);
  }
  if (candidates.empty())
    candidates.push_back(extent);
  return candidates;
}

/// Returns VTCM tile sizes for an HMX matmul root, or std::nullopt if `op` is
/// not a contraction with one static M, N and K dimension each and no batch
/// dimension. Batch matmuls keep hexagon-mlir's tiles: with a unit batch tile,
/// HMX lowering vectorizes the accumulator fill at the whole M x N tile.
///
/// M and N tiles are HMX-aligned divisors of their extents and K is staged
/// whole. The tile with the least DDR <-> VTCM traffic wins:
/// an operand whose dimensions are all untiled is the same in every iteration,
/// so it is copied once and not double-buffered (keeping M whole does this for
/// the LHS). Footprints follow hexagon-mlir's model, per operand the product of
/// the tile extents its indexing map reads times its element size.
///
/// Tiles within the budget win over tiles that overrun it. A deep K can leave
/// no HMX-aligned tile within the budget, so overrunning it is allowed up to
/// a fixed factor: the budget is not the VTCM size, and the runtime only runs
/// a dispatch that declares more with fewer others at the same time.
std::optional<SmallVector<int64_t>>
selectHmxTileSizes(const PlanningContext &context, linalg::LinalgOp op) {
  // collect m and n dims
  FailureOr<linalg::ContractionDimensions> dimensions =
      linalg::inferContractionDims(op);
  if (failed(dimensions) || dimensions->m.size() != 1 ||
      dimensions->n.size() != 1 || dimensions->k.empty() ||
      !dimensions->batch.empty())
    return std::nullopt;
  SmallVector<int64_t> ranges = op.getStaticLoopRanges();
  if (llvm::any_of(ranges, ShapedType::isDynamic))
    return std::nullopt;

  unsigned m = dimensions->m.front();
  unsigned n = dimensions->n.front();
  SmallVector<int64_t> tiles(ranges);

  // The planning budget is halved for double buffering; this search counts
  // the second buffers per operand instead.
  const bool doubleBuffering = context.options.doubleBuffering;
  const int64_t budgetBytes =
      context.options.vtcmTileBudgetBytes * (doubleBuffering ? 2 : 1);
  // if a double buffered tile does not fit HMX: allow overrun:
  // "steal" vtcm from other dispatches, since the HMX part is on the critical
  // path rt will manage vtcm and dont start other dispatches, if they dont fit
  constexpr int64_t kMaxBudgetOverrun = 2;

  std::optional<SmallVector<int64_t>> best;
  // to compare against other tiles in order:
  // in_budget, memory_traffic, num_tiles, N
  std::tuple<bool, int64_t, int64_t, int64_t> bestRank;
  for (int64_t mTile : getHmxTileCandidates(ranges[m], context.target.hmxM)) {
    for (int64_t nTile : getHmxTileCandidates(ranges[n], context.target.hmxN)) {
      tiles[m] = mTile;
      tiles[n] = nTile;
      int64_t numTiles = 1;
      // collect metrict to compare against other tiles
      // num tiles
      for (auto [range, tile] : llvm::zip_equal(ranges, tiles))
        numTiles *= range / tile;
      int64_t vtcmBytes = 0, trafficBytes = 0;
      // memory footprint and traffic
      for (auto [map, type] :
           llvm::zip_equal(op.getIndexingMapsArray(), op->getOperandTypes())) {
        int64_t bytes = *::mlir::hexagon::getElementSizeInBytes(type);
        for (AffineExpr expression : map.getResults()) {
          if (auto dim = dyn_cast<AffineDimExpr>(expression))
            bytes *= tiles[dim.getPosition()];
        }
        // The operand's tile is the same in every iteration if every
        // dimension it reads is untiled.
        bool invariant =
            llvm::all_of(map.getResults(), [&](AffineExpr expression) {
              auto dim = dyn_cast<AffineDimExpr>(expression);
              return !dim ||
                     tiles[dim.getPosition()] == ranges[dim.getPosition()];
            });
        // count invariant tiles only once
        vtcmBytes += (doubleBuffering && !invariant) ? 2 * bytes : bytes;
        trafficBytes += invariant ? bytes : bytes * numTiles;
      }
      if (vtcmBytes > kMaxBudgetOverrun * budgetBytes)
        continue; // too large
      // Ties in traffic prefer fewer tiles, then the larger N.
      auto rank = std::make_tuple(vtcmBytes > budgetBytes, trafficBytes,
                                  numTiles, -nTile);
      if (best && rank >= bestRank)
        continue;
      best = tiles;
      bestRank = rank;
    }
  }
  return best;
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
  // Dynamic loops are sized by their upper bounds, both to check that a
  // reduction staged whole fits and to search the tiles of the others.
  SmallVector<int64_t> loopUpperBounds = getLoopUpperBounds(linalgOp);
  if (std::optional<StringRef> reason =
          getUnboundedFootprintReason(linalgOp, loopUpperBounds)) {
    return handleUnavailableVTCM(
        context, dispatchShape, contract,
        Twine("failed to derive required VTCM tile sizes: ") + *reason,
        *reason);
  }
  // HMX matmuls get HMX-aligned tiles from this stage. Otherwise the VTCM
  // layout policy remains owned by the Hexagon lowering utility, searching
  // within the tile budget from the planning options; this planning stage only
  // restricts it to tiles the VTCM tiling pass can realize, and records the
  // result in the dispatch plan.
  std::optional<SmallVector<int64_t>> tileSizes;
  if (contract.hmx != LoopTilingScope::Unused)
    tileSizes = selectHmxTileSizes(context, linalgOp);
  if (!tileSizes)
    tileSizes = determineStagedTileSizes(context, linalgOp, loopUpperBounds);
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

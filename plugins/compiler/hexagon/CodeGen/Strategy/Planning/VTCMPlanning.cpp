// Copyright 2026 RooflineAI GmbH
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "VTCMPlanning.h"

#include "DecisionTrace.h"

#include "hexagon/CodeGen/IR/HmxContracts.h"
#include "hexagon/Conversion/LinalgToLLVM/Common.h"
#include "hexagon/Conversion/LinalgToLLVM/VTCMTilingOptions.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"
#include "mlir/Interfaces/IndexingMapOpInterface.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/MathExtras.h"

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

/// Size granularity of VTCM allocations: hexagon-mlir's runtime rounds a
/// 2 KiB-aligned allocation up to a multiple of 2 KiB and any other one up to a
/// multiple of 128 bytes (VTCMAllocation in bin/runtime/src/HexagonBuffer.cpp).
/// Rounding every staged tile to the larger granularity bounds both.
constexpr int64_t kVTCMAllocationGranularity = 2048;

/// VTCM bytes of one tensor of `elementBytes`-sized elements with the given
/// extents, at VTCM allocation granularity. A non-positive extent is a
/// broadcast and counts as 1, as in hexagon-mlir's footprint model
/// (generateMemoryFootprintAffineMap in VTCMTilingOptions.cpp).
int64_t getVTCMAllocationBytes(int64_t elementBytes,
                               ArrayRef<int64_t> extents) {
  int64_t bytes = elementBytes;
  for (int64_t extent : extents)
    bytes *= std::max<int64_t>(extent, 1);
  return llvm::alignTo(bytes, kVTCMAllocationGranularity);
}

/// Loop extents of `opShape` inside one iteration of the VTCM `scf.forall`:
/// a dimension shared with the root takes the root's VTCM tile, any other one
/// keeps its full static extent. Returns std::nullopt if an extent is
/// unbounded.
std::optional<SmallVector<int64_t>>
getTiledLoopExtents(const OpShape &opShape,
                    const llvm::DenseMap<GlobalDimId, int64_t> &rootTiles) {
  SmallVector<int64_t> extents(opShape.dimensions.size(), 0);
  for (const LocalDim &dimension : opShape.dimensions) {
    auto rootTile = rootTiles.find(dimension.global);
    bool isStatic = !ShapedType::isDynamic(dimension.staticExtent);
    if (rootTile == rootTiles.end() && !isStatic)
      return std::nullopt;
    int64_t extent =
        rootTile == rootTiles.end() ? dimension.staticExtent : rootTile->second;
    if (isStatic)
      extent = std::min(extent, dimension.staticExtent);
    extents[dimension.position] = extent;
  }
  return extents;
}

/// Returns the VTCM bytes HexagonVTCMTilingPass stages for one iteration of the
/// dispatch tiled by the root's VTCM tile `rootTileSizes`, or std::nullopt if
/// it cannot be bounded.
///
/// It counts:
/// each input
/// each linalg.fill, even if thats only a temporary inside the dispatch
/// Each tile uses hexagon-mlir's footprint model
/// This is an upper bound: e.g. bufferization may reuse
std::optional<int64_t>
getStagedFootprintBytes(const DispatchShape &dispatchShape,
                        ArrayRef<int64_t> rootTileSizes) {
  const OpShape &rootShape = getRootShape(dispatchShape);
  llvm::DenseMap<GlobalDimId, int64_t> rootTiles;
  for (const LocalDim &dimension : rootShape.dimensions)
    rootTiles[dimension.global] = rootTileSizes[dimension.position];

  llvm::DenseSet<Operation *> dispatchOps;
  for (const OpShape &opShape : dispatchShape.operations)
    dispatchOps.insert(opShape.op);

  // The same slice of a value read by several operations is staged only once.
  llvm::DenseSet<std::pair<Value, SmallVector<int64_t>>> stagedSlices;
  int64_t total = 0;
  for (const OpShape &opShape : dispatchShape.operations) {
    auto indexingOp = dyn_cast<IndexingMapOpInterface>(opShape.op);
    if (!indexingOp)
      return std::nullopt;
    std::optional<SmallVector<int64_t>> loopExtents =
        getTiledLoopExtents(opShape, rootTiles);
    if (!loopExtents)
      return std::nullopt;
    for (OpOperand &operand : opShape.op->getOpOperands()) {
      Value value = operand.get();
      if (!isa<RankedTensorType>(value.getType()))
        continue;
      auto dpsOp = dyn_cast<DestinationStyleOpInterface>(opShape.op);
      bool isInit = dpsOp && dpsOp.isDpsInit(&operand);
      if (Operation *producer = value.getDefiningOp();
          !isInit && producer && dispatchOps.contains(producer))
        // skip fused intermediaries
        continue;
      std::optional<unsigned> elementBytes =
          ::mlir::hexagon::getElementSizeInBytes(value.getType());
      if (!elementBytes)
        return std::nullopt;
      SmallVector<int64_t> extents =
          indexingOp.getMatchingIndexingMap(&operand).compose(*loopExtents);
      if (!stagedSlices.insert({value, extents}).second)
        // duplicate
        continue;
      total += getVTCMAllocationBytes(*elementBytes, extents);
    }
  }
  return total;
}

/// Returns the VTCM bytes of the HMX copies the HMX pipeline makes of the
/// staged root tile, or std::nullopt if they cannot be bounded.
std::optional<int64_t> getHmxCopyBytes(linalg::LinalgOp root,
                                       ArrayRef<int64_t> rootTileSizes) {
  FailureOr<linalg::ContractionDimensions> dims =
      linalg::inferContractionDims(root);
  if (failed(dims) || dims->m.size() != 1 || dims->n.size() != 1 ||
      dims->k.size() != 1)
    return std::nullopt;
  int64_t mTiles = IREE::Hexagon::getHmxTileCount(rootTileSizes[dims->m[0]]);
  int64_t nTiles = IREE::Hexagon::getHmxTileCount(rootTileSizes[dims->n[0]]);
  int64_t kTiles = IREE::Hexagon::getHmxTileCount(rootTileSizes[dims->k[0]]);
  int64_t gridTiles = mTiles * kTiles + kTiles * nTiles + mTiles * nTiles;
  return gridTiles * IREE::Hexagon::kHmxF16TileBytes +
         IREE::Hexagon::kHmxScratchVTCMBytes;
}

/// Minimum size of a VTCM pool allocation; the plugin's `malloc` rounds smaller
/// requests up to it (plugins/runtime/hexagon/dsp/rt/vtcm_pool.cpp).
constexpr int64_t kVTCMPoolMinimumBytes = 128;

/// Returns the compute tile planned for `opShape`, or an empty list if it has
/// none.
ArrayRef<TileDecision> getComputeTile(const DispatchShape &dispatchShape,
                                      const DispatchPlan &plan,
                                      const OpShape &opShape) {
  if (opShape.op == dispatchShape.root)
    return plan.strategy.rootTiling.computeTile;
  for (const OpComputeTilePlan &opPlan : plan.nonRootComputeTilePlans) {
    if (opPlan.op == opShape.op)
      return opPlan.computeTile;
  }
  return {};
}

/// Returns the bytes of the heap temporaries bufferization allocates for the
/// planned compute tiles, each rounded to the VTCM pool's granularity, or
/// std::nullopt if they cannot be bounded.
///
/// FIXME:
/// it is only used for plans that dont include VTCM, to estimate the amount of
/// heap that ends up in VTCM
std::optional<int64_t> getHeapTemporaryBytes(const DispatchShape &dispatchShape,
                                             const DispatchPlan &plan) {
  llvm::DenseSet<Operation *> dispatchOps;
  for (const OpShape &opShape : dispatchShape.operations)
    dispatchOps.insert(opShape.op);

  // The root tile each global dimension is fused at, if any.
  const RootTilingPlan &rootTiling = plan.strategy.rootTiling;
  llvm::DenseMap<GlobalDimId, int64_t> rootTiles;
  for (const LocalDim &dimension : getRootShape(dispatchShape).dimensions) {
    for (ArrayRef<TileDecision> level :
         {ArrayRef<TileDecision>(rootTiling.computeTile),
          ArrayRef<TileDecision>(rootTiling.distributionTile)}) {
      if (dimension.position < level.size() &&
          level[dimension.position].size != 0) {
        rootTiles[dimension.global] = level[dimension.position].size;
        break;
      }
    }
  }

  int64_t total = 0;
  for (const OpShape &opShape : dispatchShape.operations) {
    auto dpsOp = dyn_cast<DestinationStyleOpInterface>(opShape.op);
    auto indexingOp = dyn_cast<IndexingMapOpInterface>(opShape.op);
    if (!dpsOp || !indexingOp)
      return std::nullopt;
    ArrayRef<TileDecision> computeTile =
        getComputeTile(dispatchShape, plan, opShape);
    for (OpResult result : opShape.op->getResults()) {
      bool needsTemporary = !result.use_empty();
      for (Operation *user : result.getUsers()) {
        if (!dispatchOps.contains(user)) {
          // used outside the dispatch, i.e. the store of the result
          // so that any other user should read from the output buffer
          needsTemporary = false;
          break;
        }
      }
      if (!needsTemporary)
        continue;

      SmallVector<int64_t> loopExtents(opShape.dimensions.size(), 0);
      for (const LocalDim &dimension : opShape.dimensions) {
        int64_t tile = dimension.position < computeTile.size()
                           ? computeTile[dimension.position].size
                           : 0;
        if (tile == 0)
          tile = rootTiles.lookup(dimension.global);
        if (tile == 0 && ShapedType::isDynamic(dimension.staticExtent))
          return std::nullopt;
        loopExtents[dimension.position] =
            tile != 0 ? tile : dimension.staticExtent;
      }
      OpOperand *init = dpsOp.getTiedOpOperand(result);
      std::optional<unsigned> elementBytes =
          ::mlir::hexagon::getElementSizeInBytes(init->get().getType());
      if (!elementBytes)
        return std::nullopt;
      int64_t bytes = *elementBytes;
      for (int64_t extent :
           indexingOp.getMatchingIndexingMap(init).compose(loopExtents))
        bytes *= std::max<int64_t>(extent, 1);
      total += llvm::alignTo(bytes, kVTCMPoolMinimumBytes);
    }
  }
  return total;
}

} // namespace

void planHeapTemporaryVTCM(const PlanningContext &context,
                           const DispatchShape &dispatchShape,
                           DispatchPlan &plan) {
  // A VTCM plan already counts every staged init, which covers these
  // temporaries.
  if (plan.strategy.rootTiling.vtcm)
    return;
  /// FIXME: This predicts what bufferization allocates instead of recording a
  // decision. It exists because `hexagon_runtime_malloc` serves heap
  // temporaries from VTCM (plugins/runtime/hexagon/dsp/rt/malloc_free.c), so a
  // dispatch without a VTCM plan still uses VTCM. Drop it if `malloc` moves to
  // DDR, or derive the temporaries from the plan once the pipeline allocates
  // them explicitly.
  plan.heapTemporaryVTCMBytes = getHeapTemporaryBytes(dispatchShape, plan);
  const OpShape &rootShape = getRootShape(dispatchShape);
  if (plan.heapTemporaryVTCMBytes) {
    context.trace.recordForOp(
        DecisionStage::Resource, DecisionKind::Derived, rootShape.ordinal,
        dispatchShape.root->getName().getStringRef(),
        Twine("heap temporaries take ") + Twine(*plan.heapTemporaryVTCMBytes) +
            " bytes of VTCM");
  } else {
    context.trace.recordForOp(
        DecisionStage::Resource, DecisionKind::Fallback, rootShape.ordinal,
        dispatchShape.root->getName().getStringRef(),
        "heap temporaries cannot be bounded; the dispatch declares all "
        "of VTCM");
  }
}

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

  // The VTCM the dispatch allocates for this plan. If any part of it cannot
  // be bounded, it stays unknown and the dispatch declares all of VTCM.
  std::optional<int64_t> stagedBytes =
      getStagedFootprintBytes(dispatchShape, *tileSizes);
  std::optional<int64_t> pipelineBytes = 0;
  if (contract.usesHMX())
    pipelineBytes = getHmxCopyBytes(linalgOp, *tileSizes);

  VTCMPlan vtcm;
  for (int64_t tile : *tileSizes)
    vtcm.tileSizes.push_back(TileDecision{tile});
  if (stagedBytes && pipelineBytes)
    vtcm.footprintBytes = *stagedBytes + *pipelineBytes;
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
  if (!stagedBytes || !pipelineBytes) {
    context.trace.recordForOp(
        DecisionStage::Resource, DecisionKind::Fallback, rootShape->ordinal,
        dispatchShape.root->getName().getStringRef(),
        "VTCM footprint cannot be bounded; the dispatch declares all of "
        "VTCM");
  } else {
    context.trace.recordForOp(
        DecisionStage::Resource, DecisionKind::Derived, rootShape->ordinal,
        dispatchShape.root->getName().getStringRef(),
        Twine("VTCM footprint ") +
            Twine(*strategy.rootTiling.vtcm->footprintBytes) +
            " bytes (staged tiles " + Twine(*stagedBytes) +
            ", pipeline copies and scratch " + Twine(*pipelineBytes) + ")");
  }
  return success();
}

} // namespace mlir::iree_compiler::hexagon::codegen::planning

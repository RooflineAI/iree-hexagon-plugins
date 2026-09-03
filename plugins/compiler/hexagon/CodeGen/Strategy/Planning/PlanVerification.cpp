// Copyright 2026 RooflineAI GmbH
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "PlanVerification.h"

#include "DecisionTrace.h"
#include "RootTilePropagation.h"

#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/Interfaces/TilingInterface.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/ErrorHandling.h"

namespace mlir::iree_compiler::hexagon::codegen::planning {
namespace {

LogicalResult verifyTile(Operation *op, const TileDecision &tile) {
  if (tile.size < 0) {
    op->emitError("Hexagon tile size must be non-negative");
    return failure();
  }
  return success();
}

LogicalResult verifyTiles(Operation *op, ArrayRef<TileDecision> tiles) {
  for (const TileDecision &tile : tiles) {
    if (failed(verifyTile(op, tile)))
      return failure();
  }
  return success();
}

LogicalResult verifyDispatchShape(const PlanningContext &context,
                                  const DispatchShape &dispatchShape) {
  FunctionOpInterface entryPoint = context.entryPoint;
  FunctionOpInterface shapeEntryPoint = dispatchShape.entryPoint;
  if (!shapeEntryPoint ||
      shapeEntryPoint.getOperation() != entryPoint.getOperation()) {
    entryPoint.emitError(
        "Hexagon dispatch shape belongs to a different entry point");
    return failure();
  }

  for (auto [index, global] : llvm::enumerate(dispatchShape.globalDimensions)) {
    if (global.id != index) {
      entryPoint.emitError() << "Hexagon global dimension " << global.id
                             << " does not match its position " << index;
      return failure();
    }
  }

  llvm::DenseSet<Operation *> operations;
  SmallVector<bool> referencedGlobals(dispatchShape.globalDimensions.size(),
                                      false);
  for (auto [ordinal, opShape] : llvm::enumerate(dispatchShape.operations)) {
    if (!opShape.op) {
      entryPoint.emitError("Hexagon dispatch shape contains a null operation");
      return failure();
    }
    if (!operations.insert(opShape.op).second) {
      entryPoint.emitError(
          "Hexagon dispatch shape contains a duplicate operation");
      return failure();
    }
    if (opShape.ordinal != ordinal) {
      opShape.op->emitError() << "Hexagon operation ordinal " << opShape.ordinal
                              << " does not match its position " << ordinal;
      return failure();
    }
    FunctionOpInterface parent =
        opShape.op->getParentOfType<FunctionOpInterface>();
    if (!parent || parent.getOperation() != entryPoint.getOperation()) {
      opShape.op->emitError(
          "Hexagon dispatch operation belongs to a different entry point");
      return failure();
    }

    auto tilingOp = dyn_cast<TilingInterface>(opShape.op);
    if (!tilingOp) {
      opShape.op->emitError(
          "Hexagon dispatch shape contains a non-tiling operation");
      return failure();
    }
    SmallVector<utils::IteratorType> iteratorTypes =
        tilingOp.getLoopIteratorTypes();
    if (iteratorTypes.size() != opShape.dimensions.size()) {
      opShape.op->emitError(
          "Hexagon operation shape rank does not match its loop rank");
      return failure();
    }

    for (auto [position, dimension] : llvm::enumerate(opShape.dimensions)) {
      if (dimension.position != position) {
        opShape.op->emitError()
            << "Hexagon local dimension position " << dimension.position
            << " does not match its position " << position;
        return failure();
      }
      if (dimension.global >= dispatchShape.globalDimensions.size()) {
        opShape.op->emitError()
            << "Hexagon local dimension references missing global dimension "
            << dimension.global;
        return failure();
      }
      referencedGlobals[dimension.global] = true;
      if (dimension.iteratorType != iteratorTypes[position]) {
        opShape.op->emitError()
            << "Hexagon local dimension " << position
            << " has an iterator type inconsistent with its operation";
        return failure();
      }
      if (dimension.staticExtent < ShapedType::kDynamic) {
        opShape.op->emitError()
            << "Hexagon local dimension has invalid static extent "
            << dimension.staticExtent;
        return failure();
      }
    }
  }

  const OpShape *rootShape = nullptr;
  if (dispatchShape.root) {
    rootShape = findOpShape(dispatchShape, dispatchShape.root);
    if (!rootShape) {
      dispatchShape.root->emitError(
          "Hexagon dispatch root is absent from analyzed operations");
      return failure();
    }
  }

  SmallVector<bool> expectedPresentInRoot(dispatchShape.globalDimensions.size(),
                                          false);
  if (rootShape) {
    for (const LocalDim &dimension : rootShape->dimensions)
      expectedPresentInRoot[dimension.global] = true;
  }
  for (auto [index, global] : llvm::enumerate(dispatchShape.globalDimensions)) {
    if (!referencedGlobals[index]) {
      entryPoint.emitError() << "Hexagon global dimension " << global.id
                             << " is not referenced by any operation";
      return failure();
    }
    if (global.presentInRoot != expectedPresentInRoot[index]) {
      entryPoint.emitError() << "Hexagon global dimension " << global.id
                             << " has inconsistent root-presence information";
      return failure();
    }
  }
  return success();
}

bool carriesLevel(const DispatchShape &dispatchShape, const OpShape &opShape,
                  ArrayRef<TileDecision> computeTile, ComputeTileLevel level) {
  return llvm::any_of(opShape.dimensions, [&](const LocalDim &dimension) {
    return computeTile[dimension.position].size != 0 &&
           classifyComputeTileLevel(dispatchShape, dimension) == level;
  });
}

struct BoundaryAnchors {
  const OpShape *beforeRoot = nullptr;
  const OpShape *afterRoot = nullptr;
};

BoundaryAnchors getBoundaryAnchors(ArrayRef<const OpShape *> carryingOperations,
                                   const OpShape &rootShape,
                                   bool hasRootAnchor) {
  BoundaryAnchors anchors;
  if (!hasRootAnchor) {
    if (!carryingOperations.empty())
      anchors.afterRoot = carryingOperations.back();
    return anchors;
  }
  for (const OpShape *opShape : carryingOperations) {
    if (opShape->ordinal < rootShape.ordinal)
      anchors.beforeRoot = opShape;
    else if (opShape->ordinal > rootShape.ordinal)
      anchors.afterRoot = opShape;
  }
  return anchors;
}

bool isLoopTileConsumed(LoopTilingScope scope, const OpShape &opShape,
                        const OpShape &rootShape,
                        const BoundaryAnchors &boundaryAnchors) {
  switch (scope) {
  case LoopTilingScope::Unused:
    return false;
  case LoopTilingScope::Root:
    return opShape.op == rootShape.op;
  case LoopTilingScope::EveryConfiguredOperation:
    return true;
  case LoopTilingScope::LastConfiguredOnEachSideOfRoot:
    return &opShape == boundaryAnchors.beforeRoot ||
           &opShape == boundaryAnchors.afterRoot;
  }
  llvm_unreachable("unknown loop tiling scope");
}

LogicalResult verifyComputeTileConsumption(
    const DispatchShape &dispatchShape, const OpShape &rootShape,
    const PipelineContract &contract,
    const llvm::DenseMap<Operation *, ArrayRef<TileDecision>> &computeTiles,
    bool hasRootAnchor) {
  // Both consumers merge Common, Reduction, and Inner into one per-operation
  // vector shape. A level can therefore be useful as a vector shape even when
  // its loop-tiling scope does not include that operation.
  bool consumesConfiguredVectorShape =
      contract.runsTileToVectorSize || contract.usesConfiguredVectorSizes;

  for (ComputeTileLevel level :
       {ComputeTileLevel::CommonParallel, ComputeTileLevel::Reduction,
        ComputeTileLevel::InnerParallel}) {
    SmallVector<const OpShape *> carryingOperations;
    for (const OpShape &opShape : dispatchShape.operations) {
      auto tile = computeTiles.find(opShape.op);
      if (tile != computeTiles.end() &&
          carriesLevel(dispatchShape, opShape, tile->second, level))
        carryingOperations.push_back(&opShape);
    }

    LoopTilingScope scope = contract.getLoopTilingScope(level);
    BoundaryAnchors boundaryAnchors =
        getBoundaryAnchors(carryingOperations, rootShape, hasRootAnchor);

    for (const OpShape *opShape : carryingOperations) {
      if (consumesConfiguredVectorShape ||
          isLoopTileConsumed(scope, *opShape, rootShape, boundaryAnchors))
        continue;
      opShape->op->emitError()
          << "selected Hexagon pipeline does not consume "
          << stringifyComputeTileLevel(level) << " tiling for this operation";
      return failure();
    }
  }
  return success();
}

/// A non-root operation fused into a root-anchored loop nest computes the
/// root's tile, so its configured compute tile has to be realizable inside it.
/// See RootTilePropagation.cpp for why each of these three states is not
/// realizable; this only rejects them.
LogicalResult verifyNonRootTileFitsRootFusionTile(
    const DispatchShape &dispatchShape, const PipelineContract &contract,
    const RootFusionBounds &bounds, const OpShape &opShape,
    ArrayRef<TileDecision> computeTile) {
  if (contract.vectorCommonParallel != LoopTilingScope::Root)
    return success();

  bool refinable = canRefineComputeTileDownstream(contract, opShape.op);
  for (const LocalDim &dimension : opShape.dimensions) {
    if (classifyComputeTileLevel(dispatchShape, dimension) !=
        ComputeTileLevel::CommonParallel)
      continue;
    int64_t rootTile = bounds.getBound(opShape.op, dimension.global);
    int64_t size = computeTile[dimension.position].size;
    // A zero sends the operation to shape inference from the tiled IR, which
    // matches the fused slice by construction.
    if (rootTile == 0 || size == 0)
      continue;

    if (size > rootTile) {
      opShape.op->emitError()
          << "Hexagon compute tile " << size << " on dimension "
          << dimension.position << " exceeds the root fusion tile " << rootTile
          << "; the configured vector shape would be masked beyond the fused "
             "slice";
      return failure();
    }
    if (rootTile % size != 0) {
      opShape.op->emitError()
          << "Hexagon compute tile " << size << " on dimension "
          << dimension.position << " does not divide the root fusion tile "
          << rootTile << "; the inner loop would leave a remainder";
      return failure();
    }
    if (!refinable && size != rootTile) {
      opShape.op->emitError()
          << "Hexagon compute tile " << size << " on dimension "
          << dimension.position << " must equal the root fusion tile "
          << rootTile
          << " because the selected pipeline cannot refine this operation's "
             "vector shape";
      return failure();
    }
  }
  return success();
}

LogicalResult
verifyPipelineRequirements(FunctionOpInterface entryPoint,
                           const DispatchStrategy &strategy,
                           const PipelineContract &pipelineContract) {
  if (strategy.requestLoopPeeling &&
      pipelineContract.loopPeeling !=
          LoopPeelingSupport::TranslationInfoControlled) {
    entryPoint.emitError(
        "selected Hexagon pipeline does not accept requested loop peeling");
    return failure();
  }
  if (pipelineContract.vtcmRequirement == VTCMRequirement::Required &&
      !strategy.rootTiling.vtcm) {
    entryPoint.emitError("selected Hexagon pipeline requires VTCM");
    return failure();
  }
  if (pipelineContract.vtcmRequirement == VTCMRequirement::Unsupported &&
      strategy.rootTiling.vtcm) {
    entryPoint.emitError(
        "selected Hexagon pipeline does not consume VTCM tiling");
    return failure();
  }
  if (strategy.rootTiling.vtcm &&
      pipelineContract.cacheTilingWithVTCM == CacheTilingWithVTCM::Suppress &&
      llvm::any_of(strategy.rootTiling.cacheTile,
                   [](const TileDecision &tile) { return tile.size != 0; })) {
    entryPoint.emitError(
        "selected VTCM plan must suppress cache tiling for this pipeline");
    return failure();
  }
  return success();
}

LogicalResult verifyRootlessPlan(FunctionOpInterface entryPoint,
                                 const DispatchPlan &plan) {
  const RootTilingPlan &root = plan.strategy.rootTiling;
  if (plan.strategy.pipeline != IREE::CPU::LoweringPipeline::Default ||
      root.vtcm || !root.distributionTile.empty() || !root.cacheTile.empty() ||
      !root.computeTile.empty() || !plan.nonRootComputeTilePlans.empty()) {
    entryPoint.emitError(
        "rootless Hexagon dispatch must use an empty default plan");
    return failure();
  }
  return success();
}

LogicalResult verifyRootTilingPlan(const OpShape &rootShape,
                                   const DispatchStrategy &strategy,
                                   const PipelineContract &pipelineContract) {
  Operation *rootOp = rootShape.op;
  const RootTilingPlan &root = strategy.rootTiling;
  size_t rootRank = rootShape.dimensions.size();
  if (root.distributionTile.size() != rootRank ||
      root.cacheTile.size() != rootRank ||
      root.computeTile.size() != rootRank ||
      (root.vtcm && root.vtcm->tileSizes.size() != rootRank)) {
    rootOp->emitError("Hexagon root plan rank does not match root loop rank");
    return failure();
  }
  if (failed(verifyTiles(rootOp, root.distributionTile)) ||
      failed(verifyTiles(rootOp, root.cacheTile)) ||
      failed(verifyTiles(rootOp, root.computeTile)) ||
      (root.vtcm && failed(verifyTiles(rootOp, root.vtcm->tileSizes))))
    return failure();

  if (llvm::any_of(root.cacheTile,
                   [](const TileDecision &tile) { return tile.size > 0; }) &&
      pipelineContract.cacheParallel == LoopTilingScope::Unused) {
    rootOp->emitError(
        "root cache tile is not consumed by the selected pipeline");
    return failure();
  }
  return success();
}

LogicalResult collectAndVerifyComputeTiles(
    FunctionOpInterface entryPoint, const DispatchShape &dispatchShape,
    const DispatchPlan &plan, const PipelineContract &pipelineContract,
    const OpShape &rootShape,
    llvm::DenseMap<Operation *, ArrayRef<TileDecision>> &computeTiles) {
  const RootTilingPlan &root = plan.strategy.rootTiling;
  RootFusionBounds fusionBounds(dispatchShape, root);
  llvm::DenseSet<Operation *> configured;
  computeTiles[rootShape.op] = root.computeTile;

  for (const OpComputeTilePlan &opPlan : plan.nonRootComputeTilePlans) {
    const OpShape *opShape = findOpShape(dispatchShape, opPlan.op);
    if (!opShape || opPlan.op == dispatchShape.root ||
        !configured.insert(opPlan.op).second) {
      entryPoint.emitError(
          "invalid or duplicate Hexagon non-root compute-tile plan");
      return failure();
    }
    if (!pipelineContract.supportsIndependentNonRootComputeTiles) {
      opPlan.op->emitError(
          "pipeline cannot realize independent non-root compute tiles");
      return failure();
    }
    if (opPlan.computeTile.size() != opShape->dimensions.size()) {
      opPlan.op->emitError(
          "Hexagon compute-tile rank does not match operation loop rank");
      return failure();
    }
    if (failed(verifyTiles(opPlan.op, opPlan.computeTile)))
      return failure();
    if (failed(verifyNonRootTileFitsRootFusionTile(
            dispatchShape, pipelineContract, fusionBounds, *opShape,
            opPlan.computeTile)))
      return failure();
    computeTiles[opPlan.op] = opPlan.computeTile;
  }
  return success();
}

} // namespace

LogicalResult verifyDispatchPlan(const PlanningContext &context,
                                 const DispatchShape &dispatchShape,
                                 const DispatchPlan &plan,
                                 const PipelineContract &pipelineContract) {
  FunctionOpInterface entryPoint = context.entryPoint;
  if (failed(verifyDispatchShape(context, dispatchShape)) ||
      failed(verifyPipelineRequirements(entryPoint, plan.strategy,
                                        pipelineContract)))
    return failure();

  const OpShape *rootShape = findOpShape(dispatchShape, dispatchShape.root);
  if (!rootShape) {
    if (failed(verifyRootlessPlan(entryPoint, plan)))
      return failure();
    context.trace.record(DecisionStage::Verification, DecisionKind::Derived,
                         "verified complete dispatch plan");
    return success();
  }

  if (failed(verifyRootTilingPlan(*rootShape, plan.strategy, pipelineContract)))
    return failure();

  llvm::DenseMap<Operation *, ArrayRef<TileDecision>> computeTiles;
  if (failed(collectAndVerifyComputeTiles(entryPoint, dispatchShape, plan,
                                          pipelineContract, *rootShape,
                                          computeTiles)))
    return failure();

  bool hasRootAnchor =
      pipelineContract.requiresUniqueRootAnchor ||
      llvm::any_of(plan.strategy.rootTiling.distributionTile,
                   [](const TileDecision &tile) { return tile.size != 0; });
  if (failed(verifyComputeTileConsumption(dispatchShape, *rootShape,
                                          pipelineContract, computeTiles,
                                          hasRootAnchor)))
    return failure();

  context.trace.record(DecisionStage::Verification, DecisionKind::Derived,
                       "verified complete dispatch plan");
  return success();
}

} // namespace mlir::iree_compiler::hexagon::codegen::planning

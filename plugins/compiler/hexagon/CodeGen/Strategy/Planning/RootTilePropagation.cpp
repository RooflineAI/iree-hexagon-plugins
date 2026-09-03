// Copyright 2026 RooflineAI GmbH
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "RootTilePropagation.h"

#include "DecisionTrace.h"
#include "Strategies/StrategySupport.h"

#include "mlir/Interfaces/DestinationStyleOpInterface.h"
#include "llvm/ADT/STLExtras.h"

namespace mlir::iree_compiler::hexagon::codegen::planning {

//===----------------------------------------------------------------------===//
// Why this stage exists
//===----------------------------------------------------------------------===//
//
// Non-root compute tiles are selected from each operation's own shape, with no
// knowledge of the root. That independence is what keeps a fused consumer from
// inheriting a coarse root tile it cannot afford, but it is not free: when the
// selected pipeline anchors Common tiling on the root, a non-root operation is
// fused into a loop nest whose trip counts come from the root's tile, so the
// slice it actually computes is the root's tile. Its configured vector shape
// has to be realizable inside that slice.
//
// Two lowering pipelines and their passes decide what "realizable" means.
//
//  1. GenericVectorizationPass takes a configured vector shape verbatim
//     whenever the merged levels contain no zero. Masked vectorization can pad
//     a short slice but never truncate a long one, so a tile *larger* than the
//     fused extent yields a masked vector whose upper lanes are dead - a
//     64-wide f16 vector over a 32-element slice wastes half an HVX register
//     and adds a mask - while a tile *smaller* than the fused extent is
//     rejected outright unless some pass first tiles the operation down to it.
//
//  2. LLVMCPUTileToVectorSizePass is the pass that would perform that inner
//     tiling, and it declines two cases: it leaves a dimension alone when the
//     requested size exceeds the slice, and it skips linalg.fill entirely. A
//     fused fill therefore has no route to a smaller configured shape, and an
//     unmatched fill config silently loses vectorization: the fill reaches
//     bufferization as a linalg op and the root's accumulator is reloaded
//     instead of folded into a constant.
//
// So this stage bounds every non-root Common tile by the root's fusion tile:
//
//  - an operation whose shape TileToVectorSize can still refine keeps the
//    largest divisor of the root tile that does not exceed its own preference,
//    so the inner loop that pass creates has no remainder;
//  - an operation it cannot refine takes the root tile exactly.
//
// Which of the root's tiles bounds a dimension is decided by RootFusionBounds,
// because sharing a dimension with the root is not the same as being inside the
// loop that tiles it. A root parallel dimension bounds every fused operation. A
// root reduction dimension bounds only the producers reachable from the root's
// input operands, since TileRootAndFuseInputOperands is what populates that
// loop and it skips destination operands and consumers.
//
// A dimension absent from the root is never bounded here.
//
//===----------------------------------------------------------------------===//
// Relationship to LLVMCPU, and why this is WIP
//===----------------------------------------------------------------------===//
//
// This is not LLVMCPU's propagation. The original implementation lets a
// non-root preference *win* over the root's (see `adjustTileSizesForRootOp`).
//
// Here influence only ever flows away from the root, and every conflict is
// resolved by shrinking the non-root operation. This is not optimal and is
// meant to be addressed in the future.
//
// Finding the ideal shapes may require a register-pressure model that can pick
// the largest *feasible* tile, and upward negotiation so that a merely
// heuristic root tile can move instead of forcing consumers down.

namespace {

/// Operations reachable from the root's input operands. These are what
/// TileRootAndFuseInputOperands pulls into the root's reduction loop; the
/// producer of the root's destination operand is deliberately excluded, because
/// that level skips destination operands.
llvm::SmallDenseSet<Operation *> collectRootInputProducers(Operation *root) {
  llvm::SmallDenseSet<Operation *> producers;
  SmallVector<Value> worklist;
  if (auto destinationRoot = dyn_cast<DestinationStyleOpInterface>(root)) {
    for (OpOperand *input : destinationRoot.getDpsInputOperands())
      worklist.push_back(input->get());
  } else {
    llvm::append_range(worklist, root->getOperands());
  }
  while (!worklist.empty()) {
    Operation *definingOp = worklist.pop_back_val().getDefiningOp();
    if (!definingOp || !producers.insert(definingOp).second)
      continue;
    llvm::append_range(worklist, definingOp->getOperands());
  }
  return producers;
}

} // namespace

RootFusionBounds::RootFusionBounds(const DispatchShape &dispatchShape,
                                   const RootTilingPlan &rootTiling) {
  const OpShape *rootShape = findOpShape(dispatchShape, dispatchShape.root);
  if (!rootShape)
    return;
  for (const LocalDim &dimension : rootShape->dimensions) {
    if (dimension.position >= rootTiling.computeTile.size())
      continue;
    int64_t size = rootTiling.computeTile[dimension.position].size;
    if (size == 0)
      continue;
    if (classifyComputeTileLevel(dispatchShape, dimension) ==
        ComputeTileLevel::Reduction)
      reductionTiles.try_emplace(dimension.global, size);
    else
      parallelTiles.try_emplace(dimension.global, size);
  }
  inputProducers = collectRootInputProducers(dispatchShape.root);
}

int64_t RootFusionBounds::getBound(Operation *op, GlobalDimId global) const {
  if (auto parallel = parallelTiles.find(global);
      parallel != parallelTiles.end())
    return parallel->second;
  if (!inputProducers.contains(op))
    return 0;
  auto reduction = reductionTiles.find(global);
  return reduction == reductionTiles.end() ? 0 : reduction->second;
}

LogicalResult reconcileNonRootComputeTiles(const PlanningContext &context,
                                           const DispatchShape &dispatchShape,
                                           const PipelineContract &contract,
                                           DispatchPlan &plan) {
  // Any other scope gives each operation its own loop nest, so an
  // independently selected tile is already the extent that operation computes.
  if (contract.vectorCommonParallel != LoopTilingScope::Root)
    return success();
  if (!findOpShape(dispatchShape, dispatchShape.root))
    return success();

  RootFusionBounds bounds(dispatchShape, plan.strategy.rootTiling);
  for (OpComputeTilePlan &opPlan : plan.nonRootComputeTilePlans) {
    const OpShape *opShape = findOpShape(dispatchShape, opPlan.op);
    // A malformed plan is verification's diagnostic to report, not this
    // stage's to paper over.
    if (!opShape || opPlan.computeTile.size() != opShape->dimensions.size())
      continue;

    bool refinable = canRefineComputeTileDownstream(contract, opPlan.op);
    for (const LocalDim &dimension : opShape->dimensions) {
      if (classifyComputeTileLevel(dispatchShape, dimension) !=
          ComputeTileLevel::CommonParallel)
        continue;
      int64_t rootTile = bounds.getBound(opPlan.op, dimension.global);
      if (rootTile == 0)
        continue;

      TileDecision &tile = opPlan.computeTile[dimension.position];
      // A zero makes getVectorSizes() report a zero for the whole operation,
      // which sends it to shape inference from the tiled IR. That inference
      // already yields the fused slice, so a zero needs no bound.
      if (tile.size == 0)
        continue;
      int64_t bounded =
          refinable ? chooseStaticTilingFactor(rootTile, tile.size) : rootTile;
      if (bounded == tile.size)
        continue;
      if (tile.hardwareFixed) {
        opPlan.op->emitError()
            << "hardware-fixed Hexagon compute tile " << tile.size
            << " does not fit the root fusion tile " << rootTile;
        return failure();
      }

      int64_t previousSize = tile.size;
      tile.size = bounded;
      context.trace.recordAdjustment(
          DecisionStage::ComputeTile, opShape->ordinal,
          opPlan.op->getName().getStringRef(), "operation compute tile",
          dimension.position, previousSize, bounded,
          refinable ? "bounded by the root fusion tile"
                    : "matched to the root fusion tile because the pipeline "
                      "cannot refine this operation's vector shape");
    }
  }
  return success();
}

} // namespace mlir::iree_compiler::hexagon::codegen::planning

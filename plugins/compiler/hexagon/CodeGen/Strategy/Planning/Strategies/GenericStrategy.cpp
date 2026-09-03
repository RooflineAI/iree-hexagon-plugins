// Copyright 2026 RooflineAI GmbH
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "Strategy.h"
#include "StrategySupport.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "llvm/ADT/STLExtras.h"

#include <algorithm>

namespace mlir::iree_compiler::hexagon::codegen::planning {
namespace {

// Cap used to bound intermediate tensor tiles as a fallback in case other
// tiling levels fail unexpectedly. It is currently not derived from a cache or
// register-pressure model.
constexpr int64_t kCacheTileCap = 64;

DispatchStrategy makeBaseStrategy(const OpShape &shape) {
  DispatchStrategy strategy;
  strategy.pipeline = IREE::CPU::LoweringPipeline::DoubleTilingExpert;
  strategy.rootTiling.distributionTile =
      SmallVector<TileDecision>(shape.dimensions.size());
  strategy.rootTiling.cacheTile =
      SmallVector<TileDecision>(shape.dimensions.size());
  return strategy;
}

SmallVector<TileDecision>
inferGenericComputeTile(const PlanningContext &context, const OpShape &shape,
                        linalg::GenericOp op) {
  constexpr int64_t kGenericReductionTile = 8;
  SmallVector<TileDecision> tiles(shape.dimensions.size());
  Type resultType = op.getNumDpsInits()
                        ? op.getDpsInitOperand(0)->get().getType()
                        : op->getOperand(0).getType();
  int64_t vectorWidth = getTypeNativeVectorWidth(context, resultType);
  SmallVector<unsigned> parallelDimensions;
  int64_t innermostReduction = -1;
  for (const LocalDim &dimension : shape.dimensions) {
    if (dimension.iteratorType == utils::IteratorType::parallel)
      parallelDimensions.push_back(dimension.position);
    else if (dimension.iteratorType == utils::IteratorType::reduction)
      innermostReduction = dimension.position;
  }
  // For softmax-like loops, the innermost reduction follows the contiguous
  // axis while the remaining parallel dimension would require strided/gather
  // loads. Redirect the native vector width to that reduction.
  bool vectorizeReduction =
      parallelDimensions.size() >= 2 &&
      innermostReduction == static_cast<int64_t>(shape.dimensions.size() - 1);
  if (!parallelDimensions.empty()) {
    for (unsigned dimension : llvm::drop_end(parallelDimensions))
      tiles[dimension] = TileDecision{1};
    unsigned innermost = parallelDimensions.back();
    int64_t bound = shape.dimensions[innermost].staticExtent;
    tiles[innermost] =
        TileDecision{vectorizeReduction ? 1
                                        : (ShapedType::isDynamic(bound)
                                               ? vectorWidth
                                               : std::min(bound, vectorWidth))};
  }
  if (innermostReduction >= 0) {
    int64_t preferred =
        vectorizeReduction ? vectorWidth : kGenericReductionTile;
    tiles[innermostReduction] = TileDecision{chooseStaticTilingFactor(
        shape.dimensions[innermostReduction].staticExtent, preferred)};
  }
  return tiles;
}

} // namespace

FailureOr<std::optional<DispatchStrategy>>
selectGenericStrategy(const PlanningContext &context,
                      const DispatchShape &dispatchShape) {
  auto genericOp = dyn_cast_or_null<linalg::GenericOp>(dispatchShape.root);
  if (!genericOp)
    return std::optional<DispatchStrategy>{};

  const OpShape &shape = getRootShape(dispatchShape);
  DispatchStrategy strategy = makeBaseStrategy(shape);
  strategy.requestLoopPeeling = genericOp.hasPureTensorSemantics();
  strategy.rootTiling.computeTile =
      inferGenericComputeTile(context, shape, genericOp);

  SmallVector<unsigned> parallelDimensions;
  for (const LocalDim &dimension : shape.dimensions) {
    if (dimension.iteratorType == utils::IteratorType::parallel)
      parallelDimensions.push_back(dimension.position);
  }
  if (parallelDimensions.size() > 1) {
    for (unsigned dimension : llvm::drop_end(parallelDimensions)) {
      strategy.rootTiling.cacheTile[dimension] =
          TileDecision{chooseStaticTilingFactor(
              shape.dimensions[dimension].staticExtent, kCacheTileCap)};
    }
  }

  return std::optional<DispatchStrategy>(std::move(strategy));
}

FailureOr<std::optional<OpComputeTilePlan>>
selectGenericComputeTile(const PlanningContext &context, const DispatchShape &,
                         const DispatchStrategy &, const PipelineContract &,
                         const OpShape &opShape) {
  auto genericOp = dyn_cast<linalg::GenericOp>(opShape.op);
  if (!genericOp)
    return std::optional<OpComputeTilePlan>{};

  SmallVector<TileDecision> compute =
      inferGenericComputeTile(context, opShape, genericOp);

  return std::optional<OpComputeTilePlan>(
      OpComputeTilePlan{opShape.op, std::move(compute)});
}

} // namespace mlir::iree_compiler::hexagon::codegen::planning

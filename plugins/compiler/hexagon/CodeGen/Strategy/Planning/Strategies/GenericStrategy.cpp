// Copyright 2026 RooflineAI GmbH
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "Strategy.h"
#include "StrategySupport.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/bit.h"

#include <algorithm>

namespace mlir::iree_compiler::hexagon::codegen::planning {
namespace {

// Cap used to bound intermediate tensor tiles as a fallback in case other
// tiling levels fail unexpectedly. It is currently not derived from a cache or
// register-pressure model.
constexpr int64_t kCacheTileCap = 64;

// HVX has neither a gather nor 64-bit lanes, so LLVM lowers gathers and 64-bit
// operations lane by lane. All lanes of a vector tile are live at once, so
// wide tiles of such bodies spill far beyond the DSP stack frame limit. These
// caps bound the vector tile of the affected generics.
//
// Gathers: at most kMaxGatherLanes lanes, and at most kGatherLaneBudget lanes
// summed over all gathers of the body.
// FIXME: these are empirical numbers and may require fine-tuning
constexpr int64_t kMaxGatherLanes = 32;
constexpr int64_t kGatherLaneBudget = 256;

bool isDefinedOutside(Value value, linalg::GenericOp op) {
  return !op.getRegion().isAncestor(value.getParentRegion());
}

// Returns the loop dimension of `value` if it is a `linalg.index` op.
std::optional<uint64_t> getIndexDim(Value value) {
  if (auto indexOp = value.getDefiningOp<linalg::IndexOp>())
    return indexOp.getDim();
  return std::nullopt;
}

// Whether the vectorizer can lower `extractOp` to a contiguous load along
// `vectorDim`: the trailing index is `linalg.index vectorDim`, optionally plus
// a loop-invariant offset, and the other indices are uniform across the
// vector. Conservative: anything else counts as a gather.
bool isContiguousExtract(tensor::ExtractOp extractOp, linalg::GenericOp op,
                         unsigned vectorDim) {
  auto indices = extractOp.getIndices();
  if (indices.empty())
    return true;
  for (Value index : indices.drop_back()) {
    if (isDefinedOutside(index, op))
      continue;
    std::optional<uint64_t> dim = getIndexDim(index);
    if (!dim || *dim == vectorDim)
      return false;
  }
  Value trailing = indices.back();
  if (auto addOp = trailing.getDefiningOp<arith::AddIOp>()) {
    if (isDefinedOutside(addOp.getRhs(), op))
      trailing = addOp.getLhs();
    else if (isDefinedOutside(addOp.getLhs(), op))
      trailing = addOp.getRhs();
  }
  return getIndexDim(trailing) == vectorDim;
}

// Caps `vectorWidth`, the lanes along `vectorDim`, for the body of `op`.
// Bodies on elements wider than the widest native lane scale their vector tile
// down by the ratio, so a wide value needs no more registers than a
// native-width value of the uncapped tile.
int64_t capBodyVectorWidth(const PlanningContext &context, linalg::GenericOp op,
                           unsigned vectorDim, int64_t vectorWidth) {
  const int64_t nativeLaneBits = context.target.maxNativeLaneBits;
  int64_t numGathers = 0;
  int64_t widestBits = nativeLaneBits;
  op.getBody()->walk([&](Operation *nested) {
    if (auto extractOp = dyn_cast<tensor::ExtractOp>(nested)) {
      if (!isContiguousExtract(extractOp, op, vectorDim))
        ++numGathers;
    }
    for (Type type : nested->getResultTypes()) {
      if (type.isIntOrFloat())
        widestBits =
            std::max<int64_t>(widestBits, type.getIntOrFloatBitWidth());
    }
  });
  // fix planner assumption, that the ops are native, even for larger types:
  int64_t cap = std::max<int64_t>(1, vectorWidth * nativeLaneBits / widestBits);
  // gather ops are not vectorized efficiently: cap them
  if (numGathers > 0) {
    int64_t gatherLanes =
        std::min(kMaxGatherLanes,
                 static_cast<int64_t>(llvm::bit_floor(static_cast<uint64_t>(
                     std::max<int64_t>(1, kGatherLaneBudget / numGathers)))));
    cap = std::min(cap, gatherLanes);
  }
  return cap;
}

DispatchStrategy makeBaseStrategy(const OpShape &shape) {
  DispatchStrategy strategy;
  strategy.pipeline = IREE::Hexagon::LoweringPipeline::MultiTilingExpert;
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
  if (vectorizeReduction || !parallelDimensions.empty()) {
    unsigned vectorDim =
        vectorizeReduction ? innermostReduction : parallelDimensions.back();
    vectorWidth = capBodyVectorWidth(context, op, vectorDim, vectorWidth);
  }
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

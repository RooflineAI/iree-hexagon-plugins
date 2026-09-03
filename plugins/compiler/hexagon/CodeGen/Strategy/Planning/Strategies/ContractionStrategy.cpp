// Copyright 2026 RooflineAI GmbH
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "Strategy.h"
#include "StrategySupport.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/IR/LinalgInterfaces.h"
#include "mlir/IR/TypeUtilities.h"
#include "llvm/ADT/STLExtras.h"

#include <algorithm>

namespace mlir::iree_compiler::hexagon::codegen::planning {
namespace {

// Legacy safety cap used to bound intermediate tensor tiles. It is not derived
// from a cache or register-pressure model.
constexpr int64_t kCacheTileCap = 64;

std::optional<linalg::ContractionDimensions>
analyzeSupportedContraction(linalg::LinalgOp op) {
  // The current policies model matmul-shaped contractions with exactly one
  // M, N, and K dimension and at most one batch dimension.
  auto dimensions = linalg::inferContractionDims(op);
  if (failed(dimensions) || dimensions->m.size() != 1 ||
      dimensions->n.size() != 1 || dimensions->k.size() != 1 ||
      dimensions->batch.size() > 1)
    return std::nullopt;
  return std::move(*dimensions);
}

bool hasHmxElementTypes(linalg::LinalgOp op) {
  // HMX planning is currently restricted to f16 inputs.
  // Hardware also supports int8 though.
  return llvm::all_of(op.getDpsInputs(), [](Value input) {
    return getElementTypeOrSelf(input.getType()).isF16();
  });
}

SmallVector<TileDecision> inferContractionComputeTile(
    const PlanningContext &context, const OpShape &shape, linalg::LinalgOp op,
    const linalg::ContractionDimensions &dimensions, bool useHmx) {
  constexpr int64_t kContractionMTile = 8;
  constexpr int64_t kContractionKTile = 8;
  SmallVector<TileDecision> tiles(shape.dimensions.size());
  unsigned m = dimensions.m.front();
  unsigned n = dimensions.n.front();
  unsigned k = dimensions.k.front();
  Type accumulatorType = op.getDpsInitOperand(0)->get().getType();
  int64_t vectorWidth = getTypeNativeVectorWidth(context, accumulatorType);
  int64_t mTile = kContractionMTile;
  int64_t nTile = vectorWidth;
  int64_t kTile = std::min<int64_t>(kContractionKTile, nTile);

  bool transposedRhs = false;
  for (int64_t i = 0; i < op.getNumDpsInputs(); ++i) {
    AffineMap map = op.getIndexingMapsArray()[i];
    bool hasN = llvm::any_of(map.getResults(), [n](AffineExpr expression) {
      auto dim = dyn_cast<AffineDimExpr>(expression);
      return dim && dim.getPosition() == n;
    });
    if (!hasN)
      continue;
    if (auto dim =
            dyn_cast<AffineDimExpr>(map.getResult(map.getNumResults() - 1)))
      transposedRhs = dim.getPosition() == k;
    break;
  }
  if (transposedRhs) {
    // With B[..., n, k], vectorizing N requires strided/gather loads. Use a
    // temporary fallback to guarantee correct execution, that keeps M/N scalar
    // and tiles the contiguous K dimension to one native vector; a layout-aware
    // policy can replace this or this case could be entirely avoided through
    // compile-time packing. Another option would be to switch to a different
    // algorithm (away from outer-product). The public ggml repository contains
    // an example using inner product + reduction instead.
    mTile = 1;
    nTile = 1;
    kTile = vectorWidth;
  }
  for (unsigned batch : dimensions.batch)
    tiles[batch] = TileDecision{1};
  if (useHmx) {
    // M/N are fixed by the HMX hardware tile. K remains untiled here and is
    // handled by the HMX packing and tensor-unit lowering path.
    tiles[m] = TileDecision{context.target.hmxM, /*hardwareFixed=*/true};
    tiles[n] = TileDecision{context.target.hmxN, /*hardwareFixed=*/true};
    return tiles;
  }
  tiles[m] = TileDecision{
      chooseStaticTilingFactor(shape.dimensions[m].staticExtent, mTile)};
  int64_t nBound = shape.dimensions[n].staticExtent;
  // Preserve the full vector width for ragged N bounds; the tensor-semantics
  // CPUDouble pipeline peels the remainder instead of requiring an exact
  // divisor as M and K do.
  tiles[n] = TileDecision{
      ShapedType::isDynamic(nBound) ? nTile : std::min(nBound, nTile)};
  tiles[k] = TileDecision{
      chooseStaticTilingFactor(shape.dimensions[k].staticExtent, kTile)};
  return tiles;
}

} // namespace

FailureOr<std::optional<DispatchStrategy>>
selectHmxStrategy(const PlanningContext &context,
                  const DispatchShape &dispatchShape) {
  if (!context.options.enableHmxMatmul)
    return std::optional<DispatchStrategy>{};
  auto linalgOp = dyn_cast_or_null<linalg::LinalgOp>(dispatchShape.root);
  if (!linalgOp || !linalg::isaContractionOpInterface(linalgOp))
    return std::optional<DispatchStrategy>{};
  std::optional<linalg::ContractionDimensions> dimensions =
      analyzeSupportedContraction(linalgOp);
  if (!dimensions || !hasHmxElementTypes(linalgOp))
    return std::optional<DispatchStrategy>{};

  const OpShape &shape = getRootShape(dispatchShape);
  SmallVector<TileDecision> compute = inferContractionComputeTile(
      context, shape, linalgOp, *dimensions, /*useHmx=*/true);

  DispatchStrategy strategy;
  // The Hexagon target routes this otherwise-unused pipeline enum to its HMX
  // matmul pipeline.
  strategy.pipeline = IREE::CPU::LoweringPipeline::Mmt4dTilingExpert;
  strategy.rootTiling.distributionTile =
      SmallVector<TileDecision>(shape.dimensions.size());
  strategy.rootTiling.cacheTile =
      SmallVector<TileDecision>(shape.dimensions.size());
  // The HMX pack/runtime path accepts only plain matmul. Tile batch to one so
  // the HMX pipeline can rank-reduce batch_matmul before packing.
  for (unsigned batch : dimensions->batch)
    strategy.rootTiling.cacheTile[batch] =
        TileDecision{1, /*hardwareFixed=*/true};
  strategy.rootTiling.computeTile = std::move(compute);

  return std::optional<DispatchStrategy>(std::move(strategy));
}

FailureOr<std::optional<DispatchStrategy>>
selectContractionStrategy(const PlanningContext &context,
                          const DispatchShape &dispatchShape) {
  auto linalgOp = dyn_cast_or_null<linalg::LinalgOp>(dispatchShape.root);
  if (!linalgOp || !linalg::isaContractionOpInterface(linalgOp))
    return std::optional<DispatchStrategy>{};
  std::optional<linalg::ContractionDimensions> dimensions =
      analyzeSupportedContraction(linalgOp);
  if (!dimensions)
    return std::optional<DispatchStrategy>{};

  const OpShape &shape = getRootShape(dispatchShape);

  DispatchStrategy strategy;
  strategy.rootTiling.distributionTile =
      SmallVector<TileDecision>(shape.dimensions.size());
  strategy.rootTiling.cacheTile =
      SmallVector<TileDecision>(shape.dimensions.size());
  SmallVector<TileDecision> compute = inferContractionComputeTile(
      context, shape, linalgOp, *dimensions, /*useHmx=*/false);
  strategy.pipeline = IREE::CPU::LoweringPipeline::DoubleTilingExpert;
  strategy.requestLoopPeeling = linalgOp.hasPureTensorSemantics();
  strategy.rootTiling.computeTile = std::move(compute);
  for (unsigned batch : dimensions->batch)
    strategy.rootTiling.cacheTile[batch] = TileDecision{1};
  for (unsigned dimension : {dimensions->m.front(), dimensions->n.front()}) {
    strategy.rootTiling.cacheTile[dimension] =
        TileDecision{chooseStaticTilingFactor(
            shape.dimensions[dimension].staticExtent, kCacheTileCap)};
  }

  return std::optional<DispatchStrategy>(std::move(strategy));
}

FailureOr<std::optional<DispatchStrategy>>
selectUnsupportedContractionFallback(const PlanningContext &,
                                     const DispatchShape &dispatchShape) {
  auto linalgOp = dyn_cast_or_null<linalg::LinalgOp>(dispatchShape.root);
  if (!linalgOp || !linalg::isaContractionOpInterface(linalgOp) ||
      analyzeSupportedContraction(linalgOp))
    return std::optional<DispatchStrategy>{};

  DispatchStrategy strategy;
  strategy.pipeline = IREE::CPU::LoweringPipeline::Default;
  if (const OpShape *shape = findOpShape(dispatchShape, dispatchShape.root)) {
    strategy.rootTiling.distributionTile =
        SmallVector<TileDecision>(shape->dimensions.size());
    strategy.rootTiling.cacheTile =
        SmallVector<TileDecision>(shape->dimensions.size());
    strategy.rootTiling.computeTile =
        SmallVector<TileDecision>(shape->dimensions.size());
  }

  return std::optional<DispatchStrategy>(std::move(strategy));
}

FailureOr<std::optional<OpComputeTilePlan>>
selectContractionComputeTile(const PlanningContext &context,
                             const DispatchShape &,
                             const DispatchStrategy &strategy,
                             const PipelineContract &, const OpShape &opShape) {
  auto linalgOp = dyn_cast<linalg::LinalgOp>(opShape.op);
  if (!linalgOp || !linalg::isaContractionOpInterface(linalgOp))
    return std::optional<OpComputeTilePlan>{};
  std::optional<linalg::ContractionDimensions> dimensions =
      analyzeSupportedContraction(linalgOp);
  if (!dimensions)
    return std::optional<OpComputeTilePlan>{};

  bool useHmx =
      strategy.pipeline == IREE::CPU::LoweringPipeline::Mmt4dTilingExpert &&
      hasHmxElementTypes(linalgOp);
  SmallVector<TileDecision> compute = inferContractionComputeTile(
      context, opShape, linalgOp, *dimensions, useHmx);

  return std::optional<OpComputeTilePlan>(
      OpComputeTilePlan{opShape.op, std::move(compute)});
}

} // namespace mlir::iree_compiler::hexagon::codegen::planning

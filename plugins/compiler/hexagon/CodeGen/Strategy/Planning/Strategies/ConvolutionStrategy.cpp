// Copyright 2026 RooflineAI GmbH
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "Strategy.h"
#include "StrategySupport.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/IR/LinalgInterfaces.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "llvm/ADT/STLExtras.h"

namespace mlir::iree_compiler::hexagon::codegen::planning {
namespace {

bool is2DConv(linalg::LinalgOp op) {
  return linalg::isaConvolutionOpOfType<linalg::Conv2DNhwcHwcfOp>(op) ||
         linalg::isaConvolutionOpOfType<linalg::Conv2DNchwFchwOp>(op);
}

bool is2DDepthwiseConv(linalg::LinalgOp op) {
  return linalg::isaConvolutionOpOfType<linalg::DepthwiseConv2DNhwcHwcOp>(op);
}

bool is2DPooling(linalg::LinalgOp op) {
  return linalg::isaConvolutionOpOfType<linalg::PoolingNhwcSumOp>(op) ||
         linalg::isaConvolutionOpOfType<linalg::PoolingNhwcMaxOp>(op) ||
         linalg::isaConvolutionOpOfType<linalg::PoolingNhwcMaxUnsignedOp>(op) ||
         linalg::isaConvolutionOpOfType<linalg::PoolingNhwcMinOp>(op) ||
         linalg::isaConvolutionOpOfType<linalg::PoolingNhwcMinUnsignedOp>(op) ||
         linalg::isaConvolutionOpOfType<linalg::PoolingNchwSumOp>(op) ||
         linalg::isaConvolutionOpOfType<linalg::PoolingNchwMaxOp>(op);
}

bool isNchw(linalg::LinalgOp op) {
  return linalg::isaConvolutionOpOfType<linalg::Conv2DNchwFchwOp>(op) ||
         linalg::isaConvolutionOpOfType<linalg::PoolingNchwSumOp>(op) ||
         linalg::isaConvolutionOpOfType<linalg::PoolingNchwMaxOp>(op);
}

bool isSupportedConvolution(linalg::LinalgOp op) {
  return is2DConv(op) || is2DDepthwiseConv(op) || is2DPooling(op);
}

Type getConvolutionVectorType(linalg::LinalgOp op) {
  Operation *operation = op.getOperation();
  if (operation->getNumResults() > 0)
    return operation->getResult(0).getType();
  if (operation->getNumOperands() > 0)
    return operation->getOperand(0).getType();
  return {};
}

/// The shapes currently selected here do not reflect any reasoning and have not
/// been optimized. They were just observed to work on a subset of models and
/// are only intended to guarantee correctness.
SmallVector<TileDecision>
inferConvolutionDistributionTile(const PlanningContext &context,
                                 const OpShape &shape, linalg::LinalgOp op) {
  Type vectorType = getConvolutionVectorType(op);
  SmallVector<TileDecision> tiles(shape.dimensions.size());
  if (!vectorType)
    return tiles;
  int64_t vectorWidth = getTypeNativeVectorWidth(context, vectorType);
  SmallVector<int64_t> preferred =
      is2DConv(op) ? SmallVector<int64_t>{1, 1, vectorWidth, vectorWidth,
                                          1, 1, vectorWidth}
                   : SmallVector<int64_t>{1,           1, vectorWidth,
                                          vectorWidth, 1, vectorWidth};
  if (isNchw(op)) {
    SmallVector<int64_t> permutation =
        is2DConv(op) ? SmallVector<int64_t>{0, 3, 1, 2, 6, 4, 5}
                     : SmallVector<int64_t>{0, 3, 1, 2, 4, 5};
    SmallVector<int64_t> permuted(preferred.size());
    for (auto [newPosition, oldPosition] : llvm::enumerate(permutation))
      permuted[newPosition] = preferred[oldPosition];
    preferred = std::move(permuted);
  }
  for (const LocalDim &dimension : shape.dimensions) {
    if (dimension.iteratorType != utils::IteratorType::parallel)
      continue;
    int64_t target = dimension.position < preferred.size()
                         ? preferred[dimension.position]
                         : 1;
    tiles[dimension.position] =
        TileDecision{chooseStaticTilingFactor(dimension.staticExtent, target)};
  }
  return tiles;
}

/// Narrows the distribution tile to one native vector row: only the loop that
/// indexes the innermost result dimension keeps its tile, every other tiled
/// parallel loop is tiled to 1. The fused elementwise consumers are vectorized
/// at this shape, and a multi-row vector is unrolled op by op over all of its
/// rows. A wide consumer such as an exact-erf GELU then keeps hundreds of rows
/// live at once and spills (138 KiB of stack for a 32x25 f32 tile).
SmallVector<TileDecision>
inferConvolutionComputeTile(const PlanningContext &context,
                            const OpShape &shape, linalg::LinalgOp op) {
  SmallVector<TileDecision> tiles =
      inferConvolutionDistributionTile(context, shape, op);
  if (op.getNumDpsInits() != 1)
    return tiles;
  AffineMap resultMap = op.getMatchingIndexingMap(op.getDpsInitOperand(0));
  if (resultMap.getNumResults() == 0)
    return tiles;
  auto innermost = dyn_cast<AffineDimExpr>(resultMap.getResults().back());
  if (!innermost)
    return tiles;
  for (const LocalDim &dimension : shape.dimensions) {
    if (dimension.iteratorType != utils::IteratorType::parallel ||
        dimension.position == innermost.getPosition())
      continue;
    TileDecision &tile = tiles[dimension.position];
    if (tile.size > 1)
      tile = TileDecision{1};
  }
  return tiles;
}

} // namespace

FailureOr<std::optional<DispatchStrategy>>
selectConvolutionStrategy(const PlanningContext &context,
                          const DispatchShape &dispatchShape) {
  auto linalgOp = dyn_cast_or_null<linalg::LinalgOp>(dispatchShape.root);
  if (!linalgOp || !isSupportedConvolution(linalgOp))
    return std::optional<DispatchStrategy>{};

  const OpShape &shape = getRootShape(dispatchShape);

  DispatchStrategy strategy;
  strategy.pipeline =
      IREE::Hexagon::LoweringPipeline::ConvTileAndDecomposeExpert;
  strategy.rootTiling.distributionTile =
      inferConvolutionDistributionTile(context, shape, linalgOp);
  strategy.rootTiling.cacheTile =
      SmallVector<TileDecision>(shape.dimensions.size());
  strategy.rootTiling.computeTile =
      inferConvolutionComputeTile(context, shape, linalgOp);

  return std::optional<DispatchStrategy>(std::move(strategy));
}

FailureOr<std::optional<OpComputeTilePlan>>
selectConvolutionComputeTile(const PlanningContext &context,
                             const DispatchShape &, const DispatchStrategy &,
                             const PipelineContract &, const OpShape &opShape) {
  auto linalgOp = dyn_cast<linalg::LinalgOp>(opShape.op);
  if (!linalgOp || !isSupportedConvolution(linalgOp))
    return std::optional<OpComputeTilePlan>{};

  SmallVector<TileDecision> compute =
      inferConvolutionComputeTile(context, opShape, linalgOp);

  return std::optional<OpComputeTilePlan>(
      OpComputeTilePlan{opShape.op, std::move(compute)});
}

} // namespace mlir::iree_compiler::hexagon::codegen::planning

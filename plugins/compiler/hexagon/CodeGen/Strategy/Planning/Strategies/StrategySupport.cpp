// Copyright 2026 RooflineAI GmbH
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "StrategySupport.h"

#include "iree/compiler/Dialect/Util/IR/UtilTypes.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/TypeUtilities.h"
#include "llvm/ADT/STLExtras.h"

#include <algorithm>

namespace mlir::iree_compiler::hexagon::codegen::planning {
namespace {

Type getDefaultVectorType(Operation *op) {
  if (op->getNumResults() > 0)
    return op->getResult(0).getType();
  if (op->getNumOperands() > 0)
    return op->getOperand(0).getType();
  return {};
}

} // namespace

int64_t getTypeNativeVectorWidth(const PlanningContext &context, Type type) {
  Type elementType = getElementTypeOrSelf(type);
  if (!elementType.isIntOrFloat())
    return 1;
  int64_t byteWidth = IREE::Util::getRoundedElementByteWidth(elementType);
  return std::max<int64_t>(1, context.target.nativeVectorBytes / byteWidth);
}

int64_t chooseStaticTilingFactor(int64_t staticBound, int64_t preferred,
                                 int64_t multipleOf) {
  if (preferred <= 0)
    return 0;
  if (ShapedType::isDynamic(staticBound))
    return preferred;

  int64_t fallback = std::min(staticBound, preferred);
  int64_t step = std::max<int64_t>(1, multipleOf);

  if (staticBound <= 0 || staticBound % step != 0 || preferred < step)
    return fallback;

  // Search in units of `step`: candidate = divisor * step divides staticBound
  // exactly iff divisor divides normalizedBound.
  int64_t normalizedBound = staticBound / step;
  int64_t normalizedPreferred = preferred / step;
  int64_t smallDivisor = 1;
  for (; smallDivisor <= normalizedBound / smallDivisor; ++smallDivisor) {
    if (normalizedBound % smallDivisor != 0)
      continue;
    int64_t largeDivisor = normalizedBound / smallDivisor;
    if (largeDivisor <= normalizedPreferred)
      return largeDivisor * step;
  }
  for (--smallDivisor; smallDivisor >= 1; --smallDivisor) {
    if (normalizedBound % smallDivisor == 0 &&
        smallDivisor <= normalizedPreferred)
      return smallDivisor * step;
  }
  return fallback;
}

SmallVector<TileDecision>
inferInnermostParallelComputeTile(const PlanningContext &context,
                                  const OpShape &shape, Operation *op) {
  Type vectorType = getDefaultVectorType(op);
  SmallVector<TileDecision> tiles(shape.dimensions.size());
  if (!vectorType)
    return tiles;
  int64_t vectorWidth = getTypeNativeVectorWidth(context, vectorType);
  SmallVector<unsigned> parallelDimensions;
  for (const LocalDim &dimension : shape.dimensions) {
    if (dimension.iteratorType == utils::IteratorType::parallel)
      parallelDimensions.push_back(dimension.position);
  }
  if (parallelDimensions.empty())
    return tiles;
  if (parallelDimensions.size() > 1) {
    for (unsigned dimension : llvm::drop_end(parallelDimensions))
      tiles[dimension] = TileDecision{1};
  }
  unsigned innermost = parallelDimensions.back();
  int64_t bound = shape.dimensions[innermost].staticExtent;
  tiles[innermost] =
      TileDecision{ShapedType::isDynamic(bound) ? vectorWidth
                                                : std::min(bound, vectorWidth)};
  return tiles;
}

} // namespace mlir::iree_compiler::hexagon::codegen::planning

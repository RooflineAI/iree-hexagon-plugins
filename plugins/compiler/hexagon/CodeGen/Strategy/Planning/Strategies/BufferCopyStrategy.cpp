// Copyright 2026 RooflineAI GmbH
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "Strategy.h"
#include "StrategySupport.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"

namespace mlir::iree_compiler::hexagon::codegen::planning {
namespace {

bool isBufferLikeCopy(linalg::LinalgOp op) {
  return op.hasPureBufferSemantics() && linalg::isaCopyOpInterface(op);
}

SmallVector<TileDecision>
inferBufferCopyComputeTile(const PlanningContext &context, const OpShape &shape,
                           Operation *op) {
  return inferInnermostParallelComputeTile(context, shape, op);
}

} // namespace

FailureOr<std::optional<DispatchStrategy>>
selectBufferCopyStrategy(const PlanningContext &context,
                         const DispatchShape &dispatchShape) {
  auto linalgOp = dyn_cast_or_null<linalg::LinalgOp>(dispatchShape.root);
  if (!linalgOp || !isBufferLikeCopy(linalgOp))
    return std::optional<DispatchStrategy>{};

  const OpShape &shape = getRootShape(dispatchShape);
  SmallVector<TileDecision> compute =
      inferBufferCopyComputeTile(context, shape, dispatchShape.root);

  DispatchStrategy strategy;
  strategy.pipeline = IREE::CPU::LoweringPipeline::BufferOpsTileAndVectorize;
  strategy.rootTiling.distributionTile = compute;
  strategy.rootTiling.cacheTile =
      SmallVector<TileDecision>(shape.dimensions.size());
  strategy.rootTiling.computeTile = std::move(compute);

  return std::optional<DispatchStrategy>(std::move(strategy));
}

FailureOr<std::optional<OpComputeTilePlan>>
selectBufferCopyComputeTile(const PlanningContext &context,
                            const DispatchShape &, const DispatchStrategy &,
                            const PipelineContract &, const OpShape &opShape) {
  auto linalgOp = dyn_cast<linalg::LinalgOp>(opShape.op);
  if (!linalgOp || !isBufferLikeCopy(linalgOp))
    return std::optional<OpComputeTilePlan>{};

  SmallVector<TileDecision> compute =
      inferBufferCopyComputeTile(context, opShape, opShape.op);

  return std::optional<OpComputeTilePlan>(
      OpComputeTilePlan{opShape.op, std::move(compute)});
}

} // namespace mlir::iree_compiler::hexagon::codegen::planning

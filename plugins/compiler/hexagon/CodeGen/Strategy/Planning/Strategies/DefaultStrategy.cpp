// Copyright 2026 RooflineAI GmbH
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "Strategy.h"
#include "StrategySupport.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Interfaces/TilingInterface.h"

namespace mlir::iree_compiler::hexagon::codegen::planning {
DispatchStrategy makeCPUDefaultStrategy(const DispatchShape &dispatchShape) {
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

  return strategy;
}

FailureOr<std::optional<DispatchStrategy>>
selectDefaultTilingInterfaceStrategy(const PlanningContext &context,
                                     const DispatchShape &dispatchShape) {
  auto tilingOp = dyn_cast_or_null<TilingInterface>(dispatchShape.root);
  if (!tilingOp || tilingOp.getLoopIteratorTypes().empty())
    return std::optional<DispatchStrategy>{};

  const OpShape &shape = getRootShape(dispatchShape);
  SmallVector<TileDecision> compute =
      inferInnermostParallelComputeTile(context, shape, dispatchShape.root);
  DispatchStrategy strategy;
  strategy.pipeline = isa<linalg::LinalgOp>(dispatchShape.root)
                          ? IREE::CPU::LoweringPipeline::DoubleTilingExpert
                          : IREE::CPU::LoweringPipeline::Default;
  if (auto linalgOp = dyn_cast<linalg::LinalgOp>(dispatchShape.root))
    strategy.requestLoopPeeling = linalgOp.hasPureTensorSemantics();
  strategy.rootTiling.distributionTile =
      SmallVector<TileDecision>(shape.dimensions.size());
  strategy.rootTiling.cacheTile =
      SmallVector<TileDecision>(shape.dimensions.size());
  strategy.rootTiling.computeTile = std::move(compute);

  return std::optional<DispatchStrategy>(std::move(strategy));
}

FailureOr<std::optional<OpComputeTilePlan>>
selectDefaultTilingInterfaceComputeTile(const PlanningContext &context,
                                        const DispatchShape &,
                                        const DispatchStrategy &,
                                        const PipelineContract &,
                                        const OpShape &opShape) {
  auto tilingOp = dyn_cast<TilingInterface>(opShape.op);
  if (!tilingOp || tilingOp.getLoopIteratorTypes().empty())
    return std::optional<OpComputeTilePlan>{};

  SmallVector<TileDecision> compute =
      inferInnermostParallelComputeTile(context, opShape, opShape.op);

  return std::optional<OpComputeTilePlan>(
      OpComputeTilePlan{opShape.op, std::move(compute)});
}

} // namespace mlir::iree_compiler::hexagon::codegen::planning

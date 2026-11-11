// Copyright 2026 RooflineAI GmbH
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HEXAGON_CODEGEN_PLANNING_STRATEGIES_STRATEGYSUPPORT_H_
#define IREE_HEXAGON_CODEGEN_PLANNING_STRATEGIES_STRATEGYSUPPORT_H_

#include "../DispatchPlanTypes.h"

#include "mlir/Dialect/Linalg/IR/LinalgInterfaces.h"
#include "mlir/Support/LogicalResult.h"

#include <optional>

namespace mlir::iree_compiler::hexagon::codegen::planning {

int64_t getTypeNativeVectorWidth(const PlanningContext &context, Type type);

/// For a static bound, chooses the largest exact divisor no greater than the
/// preferred size from candidates aligned to `multipleOf`. Dynamic bounds keep
/// the preferred size. If no aligned divisor exists, falls back to the smaller
/// of the static bound and preferred size.
int64_t chooseStaticTilingFactor(int64_t staticBound, int64_t preferred,
                                 int64_t multipleOf = 1);

/// Tile outer parallel dimensions to one and the innermost parallel dimension
/// to the native element vector width.
SmallVector<TileDecision>
inferInnermostParallelComputeTile(const PlanningContext &context,
                                  const OpShape &shape, Operation *op);

/// The tightest constant upper bound of loop `loop` of `op` over the operand
/// dimensions it indexes directly: its static extent, or for a dynamic loop
/// e.g. the maximum from a `util.assume.int`. std::nullopt if there is none.
std::optional<int64_t> getLoopUpperBound(linalg::LinalgOp op, unsigned loop);

} // namespace mlir::iree_compiler::hexagon::codegen::planning

#endif // IREE_HEXAGON_CODEGEN_PLANNING_STRATEGIES_STRATEGYSUPPORT_H_

// Copyright 2026 RooflineAI GmbH
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef ROOF_HEXAGON_CODEGEN_PLANNING_VECTORTILESEARCH_H_
#define ROOF_HEXAGON_CODEGEN_PLANNING_VECTORTILESEARCH_H_

#include "DispatchPlanTypes.h"

#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/StringMap.h"

namespace mlir::iree_compiler::hexagon::codegen::planning {

/// Tuning knobs for `searchVectorTiling`'s cost function
struct VectorTileSearchConfig {
  /// Registers kept free below architecturalVectorRegisterCount, to cover
  /// the estimator's documented under-estimates and avoid spills
  int64_t registerMargin = 8;
  /// caps the unroll count to reduce code and compile time
  int64_t maxUnrolledVectorOps = 128;
  /// Tighter unroll caps for expensive body ops
  llvm::StringMap<int64_t> maxUnrolledVectorOpsByExpansionOp = {
      {"math.tanh", 1},
      {"math.exp", 1},
      {"math.rsqrt", 1},
      {"math.sqrt", 1},
  };
};

/// enumerates every legal vector tile inside the root's VTCM tile, scores each
/// with `DispatchRegisterGraph::evaluate`, and replaces
/// `strategy.rootTiling.computeTile` with the cheapest one that fits the
/// register budget.
LogicalResult
searchVectorTiling(const PlanningContext &context,
                      const DispatchShape &dispatchShape,
                      DispatchStrategy &strategy,
                      const VectorTileSearchConfig &config = {});

} // namespace mlir::iree_compiler::hexagon::codegen::planning

#endif // ROOF_HEXAGON_CODEGEN_PLANNING_VECTORTILESEARCH_H_

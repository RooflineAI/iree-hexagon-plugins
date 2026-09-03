// Copyright 2026 RooflineAI GmbH
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef ROOF_HEXAGON_CODEGEN_PLANNING_ROOTTILEPROPAGATION_H_
#define ROOF_HEXAGON_CODEGEN_PLANNING_ROOTTILEPROPAGATION_H_

#include "DispatchPlanTypes.h"
#include "PipelineContract.h"

#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"

namespace mlir::iree_compiler::hexagon::codegen::planning {

/// `DispatchShape` records which dimensions operations share, but not which
/// operations are fused into each loop created by tiling the root. That
/// placement determines which root tile bounds a non-root operation:
///
///   - the root's Common level builds the fusion loop nest containing all
///     operations fused at that level, so a root parallel dimension bounds all
///     of them;
///   - the root's Reduction level starts producer fusion from the root's DPS
///     input operands, while excluding the root destination and disabling
///     consumer fusion. Fusion then recursively follows the dependencies of
///     those producers, including DPS init operands. A root reduction dimension
///     therefore bounds operations in that transitive producer graph, but not
///     unrelated consumers merely sharing the same dimension.
///
/// For example, consider this schematic IR:
///
/// ```mlir
/// // Root loops: (m, n, k); result shape: (m, n).
/// %root = linalg.matmul ins(%lhs, %rhs) outs(%init)
///
/// // Maps: %root(m, n), %lhs(m, k) -> %output(m, n, k).
/// %consumer = linalg.generic
///     ins(%root, %lhs) outs(%consumerInit)
/// ```
///
/// The consumer shares the root's reduction dimension `k` through `%lhs`, but
/// not through `%root`. It may be fused into the root's Common loop nest, but
/// consumer fusion is disabled at the Reduction level, so it remains outside
/// the root's reduction loop.
class RootFusionBounds {
public:
  RootFusionBounds(const DispatchShape &dispatchShape,
                   const RootTilingPlan &rootTiling);

  /// The extent `op` sees on `global` once fused, or 0 when nothing bounds it.
  int64_t getBound(Operation *op, GlobalDimId global) const;

private:
  llvm::SmallDenseMap<GlobalDimId, int64_t> parallelTiles;
  llvm::SmallDenseMap<GlobalDimId, int64_t> reductionTiles;
  llvm::SmallDenseSet<Operation *> inputProducers;
};

/// Bounds every already-selected non-root compute tile by the root's fusion
/// tile so that the downstream pipeline can realize it.
///
/// This runs after independent per-operation selection and mutates plan state
/// only; it never mutates IR. It is a no-op for pipelines that do not anchor
/// Common tiling on the root, because those give every operation its own loop.
LogicalResult reconcileNonRootComputeTiles(const PlanningContext &context,
                                           const DispatchShape &dispatchShape,
                                           const PipelineContract &contract,
                                           DispatchPlan &plan);

} // namespace mlir::iree_compiler::hexagon::codegen::planning

#endif // ROOF_HEXAGON_CODEGEN_PLANNING_ROOTTILEPROPAGATION_H_

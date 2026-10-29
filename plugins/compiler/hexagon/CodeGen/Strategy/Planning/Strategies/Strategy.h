// Copyright 2026 RooflineAI GmbH
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HEXAGON_CODEGEN_PLANNING_STRATEGIES_STRATEGY_H_
#define IREE_HEXAGON_CODEGEN_PLANNING_STRATEGIES_STRATEGY_H_

#include "../DispatchPlanTypes.h"
#include "../PipelineContract.h"

#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"

#include <optional>

namespace mlir::linalg {
class LinalgOp;
} // namespace mlir::linalg

namespace mlir::iree_compiler::hexagon::codegen::planning {

/// Returns whether `op` has the element types HMX supports: f16 inputs and an
/// f16 or f32 output.
bool hasHmxElementTypes(linalg::LinalgOp op);

/// Returns whether `op` is a contraction the HMX strategy can plan: one M, N
/// and K dimension, at most one batch dimension, and HMX element types.
/// HMX lowering additionally requires isHmxLowerableMatmul.
bool isHmxCandidateContraction(linalg::LinalgOp op);

/// Returns whether `op` is what HexagonConvertMatmulToHmx lowers: a named
/// linalg.matmul or linalg.batch_matmul with HMX element types, one M, N and K
/// dimension, at most one batch dimension, and an output that keeps M before N.
/// The batch dimension is dropped before packing, so its position does not
/// matter.
bool isHmxLowerableMatmul(linalg::LinalgOp op);

/// A dispatch strategy selector has three outcomes:
///   - success(nullopt): it does not apply;
///   - success(strategy): it selected a complete dispatch strategy;
///   - failure: it matched but could not produce a legal strategy.
using StrategySelector = FailureOr<std::optional<DispatchStrategy>> (*)(
    const PlanningContext &, const DispatchShape &);

struct StrategyCandidate {
  llvm::StringRef name;
  StrategySelector select = nullptr;
};

/// Per-op compute-tile policies use the same three-way result.
using ComputeTilePolicySelector = FailureOr<std::optional<OpComputeTilePlan>> (
        *)(const PlanningContext &, const DispatchShape &,
           const DispatchStrategy &, const PipelineContract &, const OpShape &);

struct TilePolicy {
  // name is only used for logging and tracing
  llvm::StringRef name;
  ComputeTilePolicySelector select = nullptr;
};

// === Strategies ===
FailureOr<std::optional<DispatchStrategy>>
selectBufferCopyStrategy(const PlanningContext &, const DispatchShape &);
FailureOr<std::optional<DispatchStrategy>>
selectConvolutionStrategy(const PlanningContext &, const DispatchShape &);
FailureOr<std::optional<DispatchStrategy>>
selectHmxStrategy(const PlanningContext &, const DispatchShape &);
FailureOr<std::optional<DispatchStrategy>>
selectContractionStrategy(const PlanningContext &, const DispatchShape &);
FailureOr<std::optional<DispatchStrategy>>
selectUnsupportedContractionFallback(const PlanningContext &,
                                     const DispatchShape &);
FailureOr<std::optional<DispatchStrategy>>
selectGenericStrategy(const PlanningContext &, const DispatchShape &);
FailureOr<std::optional<DispatchStrategy>>
selectDefaultTilingInterfaceStrategy(const PlanningContext &,
                                     const DispatchShape &);
DispatchStrategy makeHexagonDefaultStrategy(const DispatchShape &);
// === Tiling policies ===
FailureOr<std::optional<OpComputeTilePlan>>
selectBufferCopyComputeTile(const PlanningContext &, const DispatchShape &,
                            const DispatchStrategy &, const PipelineContract &,
                            const OpShape &);
FailureOr<std::optional<OpComputeTilePlan>>
selectConvolutionComputeTile(const PlanningContext &, const DispatchShape &,
                             const DispatchStrategy &, const PipelineContract &,
                             const OpShape &);
FailureOr<std::optional<OpComputeTilePlan>>
selectContractionComputeTile(const PlanningContext &, const DispatchShape &,
                             const DispatchStrategy &, const PipelineContract &,
                             const OpShape &);
FailureOr<std::optional<OpComputeTilePlan>>
selectGenericComputeTile(const PlanningContext &, const DispatchShape &,
                         const DispatchStrategy &, const PipelineContract &,
                         const OpShape &);
FailureOr<std::optional<OpComputeTilePlan>>
selectDefaultTilingInterfaceComputeTile(const PlanningContext &,
                                        const DispatchShape &,
                                        const DispatchStrategy &,
                                        const PipelineContract &,
                                        const OpShape &);

} // namespace mlir::iree_compiler::hexagon::codegen::planning

#endif // IREE_HEXAGON_CODEGEN_PLANNING_STRATEGIES_STRATEGY_H_

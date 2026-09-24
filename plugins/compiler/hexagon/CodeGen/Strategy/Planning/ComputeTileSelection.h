// Copyright 2026 RooflineAI GmbH
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef ROOF_HEXAGON_CODEGEN_PLANNING_COMPUTETILESELECTION_H_
#define ROOF_HEXAGON_CODEGEN_PLANNING_COMPUTETILESELECTION_H_

#include "DispatchPlanTypes.h"
#include "PipelineContract.h"

#include "mlir/Support/LogicalResult.h"

namespace mlir::iree_compiler::hexagon::codegen::planning {

/// After strategy and contract resolution, selects one compute tile value per
/// loop dimension for a non-root operation.
///
/// Root compute tiling is already complete in RootTilingPlan. The initial
/// implementation treats each remaining op independently and does not copy
/// root tile values to it. CPU vector tiling levels are not selected here.
FailureOr<std::optional<OpComputeTilePlan>> selectNonRootComputeTile(
    const PlanningContext &context, const DispatchShape &dispatchShape,
    const DispatchStrategy &strategy, const PipelineContract &pipelineContract,
    const OpShape &opShape);

} // namespace mlir::iree_compiler::hexagon::codegen::planning

#endif // ROOF_HEXAGON_CODEGEN_PLANNING_COMPUTETILESELECTION_H_

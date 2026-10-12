// Copyright 2026 RooflineAI GmbH
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef ROOF_HEXAGON_CODEGEN_PLANNING_VTCMPLANNING_H_
#define ROOF_HEXAGON_CODEGEN_PLANNING_VTCMPLANNING_H_

#include "DispatchPlanTypes.h"
#include "PipelineContract.h"

#include "mlir/Support/LogicalResult.h"

namespace mlir::iree_compiler::hexagon::codegen::planning {

/// After pipeline-contract resolution, derives optional/required root VTCM
/// resources and reconciles them with the initial strategy.
/// This mutates plan state only; it never mutates IR.
LogicalResult planVTCMTiling(const PlanningContext &context,
                             const DispatchShape &dispatchShape,
                             const PipelineContract &contract,
                             DispatchStrategy &strategy);

/// FIXME:
///  For a plan without VTCM staging, estimates the VTCM that heap temporaries
///  take once compute tiles are final, and records it in the plan.
///  This mutates plan state only; it never mutates IR.
///  This is necessary for mallocs that end up in VTCM
void planHeapTemporaryVTCM(const PlanningContext &context,
                           const DispatchShape &dispatchShape,
                           DispatchPlan &plan);

} // namespace mlir::iree_compiler::hexagon::codegen::planning

#endif // ROOF_HEXAGON_CODEGEN_PLANNING_VTCMPLANNING_H_

// Copyright 2026 RooflineAI GmbH
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef ROOF_HEXAGON_CODEGEN_PLANNING_PLANVERIFICATION_H_
#define ROOF_HEXAGON_CODEGEN_PLANNING_PLANVERIFICATION_H_

#include "DispatchPlanTypes.h"
#include "PipelineContract.h"

#include "mlir/Support/LogicalResult.h"

namespace mlir::iree_compiler::hexagon::codegen::planning {

/// Verifies the complete, reconciled plan before any MLIR attributes are
/// built. Verification diagnoses invalid ownership, rank, fixedness,
/// divisibility, and pipeline-consumption states; it does not repair them.
LogicalResult verifyDispatchPlan(const PlanningContext &context,
                                 const DispatchShape &dispatchShape,
                                 const DispatchPlan &plan,
                                 const PipelineContract &pipelineContract);

} // namespace mlir::iree_compiler::hexagon::codegen::planning

#endif // ROOF_HEXAGON_CODEGEN_PLANNING_PLANVERIFICATION_H_

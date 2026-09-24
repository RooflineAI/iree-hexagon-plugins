// Copyright 2026 RooflineAI GmbH
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef ROOF_HEXAGON_CODEGEN_PLANNING_DISPATCHANALYSIS_H_
#define ROOF_HEXAGON_CODEGEN_PLANNING_DISPATCHANALYSIS_H_

#include "DispatchPlanTypes.h"

#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir::iree_compiler::hexagon::codegen::planning {

/// Selects and records the root, constructs the unified dimension graph, and
/// classifies each global dimension's coverage.
/// This function does not make any decision or mutate the IR.
FailureOr<DispatchShape>
analyzeDispatch(const PlanningContext &context,
                ArrayRef<Operation *> orderedComputeOps);

} // namespace mlir::iree_compiler::hexagon::codegen::planning

#endif // ROOF_HEXAGON_CODEGEN_PLANNING_DISPATCHANALYSIS_H_

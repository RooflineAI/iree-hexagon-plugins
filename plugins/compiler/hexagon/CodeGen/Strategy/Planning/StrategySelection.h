// Copyright 2026 RooflineAI GmbH
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef ROOF_HEXAGON_CODEGEN_PLANNING_STRATEGYSELECTION_H_
#define ROOF_HEXAGON_CODEGEN_PLANNING_STRATEGYSELECTION_H_

#include "DispatchPlanTypes.h"

#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir::iree_compiler::hexagon::codegen::planning {

/// Tries the ordered strategy registry and returns the first successful
/// dispatch strategy.
FailureOr<DispatchStrategy>
selectDispatchStrategy(const PlanningContext &context,
                       const DispatchShape &dispatchShape);

} // namespace mlir::iree_compiler::hexagon::codegen::planning

#endif // ROOF_HEXAGON_CODEGEN_PLANNING_STRATEGYSELECTION_H_

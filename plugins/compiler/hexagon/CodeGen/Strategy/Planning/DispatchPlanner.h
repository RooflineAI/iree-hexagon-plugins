// Copyright 2026 RooflineAI GmbH
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef ROOF_HEXAGON_CODEGEN_PLANNING_DISPATCHPLANNER_H_
#define ROOF_HEXAGON_CODEGEN_PLANNING_DISPATCHPLANNER_H_

#include "DispatchPlanTypes.h"

#include "mlir/Support/LogicalResult.h"

namespace mlir::iree_compiler::hexagon::codegen::planning {

/// Plans, verifies, encodes, and applies the lowering configuration for one
/// dispatch. This is the sole mutation entry point for the planning library.
LogicalResult configureDispatch(FunctionOpInterface entryPoint,
                                const PlanningOptions &options);

} // namespace mlir::iree_compiler::hexagon::codegen::planning

#endif // ROOF_HEXAGON_CODEGEN_PLANNING_DISPATCHPLANNER_H_

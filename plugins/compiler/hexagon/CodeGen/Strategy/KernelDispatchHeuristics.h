// Copyright 2026 RooflineAI GmbH
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef ROOF_HEXAGON_CODEGEN_KERNELDISPATCHHEURISTICS_H_
#define ROOF_HEXAGON_CODEGEN_KERNELDISPATCHHEURISTICS_H_

#include <optional>

#include "KernelDispatchTypes.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/IR/Operation.h"
#include "mlir/Interfaces/FunctionInterfaces.h"

namespace mlir::iree_compiler::hexagon::codegen {

/// Returns true when an operation should receive a Hexagon lowering config.
///
/// This is shared by root selection and propagation so both phases reason about
/// the same subset of compute ops.
bool shouldSetLoweringConfig(Operation *op);

/// Returns the Hexagon lowering plan for a single op
std::optional<OpLoweringPlan>
inferOpLoweringPlan(FunctionOpInterface entryPointFn, Operation *op,
                    const PolicyConfig &policyConfig);

/// Returns the root lowering plan, including the downstream pipeline
RootLoweringPlan selectRootLoweringPlan(FunctionOpInterface entryPointFn,
                                        Operation *rootOperation);

} // namespace mlir::iree_compiler::hexagon::codegen

#endif // ROOF_HEXAGON_CODEGEN_KERNELDISPATCHHEURISTICS_H_

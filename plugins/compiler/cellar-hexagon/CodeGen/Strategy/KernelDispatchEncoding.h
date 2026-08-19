// Copyright 2026 RooflineAI GmbH
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef ROOF_HEXAGON_CODEGEN_KERNELDISPATCHENCODING_H_
#define ROOF_HEXAGON_CODEGEN_KERNELDISPATCHENCODING_H_

#include "KernelDispatchTypes.h"

#include "mlir/IR/Operation.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir::iree_compiler::cellar_hexagon::codegen {

/// Applies the selected root plan by setting both lowering config and
/// translation info on the root op / entry point.
LogicalResult applyRootLoweringPlan(FunctionOpInterface entryPointFn,
                                    Operation *rootOperation,
                                    const RootLoweringPlan &rootPlan);

/// Applies the lowering config for a non-root op.
void applyOpLoweringPlan(Operation *op, const OpLoweringPlan &opPlan);

} // namespace mlir::iree_compiler::cellar_hexagon::codegen

#endif // ROOF_HEXAGON_CODEGEN_KERNELDISPATCHENCODING_H_

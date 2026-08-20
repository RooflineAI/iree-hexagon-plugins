// Copyright 2026 RooflineAI GmbH
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef ROOF_HEXAGON_CODEGEN_KERNELDISPATCHPROPAGATION_H_
#define ROOF_HEXAGON_CODEGEN_KERNELDISPATCHPROPAGATION_H_

#include "KernelDispatchTypes.h"

#include "mlir/IR/Operation.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"

#include <utility>

namespace mlir::iree_compiler::hexagon::codegen {

/// Derives non-root lowering plans from the selected root plan.
class NonRootPlanPropagator {
public:
  NonRootPlanPropagator(FunctionOpInterface entryPointFn,
                        Operation *rootOperation,
                        llvm::ArrayRef<Operation *> computeOps,
                        const RootLoweringPlan &rootPlan);

  /// Infers the lowering plans for non-root compute ops that should receive a
  /// Hexagon lowering config.
  llvm::SmallVector<std::pair<Operation *, OpLoweringPlan>, 0>
  inferPlans() const;

private:
  FunctionOpInterface entryPointFn;
  Operation *rootOperation;
  llvm::SmallVector<Operation *> computeOps;
  RootLoweringPlan rootPlan;
};

} // namespace mlir::iree_compiler::hexagon::codegen

#endif // ROOF_HEXAGON_CODEGEN_KERNELDISPATCHPROPAGATION_H_

// Copyright 2025 RooflineAI GmbH
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// This header declares the Hexagon-specific conversion-to-LLVM pass factories.

#ifndef ROOF_HEXAGON_CONVERSION_HEXAGONCONVERTTOLLVM_H_
#define ROOF_HEXAGON_CONVERSION_HEXAGONCONVERTTOLLVM_H_

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir::iree_compiler::hexagon::codegen {

// Full conversion in a single pass (default behavior, identical to LLVMCPU).
std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createHexagonConvertToLLVMPass(bool reassociateFpReductions);

// Explicitly split conversion for hybrid Hexagon pipelines:
// phase 1 lowers HAL ABI/function/vector/math/cf but keeps
// hal.interface.binding.subspan + memref finalization for phase 2.
// This split is required when reusing passes from hexagon-mlir
// In its current state, phase 1 is a subset of phase 2.
std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createHexagonConvertToLLVMPassPhase1(bool reassociateFpReductions);

// Phase 2 completes lowering once address spaces have been normalized.
std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createHexagonConvertToLLVMPassPhase2(bool reassociateFpReductions);

} // namespace mlir::iree_compiler::hexagon::codegen

#endif // ROOF_HEXAGON_CONVERSION_HEXAGONCONVERTTOLLVM_H_

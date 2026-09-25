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

// Final conversion in one transaction, including HexagonMem, DMA, HexKL, HAL
// ABI, and standard-to-LLVM patterns.
std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createHexagonConvertToLLVMPass(bool reassociateFpReductions);

} // namespace mlir::iree_compiler::hexagon::codegen

#endif // ROOF_HEXAGON_CONVERSION_HEXAGONCONVERTTOLLVM_H_

// Copyright 2026 RooflineAI GmbH
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HEXAGON_PLUGINS_CODEGEN_CONVERSION_HEXAGONABITOLLVM_H_
#define IREE_HEXAGON_PLUGINS_CODEGEN_CONVERSION_HEXAGONABITOLLVM_H_

#include "hexagon/CodeGen/Conversion/HexagonDispatchABI.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/IR/PatternMatch.h"

namespace mlir::iree_compiler::hexagon::codegen {

// Populates conversion patterns for the IREE local executable entry point and
// hal.interface operations.
void populateHexagonABIToLLVMConversionPatterns(
    HexagonDispatchABI &abi, LLVMTypeConverter &typeConverter,
    RewritePatternSet &patterns);

// Populates the local executable dispatch instrumentation conversions.
void populateHexagonInstrumentationToLLVMConversionPatterns(
    HexagonDispatchABI &abi, LLVMTypeConverter &typeConverter,
    RewritePatternSet &patterns);

} // namespace mlir::iree_compiler::hexagon::codegen

#endif // IREE_HEXAGON_PLUGINS_CODEGEN_CONVERSION_HEXAGONABITOLLVM_H_

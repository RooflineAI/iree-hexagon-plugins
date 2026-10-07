// Copyright 2026 RooflineAI GmbH
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HEXAGON_PLUGINS_CODEGEN_CONVERSION_HEXAGONHMXTOLLVM_H_
#define IREE_HEXAGON_PLUGINS_CODEGEN_CONVERSION_HEXAGONHMXTOLLVM_H_

#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/IR/PatternMatch.h"

namespace mlir::iree_compiler::hexagon::codegen {

// Populates the conversion of bufferized HMX operations to calls into the
// native DSP HMX kernels, and erases the HMX accumulator type. The buffers are
// expected to satisfy the kernel ABI checked by the
// `iree-hexagon-verify-hmx-runtime-abi` pass.
void populateHexagonHmxToLLVMConversionPatterns(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns);

} // namespace mlir::iree_compiler::hexagon::codegen

#endif // IREE_HEXAGON_PLUGINS_CODEGEN_CONVERSION_HEXAGONHMXTOLLVM_H_

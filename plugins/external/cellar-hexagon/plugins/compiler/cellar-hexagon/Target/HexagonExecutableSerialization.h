// Copyright 2025 RooflineAI GmbH
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef ROOF_HEXAGON_HAL_HEXAGONEXECUTABLESERIALIZATION_H_
#define ROOF_HEXAGON_HAL_HEXAGONEXECUTABLESERIALIZATION_H_

#include "cellar-hexagon/Target/HexagonOptions.h"

#include "iree/compiler/Dialect/HAL/Target/TargetBackend.h"

namespace mlir::iree_compiler::cellar_hexagon::target {

mlir::LogicalResult serializeHexagonExecutable(
    const HexagonOptions &options,
    const mlir::iree_compiler::IREE::HAL::TargetBackend::SerializationOptions
        &serializationOptions,
    mlir::iree_compiler::IREE::HAL::ExecutableVariantOp variantOp,
    mlir::OpBuilder &executableBuilder);

} // namespace mlir::iree_compiler::cellar_hexagon::target

#endif // ROOF_HEXAGON_HAL_HEXAGONEXECUTABLESERIALIZATION_H_

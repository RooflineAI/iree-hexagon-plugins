// Copyright 2025 RooflineAI GmbH
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef ROOF_HEXAGON_HAL_HEXAGONLLVMTARGET_H_
#define ROOF_HEXAGON_HAL_HEXAGONLLVMTARGET_H_

#include "hexagon/Target/HexagonOptions.h"

#include "compiler/plugins/target/LLVMCPU/LLVMTargetOptions.h"

namespace mlir::iree_compiler::hexagon::target {

void initializeHexagonTarget();

mlir::iree_compiler::IREE::HAL::LLVMTarget
createLLVMTargetForHexagon(const HexagonOptions &options);

} // namespace mlir::iree_compiler::hexagon::target

#endif // ROOF_HEXAGON_HAL_HEXAGONLLVMTARGET_H_

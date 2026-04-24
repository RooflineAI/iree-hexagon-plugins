// Copyright 2025 RooflineAI GmbH
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef ROOF_HEXAGON_HAL_LINKING_HEXAGONLINKERTOOL_H_
#define ROOF_HEXAGON_HAL_LINKING_HEXAGONLINKERTOOL_H_

#include "compiler/plugins/target/LLVMCPU/LinkerTool.h"

namespace mlir::iree_compiler::cellar_hexagon::target::linking {

std::unique_ptr<mlir::iree_compiler::IREE::HAL::LinkerTool>
createHexagonLinkerTool(
    const llvm::Triple &targetTriple,
    mlir::iree_compiler::IREE::HAL::LLVMTargetOptions &targetOptions,
    bool allowNativeUndefinedSymbols = false);

} // namespace mlir::iree_compiler::cellar_hexagon::target::linking

#endif // ROOF_HEXAGON_HAL_LINKING_HEXAGONLINKERTOOL_H_

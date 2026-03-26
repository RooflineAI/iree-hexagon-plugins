// Copyright 2025 RooflineAI GmbH
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "compiler/plugins/target/LLVMCPU/LinkerTool.h"

namespace mlir::iree_compiler::IREE::HAL {

std::unique_ptr<LinkerTool>
createHexagonLinkerTool(const llvm::Triple &targetTriple,
                        LLVMTargetOptions &targetOptions);

} // namespace mlir::iree_compiler::IREE::HAL

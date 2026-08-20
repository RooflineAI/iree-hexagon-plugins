// Copyright 2025 RooflineAI GmbH
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "hexagon/Target/HexagonOptions.h"

#include "llvm/Support/CommandLine.h"

namespace mlir::iree_compiler::hexagon::target {

void HexagonOptions::bindOptions(mlir::iree_compiler::OptionsBinder &binder) {
  static llvm::cl::OptionCategory category("Hexagon HAL Target");

  binder.opt<std::string>(
      "iree-hexagon-v", version, llvm::cl::cat(category),
      llvm::cl::desc("Hexagon ISA version to target (e.g. 68, 69, 73, 79)."));

  // This is just passed raw to the LLVM backend for now
  binder.opt<std::string>(
      "iree-hexagon-features", features, llvm::cl::cat(category),
      llvm::cl::desc(
          "Hexagon features supported to be passed to the llvm backend (e.g. "
          "+hvxv79,+hvx-length128b). Use llc to determine other options."));

  binder.opt<std::string>(
      "iree-hexagon-linker-path", linker, llvm::cl::cat(category),
      llvm::cl::desc("Hexagon linker tool path to use during serialization. "
                     "Currently supported linkers are lld and hexagon-clang "
                     "(available in Hexagon's SDK)"));
}

} // namespace mlir::iree_compiler::hexagon::target

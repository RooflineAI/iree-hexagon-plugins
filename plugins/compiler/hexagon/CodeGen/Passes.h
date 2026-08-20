// Copyright 2025 RooflineAI GmbH
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// This file is the public facade for Hexagon codegen passes and pipelines.

#ifndef ROOF_HEXAGON_CODEGEN_PASSES_H_
#define ROOF_HEXAGON_CODEGEN_PASSES_H_

#include <string>

#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"

namespace mlir::iree_compiler::hexagon::codegen {

// Similar to the LLVMCPUPipelineOptions struct, but simplified for Hexagon
struct HexagonPipelineOptions {
  bool useConfiguredVectorSizes = true;
  bool enablePeeling = false;
  bool enableVectorMasking = true;
};

struct HexagonVectorLoweringPassOptions {
  std::string splitVectorTransfersTo = "";
};

// Registers the passes reused from the LLVMCPU backend so they can be
// referenced by name (e.g. from custom pipelines or debugging utilities).
void registerHexagonPasses();

// Registers Hexagon pass pipelines (configuration, translation, linking).
void registerHexagonCodeGenPasses();

#define GEN_PASS_DECL
#include "hexagon/CodeGen/Passes.h.inc" // IWYU pragma: keep

} // namespace mlir::iree_compiler::hexagon::codegen

#endif // ROOF_HEXAGON_CODEGEN_PASSES_H_

// Copyright 2025 RooflineAI GmbH
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// This file is the public facade for Hexagon codegen passes and pipelines.

#ifndef ROOF_HEXAGON_CODEGEN_PASSES_H_
#define ROOF_HEXAGON_CODEGEN_PASSES_H_

#include <string>

#include "iree/compiler/Codegen/Dialect/CPU/IR/IREECPUTypes.h"
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

// Registers Hexagon-owned passes for custom pipelines and debugging utilities.
void registerHexagonPasses();

// Registers Hexagon pass pipelines (configuration, translation, linking).
void registerHexagonCodeGenPasses();

#define GEN_PASS_DECL
#include "hexagon/CodeGen/Passes.h.inc" // IWYU pragma: keep

std::unique_ptr<InterfacePass<mlir::FunctionOpInterface>>
createHexagonSplitReductionPass(bool enableReassociateFpReductions);

// CPU tiling levels are retained until Hexagon configuration migration.
std::unique_ptr<InterfacePass<mlir::FunctionOpInterface>>
createHexagonTilePass(IREE::CPU::TilingLevel tilingLevel, bool skipRootOp);

std::unique_ptr<InterfacePass<mlir::FunctionOpInterface>>
createHexagonTileAndFuseProducerConsumerPass(
    IREE::CPU::TilingLevel tilingLevel);

std::unique_ptr<InterfacePass<mlir::FunctionOpInterface>>
createHexagonTileRootAndFuseInputOperandsPass(
    IREE::CPU::TilingLevel tilingLevel);

std::unique_ptr<InterfacePass<mlir::FunctionOpInterface>>
createHexagonTileLastOpAndFuseProducerConsumerPass(
    IREE::CPU::TilingLevel tilingLevel);

} // namespace mlir::iree_compiler::hexagon::codegen

#endif // ROOF_HEXAGON_CODEGEN_PASSES_H_

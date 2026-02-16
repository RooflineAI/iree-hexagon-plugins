// Copyright 2025 RooflineAI GmbH
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef ROOF_HEXAGON_CODEGEN_PASSES_H_
#define ROOF_HEXAGON_CODEGEN_PASSES_H_

#include <optional>
#include <string>

#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"

// This includes the LLVMCPU + builtin passes.
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenInterfaces.h"
#include "iree/compiler/Codegen/LLVMCPU/Passes.h"

namespace cellar::target::hexagon {

// Similar to the LLVMCPUPipelineOptions struct, but simplified for Hexagon
struct HexagonPipelineOptions {
  bool disableDistribution = false;
  bool decomposePackUnPackOps = true;
  bool useConfiguredVectorSizes = true;
  bool enablePeeling = false;
  bool enableVectorMasking = false;
};

struct HexagonVectorLoweringPassOptions {
  std::string splitVectorTransfersTo = "";
};

// Registers the passes reused from the LLVMCPU backend so they can be
// referenced by name (e.g. from custom pipelines or debugging utilities).
void registerHexagonPasses();

// Registers Hexagon pass pipelines (configuration, translation, linking).
void registerHexagonCodeGenPasses();

// Populates passes needed for preprocessing before codegen lowerings.
void buildHexagonConfigurationPassPipeline(mlir::OpPassManager &passManager);

// Populates passes needed to lower high level ops to LLVM IR for Hexagon.
// The options are currently forwarded directly to the LLVMCPU pipeline and can
// be pruned or specialized as needed.
void buildHexagonTranslationPassPipeline(
    mlir::OpPassManager &passManager,
    const HexagonPipelineOptions &pipelineOptions = HexagonPipelineOptions{});

// Populates passes needed to link HAL executables across Hexagon targets.
void buildHexagonLinkingPassPipeline(
    mlir::OpPassManager &passManager,
    std::optional<std::string> target = std::nullopt);

//------------------------------------------------------------------------------
// Pipeline builders (used by HexagonLowerExecutableTargetPass)
//------------------------------------------------------------------------------

void addHexagonDefaultPassPipeline(mlir::OpPassManager &funcPassManager,
                                   const HexagonPipelineOptions &pipelineOpt);

void addHexagonBufferOpsTileAndVectorizePipeline(
    mlir::OpPassManager &funcPassManager,
    const HexagonPipelineOptions &pipelineOpt);

void addHexagonMultiTilingExpertPassPipeline(
    mlir::OpPassManager &funcPassManager,
    mlir::iree_compiler::IREE::Codegen::LoweringConfigAttrInterface
        loweringConfig,
    const HexagonPipelineOptions &pipelineOpt);

void addHexagonConvTileAndDecomposeExpertPassPipeline(
    mlir::OpPassManager &funcPassManager,
    const HexagonPipelineOptions &pipelineOpt);

void addHexagonMmt4dTilingExpertPassPipeline(
    mlir::OpPassManager &funcPassManager,
    const HexagonPipelineOptions &pipelineOpt);

void addHexagonDataTilingPipeline(mlir::OpPassManager &funcPassManager,
                                  const HexagonPipelineOptions &pipelineOpt);

void addHexagonLinalgExtTileAndVectorizePipeline(
    mlir::OpPassManager &funcPassManager,
    const HexagonPipelineOptions &pipelineOpt);

//------------------------------------------------------------------------------
// TableGen
//------------------------------------------------------------------------------

#define GEN_PASS_DECL
#include "target/CellarHexagon/CodeGen/Passes.h.inc" // IWYU pragma: keep

} // namespace cellar::target::hexagon

#endif // ROOF_HEXAGON_CODEGEN_PASSES_H_

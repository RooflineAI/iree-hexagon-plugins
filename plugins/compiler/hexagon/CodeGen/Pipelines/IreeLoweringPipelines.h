// Copyright 2025 RooflineAI GmbH
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// This header declares the hexagon version of the expert pipelines from IREE's
// LLVMCPU plugin. They are selected by the HexagonLowerExecutableTargetPass.

#ifndef ROOF_HEXAGON_CODEGEN_IREELOWERINGPIPELINES_H_
#define ROOF_HEXAGON_CODEGEN_IREELOWERINGPIPELINES_H_

#include "hexagon/CodeGen/Passes.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenInterfaces.h"

namespace mlir::iree_compiler::hexagon::codegen {

bool isHexagonFailOnOutOfBoundsStackAllocationEnabled();

/// Entry point for the iree-based pipeline
void buildHexagonIreeTranslationRoute(
    mlir::OpPassManager &variantPassManager,
    const HexagonPipelineOptions &pipelineOpt);

//===---------------------------------------------------------------------===//
// Expert pipelines (copied from LLVMCPU)
//===---------------------------------------------------------------------===//
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
// Helper functions
//------------------------------------------------------------------------------
// These helper functions are made public for now for experimental purposes
void addHexagonVariantFinalizationPasses(
    mlir::OpPassManager &variantPassManager);

void addHexagonLowerToLLVMPasses(mlir::OpPassManager &modulePassManager);

void addHexagonTileAndDistributePasses(
    mlir::OpPassManager &funcPassManager,
    const HexagonPipelineOptions &pipelineOpt);

} // namespace mlir::iree_compiler::hexagon::codegen

#endif // ROOF_HEXAGON_CODEGEN_IREELOWERINGPIPELINES_H_

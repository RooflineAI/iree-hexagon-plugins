// Copyright 2026 RooflineAI GmbH
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef ROOF_HEXAGON_CODEGEN_PIPELINES_HEXAGONLOWEREXECUTABLETARGET_H_
#define ROOF_HEXAGON_CODEGEN_PIPELINES_HEXAGONLOWEREXECUTABLETARGET_H_

#include "hexagon/CodeGen/Passes.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenInterfaces.h"
#include "iree/compiler/Codegen/Utils/CodegenPipelineOptions.h"

namespace mlir::iree_compiler::hexagon::codegen {

struct HexagonCodegenPipelineOptions final
    : CodegenPipelineOptionsBase<HexagonCodegenPipelineOptions> {
  HexagonCodegenPipelineOptions(
      HexagonPipelineOptions options,
      IREE::Codegen::LoweringConfigAttrInterface loweringConfig)
      : options(options), loweringConfig(loweringConfig) {}

  HexagonPipelineOptions options;
  IREE::Codegen::LoweringConfigAttrInterface loweringConfig;
};

/// Builds the concrete pass pipeline selected by a Hexagon PipelineAttr.
///
/// registerHexagonCodeGenPasses registers this function as the callback used by
/// IREE::Hexagon::PipelineAttr::buildPipeline. The callback keeps the Hexagon
/// dialect IR library independent of the higher-level CodeGen pipeline library:
/// the attribute implements PipelineAttrInterface in the IR layer, while this
/// function can depend on and dispatch to the concrete Hexagon pipeline
/// builders.
LogicalResult buildHexagonPipeline(Attribute pipelineAttr,
                                   OpPassManager &passManager,
                                   const CodegenPipelineOptions *options);

} // namespace mlir::iree_compiler::hexagon::codegen

#endif // ROOF_HEXAGON_CODEGEN_PIPELINES_HEXAGONLOWEREXECUTABLETARGET_H_

// Copyright 2025 RooflineAI GmbH
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// This pass is in charge of invoking the appropriate Hexagon-specific pass
// pipelines based on the selected lowering strategy.
// This file was created in the image of the equivalent LLVMCPU file.

#include "hexagon/CodeGen/Pipelines/HexagonLowerExecutableTarget.h"
#include "hexagon/CodeGen/IR/HexagonAttrs.h"
#include "hexagon/CodeGen/IR/HexagonDialect.h"
#include "hexagon/CodeGen/Passes.h"
#include "hexagon/CodeGen/Pipelines/LoweringPipelines.h"

#include "hexagon/Dialect/HexKL/IR/HexKLDialect.h"
#include "hexagon/Dialect/HexagonMem/IR/HexagonMemDialect.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "iree/compiler/Codegen/Utils/CPUUtils.h"
#include "iree/compiler/Codegen/Utils/Utils.h"
#include "iree/compiler/Dialect/HAL/IR/HALDialect.h"
#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtDialect.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/PDL/IR/PDL.h"
#include "mlir/Dialect/PDLInterp/IR/PDLInterp.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Interfaces/TilingInterface.h"
#include "llvm/ADT/STLExtras.h"

namespace mlir::iree_compiler::hexagon::codegen {
namespace IREE = mlir::iree_compiler::IREE;

#define GEN_PASS_DEF_HEXAGONLOWEREXECUTABLETARGETPASS
#include "hexagon/CodeGen/Passes.h.inc"

namespace {
class HexagonLowerExecutableTargetPass
    : public impl::HexagonLowerExecutableTargetPassBase<
          HexagonLowerExecutableTargetPass> {
public:
  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    // clang-format off
    registry.insert<IREE::HAL::HALDialect,
                    IREE::Hexagon::IREEHexagonDialect,
                    IREE::LinalgExt::IREELinalgExtDialect,
                    bufferization::BufferizationDialect,
                    mlir::hexagonmem::HexagonMemDialect,
                    mlir::hexkl::HexKLDialect,
                    linalg::LinalgDialect,
                    LLVM::LLVMDialect,
                    pdl::PDLDialect,
                    pdl_interp::PDLInterpDialect,
                    scf::SCFDialect,
                    tensor::TensorDialect,
                    transform::TransformDialect,
                    vector::VectorDialect>();
    // clang-format on
  }
  void runOnOperation() override;
};
} // namespace

static IREE::Codegen::LoweringConfigAttrInterface
getRootLoweringConfig(mlir::FunctionOpInterface funcOp) {
  auto rootOp = getRootOperation(getComputeOps(funcOp));
  if (failed(rootOp) || !rootOp.value()) {
    return nullptr;
  }
  return getLoweringConfig(rootOp.value());
}

LogicalResult buildHexagonPipeline(Attribute pipelineAttr,
                                   OpPassManager &passManager,
                                   const CodegenPipelineOptions *options) {
  auto hexagonPipeline = cast<IREE::Hexagon::PipelineAttr>(pipelineAttr);
  const auto *hexagonOptions =
      dyn_cast_if_present<HexagonCodegenPipelineOptions>(options);
  HexagonPipelineOptions pipelineOptions;
  IREE::Codegen::LoweringConfigAttrInterface loweringConfig;
  if (hexagonOptions) {
    pipelineOptions = hexagonOptions->options;
    loweringConfig = hexagonOptions->loweringConfig;
  }

  switch (hexagonPipeline.getValue()) {
  case IREE::Hexagon::LoweringPipeline::Default:
    addHexagonDefaultPassPipeline(passManager, pipelineOptions);
    return success();
  case IREE::Hexagon::LoweringPipeline::BufferOpsTileAndVectorize:
    addHexagonBufferOpsTileAndVectorizePipeline(passManager, pipelineOptions);
    return success();
  case IREE::Hexagon::LoweringPipeline::MultiTilingExpert:
    if (!loweringConfig)
      return failure();
    addHexagonMultiTilingExpertPassPipeline(passManager, loweringConfig,
                                            pipelineOptions);
    return success();
  case IREE::Hexagon::LoweringPipeline::ConvTileAndDecomposeExpert:
    addHexagonConvTileAndDecomposeExpertPassPipeline(passManager,
                                                     pipelineOptions);
    return success();
  case IREE::Hexagon::LoweringPipeline::HmxMatmulExpert:
    if (!loweringConfig)
      return failure();
    addHexagonHmxMatmulExpertPassPipeline(passManager, pipelineOptions);
    return success();
  case IREE::Hexagon::LoweringPipeline::DataTiling:
    addHexagonDataTilingPipeline(passManager, pipelineOptions);
    return success();
  case IREE::Hexagon::LoweringPipeline::LinalgExtTileAndVectorize:
    addHexagonLinalgExtTileAndVectorizePipeline(passManager, pipelineOptions);
    return success();
  }
  return failure();
}

void HexagonLowerExecutableTargetPass::runOnOperation() {
  mlir::FunctionOpInterface funcOp = getOperation();

  auto targetAttr = IREE::HAL::ExecutableTargetAttr::lookup(funcOp);
  if (!targetAttr) {
    // Do nothing without a target.
    return;
  }

  auto translationInfo = getTranslationInfo(funcOp);
  if (!translationInfo) {
    // Strategy selection has not run (or user did not specify anything).
    // Keep this pass a no-op in that case.
    return;
  }

  HexagonPipelineOptions pipelineOpts;
  pipelineOpts.enablePeeling = isOptEnabled(funcOp, getEnableLoopPeelingStr());

  mlir::OpPassManager passManager(mlir::func::FuncOp::getOperationName());
  mlir::Attribute pipelineAttr = translationInfo.getPassPipeline();
  if (mlir::isa<IREE::Codegen::NoPipelineAttr>(pipelineAttr)) {
    return;
  }
  auto hexagonPipeline =
      mlir::dyn_cast<IREE::Hexagon::PipelineAttr>(pipelineAttr);
  if (!hexagonPipeline) {
    funcOp.emitOpError("unsupported pipeline on Hexagon target");
    return signalPassFailure();
  }
  // Attribute verification checks the stage structure. Iteration rank depends
  // on the annotated operation and must be checked before scheduling tiling.
  WalkResult configCheck = funcOp.walk([](TilingInterface op) {
    auto config = getLoweringConfig<IREE::Hexagon::LoweringConfigAttr>(op);
    if (!config)
      return WalkResult::advance();
    size_t rank = op.getLoopIteratorTypes().size();
    for (int level : IREE::Hexagon::getTilingLevelsAsInts()) {
      if (config.hasTilingLevel(level) &&
          config.getStaticTilingLevelSizes(level, op).size() != rank) {
        op.emitOpError(
            "Hexagon lowering config rank must match iteration rank ")
            << rank;
        return WalkResult::interrupt();
      }
    }
    return WalkResult::advance();
  });
  if (configCheck.wasInterrupted())
    return signalPassFailure();
  HexagonCodegenPipelineOptions options(pipelineOpts,
                                        getRootLoweringConfig(funcOp));
  if (failed(hexagonPipeline.buildPipeline(passManager, &options))) {
    funcOp.emitOpError("failed to build Hexagon pass pipeline");
    return signalPassFailure();
  }

  if (failed(runPipeline(passManager, funcOp))) {
    return signalPassFailure();
  }
}

} // namespace mlir::iree_compiler::hexagon::codegen

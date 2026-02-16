// Copyright 2025 RooflineAI GmbH
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "Passes.h"

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
#include "llvm/ADT/STLExtras.h"

using namespace mlir;
using namespace mlir::iree_compiler;

namespace cellar::target::hexagon {

#define GEN_PASS_DEF_HEXAGONLOWEREXECUTABLETARGETPASS
#include "target/CellarHexagon/CodeGen/Passes.h.inc"

namespace {
class HexagonLowerExecutableTargetPass
    : public impl::HexagonLowerExecutableTargetPassBase<
          HexagonLowerExecutableTargetPass> {
public:
  void getDependentDialects(DialectRegistry &registry) const override {
    // clang-format off
    registry.insert<IREE::HAL::HALDialect,
                    IREE::LinalgExt::IREELinalgExtDialect,
                    bufferization::BufferizationDialect,
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
getRootLoweringConfig(FunctionOpInterface funcOp) {
  SmallVector<Operation *> computeOps = getComputeOps(funcOp);
  for (Operation *op : computeOps) {
    IREE::Codegen::LoweringConfigAttrInterface loweringConfig =
        getLoweringConfig(op);
    if (loweringConfig && loweringConfig.hasWorkgroupTilingLevel()) {
      return loweringConfig;
    }
  }
  return nullptr;
}

void HexagonLowerExecutableTargetPass::runOnOperation() {
  FunctionOpInterface funcOp = getOperation();

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
  IREE::Codegen::LoweringConfigAttrInterface loweringConfig =
      getRootLoweringConfig(funcOp);
  if (loweringConfig &&
      llvm::all_of(loweringConfig.getWorkgroupTileSizes(),
                   [](int64_t tileSize) { return tileSize == 0; })) {
    pipelineOpts.disableDistribution = true;
  }
  pipelineOpts.decomposePackUnPackOps =
      isOptEnabled(funcOp, getEnableDecompositionStr());
  pipelineOpts.enablePeeling = isOptEnabled(funcOp, getEnableLoopPeelingStr());

  OpPassManager passManager(func::FuncOp::getOperationName());
  switch (translationInfo.getDispatchLoweringPassPipeline()) {
  case IREE::Codegen::DispatchLoweringPassPipeline::None:
    return;
  case IREE::Codegen::DispatchLoweringPassPipeline::CPUDefault:
    addHexagonDefaultPassPipeline(passManager, pipelineOpts);
    break;
  case IREE::Codegen::DispatchLoweringPassPipeline::
      CPUBufferOpsTileAndVectorize:
    addHexagonBufferOpsTileAndVectorizePipeline(passManager, pipelineOpts);
    break;
  case IREE::Codegen::DispatchLoweringPassPipeline::CPUDoubleTilingExpert:
    if (!loweringConfig) {
      funcOp.emitOpError("expected a valid lowering_config for the selected "
                         "CPUDoubleTilingExpert pipeline");
      return signalPassFailure();
    }
    addHexagonMultiTilingExpertPassPipeline(passManager, loweringConfig,
                                            pipelineOpts);
    break;
  case IREE::Codegen::DispatchLoweringPassPipeline::
      CPUConvTileAndDecomposeExpert:
    addHexagonConvTileAndDecomposeExpertPassPipeline(passManager, pipelineOpts);
    break;
  case IREE::Codegen::DispatchLoweringPassPipeline::Mmt4dTilingExpert:
    addHexagonMmt4dTilingExpertPassPipeline(passManager, pipelineOpts);
    break;
  case IREE::Codegen::DispatchLoweringPassPipeline::CPUDataTiling:
    addHexagonDataTilingPipeline(passManager, pipelineOpts);
    break;
  case IREE::Codegen::DispatchLoweringPassPipeline::
      CPULinalgExtTileAndVectorize:
    addHexagonLinalgExtTileAndVectorizePipeline(passManager, pipelineOpts);
    break;
  default:
    funcOp.emitOpError("unsupported pipeline on Hexagon target");
    return signalPassFailure();
  }

  if (failed(runPipeline(passManager, funcOp))) {
    return signalPassFailure();
  }
}

} // namespace cellar::target::hexagon

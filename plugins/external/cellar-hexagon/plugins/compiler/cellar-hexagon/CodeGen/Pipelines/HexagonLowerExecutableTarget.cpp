// Copyright 2025 RooflineAI GmbH
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// This pass is in charge of invoking the appropriate Hexagon-specific pass
// pipelines based on the selected lowering strategy.
// This file was created in the image of the equivalent LLVMCPU file.

#include "cellar-hexagon/CodeGen/Passes.h"
#include "cellar-hexagon/CodeGen/Pipelines/IreeLoweringPipelines.h"

#include "hexagon/Dialect/HexKL/IR/HexKLDialect.h"
#include "hexagon/Dialect/HexagonMem/IR/HexagonMemDialect.h"
#include "iree/compiler/Codegen/Dialect/CPU/IR/IREECPUTypes.h"
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

namespace mlir::iree_compiler::cellar_hexagon::codegen {
namespace IREE = mlir::iree_compiler::IREE;

#define GEN_PASS_DEF_HEXAGONLOWEREXECUTABLETARGETPASS
#include "cellar-hexagon/CodeGen/Passes.h.inc"

namespace {
class HexagonLowerExecutableTargetPass
    : public impl::HexagonLowerExecutableTargetPassBase<
          HexagonLowerExecutableTargetPass> {
public:
  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    // clang-format off
    registry.insert<IREE::HAL::HALDialect,
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
  // The pipeline is carried on translation_info as a pipeline attribute. A
  // NoPipelineAttr means strategy selection left nothing to do. Hexagon reuses
  // the CPU pipeline names (set in KernelDispatch) but runs its own passes, so
  // decode the IREE::CPU::PipelineAttr and switch on its LoweringPipeline
  // value.
  // TODO: we are currently reusing attributes from the LLVMCPU lowering
  // pipeline to mark what pipeline should be used for Hexagon. This should be
  // changed to use custom attributes for Hexagon.
  mlir::Attribute pipelineAttr = translationInfo.getPassPipeline();
  if (mlir::isa<IREE::Codegen::NoPipelineAttr>(pipelineAttr)) {
    return;
  }
  auto cpuPipeline = mlir::dyn_cast<IREE::CPU::PipelineAttr>(pipelineAttr);
  if (!cpuPipeline) {
    funcOp.emitOpError("unsupported pipeline on Hexagon target");
    return signalPassFailure();
  }
  switch (cpuPipeline.getValue()) {
  case IREE::CPU::LoweringPipeline::Default:
    addHexagonDefaultPassPipeline(passManager, pipelineOpts);
    break;

  case IREE::CPU::LoweringPipeline::BufferOpsTileAndVectorize:
    addHexagonBufferOpsTileAndVectorizePipeline(passManager, pipelineOpts);
    break;

  case IREE::CPU::LoweringPipeline::DoubleTilingExpert:
    if (auto loweringConfig = getRootLoweringConfig(funcOp)) {
      addHexagonMultiTilingExpertPassPipeline(passManager, loweringConfig,
                                              pipelineOpts);
      break;
    }
    funcOp.emitWarning()
        << "selected DoubleTilingExpert pipeline requires a root "
           "lowering_config, but no compute root with lowering_config was "
           "found";
    return signalPassFailure();

  // For now I am just assuming that we use im2col and forget about special
  // pipelines for convolutions
  case IREE::CPU::LoweringPipeline::ConvTileAndDecomposeExpert:
    addHexagonConvTileAndDecomposeExpertPassPipeline(passManager, pipelineOpts);
    break;

  // This lowering pipeline
  // takes special care for linalg.matmul and batch_matmul when data tiling is
  // enabled. Since we are disabling iree's data tiling and managing through
  // hexagon-mlir's passes that take advantage of the VTCM and vector unit,
  // this pipeline is useless for hexagon.
  case IREE::CPU::LoweringPipeline::Mmt4dTilingExpert:
    addHexagonMmt4dTilingExpertPassPipeline(passManager, pipelineOpts);
    break;

  // This pipeline is used when only data layout transformations are needed
  // but no reduction happens (only linalg.pack/unpack ops).
  case IREE::CPU::LoweringPipeline::DataTiling:
    addHexagonDataTilingPipeline(passManager, pipelineOpts);
    break;

  // This lowering pipeline is supposed to be used when linalgExt ops are
  // present (attention, FFT, Winograd).
  case IREE::CPU::LoweringPipeline::LinalgExtTileAndVectorize:
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

} // namespace mlir::iree_compiler::cellar_hexagon::codegen

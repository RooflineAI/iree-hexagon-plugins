// Copyright 2026 RooflineAI GmbH
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "KernelDispatchEncoding.h"
#include "KernelDispatch.h"

#include "cellar-hexagon/CodeGen/IR/HexagonAttrs.h"
#include "cellar-hexagon/CodeGen/Pipelines/TranslationPipeline.h"

#include "iree/compiler/Codegen/Dialect/CPU/IR/IREECPUTypes.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenInterfaces.h"
#include "iree/compiler/Codegen/Utils/CPUUtils.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "llvm/ADT/STLExtras.h"

namespace mlir::iree_compiler::cellar_hexagon::codegen {
namespace {

using mlir::iree_compiler::getLoweringConfig;
using mlir::iree_compiler::setLoweringConfig;
using mlir::iree_compiler::setOpConfigAndEntryPointFnTranslation;
using mlir::iree_compiler::IREE::CPU::LoweringConfigAttr;
using mlir::iree_compiler::IREE::CPU::TilingLevel;
using mlir::iree_compiler::IREE::Hexagon::VTCMTilingConfigAttr;

static void encodeVTCMConfig(Operation *op,
                             const std::optional<VTCMTilingPlan> &vtcmTiling) {
  if (!isHexagonVTCMTilingEnabled() || !vtcmTiling ||
      !mlir::isa<mlir::linalg::LinalgOp>(op)) {
    return;
  }
  const VTCMTilingPlan &vtcmPlan = *vtcmTiling;
  op->setAttr(kHexagonVTCMTilingConfigAttrName,
              VTCMTilingConfigAttr::get(op->getContext(), vtcmPlan.tileSizes));
}

static LoweringConfigAttr encodeTilingConfig(MLIRContext *ctx,
                                             const TileLevelsPlan &tileLevels,
                                             bool preserveZeroDistribution) {
  llvm::SmallVector<mlir::NamedAttribute> items;
  auto appendLoweringConfigLevelAttr =
      [&](TilingLevel level, llvm::ArrayRef<int64_t> tileSizes,
          llvm::ArrayRef<bool> scalableFlags = {}) {
        bool allZero =
            llvm::all_of(tileSizes, [](int64_t size) { return size == 0; });
        if (tileSizes.empty() ||
            (allZero && !(preserveZeroDistribution &&
                          level == TilingLevel::DistributionTiles))) {
          return;
        }
        items.emplace_back(
            mlir::iree_compiler::IREE::CPU::getTilingLevelName(level),
            LoweringConfigAttr::getTilingLevelAttr(ctx, tileSizes,
                                                   scalableFlags));
      };

  appendLoweringConfigLevelAttr(TilingLevel::DistributionTiles,
                                tileLevels.distribution);
  appendLoweringConfigLevelAttr(TilingLevel::CacheParallelTiles,
                                tileLevels.cacheParallel);
  appendLoweringConfigLevelAttr(TilingLevel::VectorCommonParallelTiles,
                                tileLevels.vectorCommonParallel);
  // This is needed to avoid crashes in some LLVMCPU pipeline passes that access
  // values without checking if they exist
  llvm::SmallVector<bool> reductionScalableFlags(
      tileLevels.vectorReduction.size(), false);
  appendLoweringConfigLevelAttr(TilingLevel::VectorReductionTiles,
                                tileLevels.vectorReduction,
                                reductionScalableFlags);
  appendLoweringConfigLevelAttr(TilingLevel::VectorInnerParallelTiles,
                                tileLevels.vectorInnerParallel);

  return LoweringConfigAttr::get(ctx, items);
}

// Builds a translation_info tagging the function with a CPU lowering pipeline.
// Hexagon reuses the CPU pipeline names but runs its own passes in
// HexagonLowerExecutableTarget; mirrors LLVMCPU's getCPUTranslationInfo.
static IREE::Codegen::TranslationInfoAttr
getHexagonTranslationInfo(MLIRContext *ctx,
                          IREE::CPU::LoweringPipeline pipeline,
                          DictionaryAttr configuration = {}) {
  return IREE::Codegen::TranslationInfoAttr::get(
      ctx, IREE::CPU::PipelineAttr::get(ctx, pipeline), SymbolRefAttr(),
      /*workgroupSize=*/{}, /*subgroupSize=*/0, configuration);
}

} // namespace

LogicalResult applyRootLoweringPlan(FunctionOpInterface entryPointFn,
                                    Operation *rootOperation,
                                    const RootLoweringPlan &rootPlan) {
  assert(rootPlan.opPlan && "expected root plan to have an op plan");
  // LLVMCPUTileAndFuseProducerConsumerPass anchors on the unique root op that
  // has both the requested tiling level and a distribution tiling level.
  // Hexagon disables workgroup tiling, but the null distribution level still
  // acts as an anchor marker.
  bool preserveZeroDistribution =
      rootPlan.pipeline == mlir::iree_compiler::IREE::CPU::LoweringPipeline::
                               DoubleTilingExpert &&
      !rootPlan.opPlan->tileLevels.distribution.empty();
  auto loweringConfig =
      encodeTilingConfig(rootOperation->getContext(),
                         rootPlan.opPlan->tileLevels, preserveZeroDistribution);

  DictionaryAttr pipelineConfig;
  if (rootPlan.enableLoopPeeling) {
    MLIRContext *ctx = rootOperation->getContext();
    StringAttr attrName = getEnableLoopPeelingAttrName(ctx);
    pipelineConfig = DictionaryAttr::get(
        ctx, ArrayRef<NamedAttribute>{{attrName, UnitAttr::get(ctx)}});
  }
  if (failed(setOpConfigAndEntryPointFnTranslation(
          entryPointFn, rootOperation, loweringConfig,
          getHexagonTranslationInfo(rootOperation->getContext(),
                                    rootPlan.pipeline, pipelineConfig)))) {
    return failure();
  }

  encodeVTCMConfig(rootOperation, rootPlan.vtcmTiling);

  return success();
}

void applyOpLoweringPlan(Operation *op, const OpLoweringPlan &opPlan) {
  if (getLoweringConfig(op)) {
    return;
  }
  setLoweringConfig(op, encodeTilingConfig(op->getContext(), opPlan.tileLevels,
                                           /*preserveZeroDistribution=*/false));
}

} // namespace mlir::iree_compiler::cellar_hexagon::codegen

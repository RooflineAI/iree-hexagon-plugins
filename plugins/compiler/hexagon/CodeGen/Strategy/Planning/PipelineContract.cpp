// Copyright 2026 RooflineAI GmbH
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "PipelineContract.h"

#include "DispatchPlanTypes.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/Operation.h"
#include "llvm/Support/ErrorHandling.h"

namespace mlir::iree_compiler::hexagon::codegen::planning {

LoopTilingScope
PipelineContract::getLoopTilingScope(ComputeTileLevel level) const {
  switch (level) {
  case ComputeTileLevel::CommonParallel:
    return vectorCommonParallel;
  case ComputeTileLevel::Reduction:
    return vectorReduction;
  case ComputeTileLevel::InnerParallel:
    return vectorInnerParallel;
  }
  llvm_unreachable("unknown compute tile level");
}

FailureOr<PipelineContract>
getPipelineContract(IREE::CPU::LoweringPipeline pipeline) {
  using Pipeline = IREE::CPU::LoweringPipeline;
  PipelineContract contract;
  switch (pipeline) {
  case Pipeline::BufferOpsTileAndVectorize:
    contract.vectorCommonParallel = LoopTilingScope::EveryConfiguredOperation;
    contract.usesConfiguredVectorSizes = true;
    contract.supportsIndependentNonRootComputeTiles = true;
    contract.loopPeeling = LoopPeelingSupport::TranslationInfoControlled;
    return contract;
  case Pipeline::DoubleTilingExpert:
    contract.requiresUniqueRootAnchor = true;
    contract.cacheParallel = LoopTilingScope::Root;
    contract.vectorCommonParallel = LoopTilingScope::Root;
    contract.vectorReduction = LoopTilingScope::EveryConfiguredOperation;
    contract.vectorInnerParallel =
        LoopTilingScope::LastConfiguredOnEachSideOfRoot;
    contract.runsTileToVectorSize = true;
    contract.usesConfiguredVectorSizes = true;
    contract.supportsIndependentNonRootComputeTiles = true;
    contract.vtcmRequirement = VTCMRequirement::Optional;
    contract.cacheTilingWithVTCM = CacheTilingWithVTCM::Suppress;
    contract.loopPeeling = LoopPeelingSupport::TranslationInfoControlled;
    return contract;
  case Pipeline::Mmt4dTilingExpert:
    // Hexagon repurposes this CPU pipeline slot for HMX. HMX pack operations
    // zero-pad partial tiles and unpack clips the result to its logical bounds,
    // so this pipeline does not peel loops.
    contract.requiresUniqueRootAnchor = true;
    contract.cacheParallel = LoopTilingScope::EveryConfiguredOperation;
    contract.vectorCommonParallel = LoopTilingScope::Root;
    contract.vectorInnerParallel =
        LoopTilingScope::LastConfiguredOnEachSideOfRoot;
    contract.runsTileToVectorSize = true;
    contract.usesConfiguredVectorSizes = true;
    contract.supportsIndependentNonRootComputeTiles = true;
    contract.vtcmRequirement = VTCMRequirement::Required;
    return contract;
  case Pipeline::ConvTileAndDecomposeExpert:
    contract.requiresUniqueRootAnchor = true;
    contract.vectorCommonParallel = LoopTilingScope::Root;
    contract.vectorReduction = LoopTilingScope::Root;
    contract.usesConfiguredVectorSizes = true;
    contract.loopPeeling = LoopPeelingSupport::TranslationInfoControlled;
    return contract;
  case Pipeline::Default:
    contract.vectorCommonParallel =
        LoopTilingScope::LastConfiguredOnEachSideOfRoot;
    return contract;
  default:
    return failure();
  }
}

bool canRefineComputeTileDownstream(const PipelineContract &contract,
                                    Operation *op) {
  if (!contract.runsTileToVectorSize)
    return false;
  // LLVMCPUTileToVectorSizePass skips linalg.fill unconditionally, on the
  // grounds that a fill usually feeds a reduction consumer and tiling it
  // would only add a loop. A fill's configured shape therefore has to match
  // the extent it is fused into, because nothing will tile it down to a
  // smaller one.
  return !isa<linalg::FillOp>(op);
}

} // namespace mlir::iree_compiler::hexagon::codegen::planning

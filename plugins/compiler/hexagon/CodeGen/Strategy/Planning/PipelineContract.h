// Copyright 2026 RooflineAI GmbH
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef ROOF_HEXAGON_CODEGEN_PLANNING_PIPELINECONTRACT_H_
#define ROOF_HEXAGON_CODEGEN_PLANNING_PIPELINECONTRACT_H_

#include "iree/compiler/Codegen/Dialect/CPU/IR/IREECPUTypes.h"
#include "mlir/IR/Operation.h"

#include "mlir/Support/LogicalResult.h"

namespace mlir::iree_compiler::hexagon::codegen::planning {

enum class ComputeTileLevel;

/// Which operations have their tile value consumed as loop tiling across the
/// whole downstream pipeline. A level used by multiple passes records the
/// effective union of those consumers. This is distinct from contributing to a
/// per-op vector shape through LoweringConfigAttr::getVectorSizes().
enum class LoopTilingScope {
  /// No downstream pass consumes this level as loop tiling.
  Unused,
  /// A root-anchored tile-and-fuse pass consumes this level.
  Root,
  /// Every configured operation is consumed, possibly by multiple passes whose
  /// effective scopes together cover both the root and non-root operations.
  EveryConfiguredOperation,
  /// The downstream pass independently selects the last configured operation
  /// before and after the distribution root. Without a distribution root, it
  /// selects only the globally last configured operation.
  LastConfiguredOnEachSideOfRoot,
};

enum class VTCMRequirement {
  /// The pipeline cannot consume a VTCM plan.
  Unsupported,
  /// Attempt VTCM planning when enabled, but continue when it is unavailable.
  Optional,
  /// The pipeline is invalid unless it receives a VTCM plan.
  Required,
};

// TODO: This is currently needed in order for HMX tiling to have an additional
// level of tiling. This is yet again another place where introducing custom
// levels of tilings would be best, as documented in the README.md.
// Such a change would simplify this logic away.
enum class CacheTilingWithVTCM {
  /// Retain cache tiling after a VTCM plan is selected.
  Preserve,
  /// A selected VTCM plan supersedes ordinary cache tiling.
  Suppress,
};

enum class LoopPeelingSupport {
  /// The pipeline does not run loop peeling.
  Unsupported,
  /// The pipeline runs loop peeling when requested in translation info.
  TranslationInfoControlled,
  /// The pipeline runs loop peeling without a translation-info request.
  Unconditional,
};

/// Machine-readable subset of downstream pipeline semantics needed by plan
/// verification and encoding. This replaces the pass-consumption table that is
/// currently maintained only in comments.
struct PipelineContract {
  /// Returns the loop-tiling scope that consumes `level`.
  LoopTilingScope getLoopTilingScope(ComputeTileLevel level) const;

  /// Root-anchored LLVMCPU passes discover their root through the unique
  /// operation carrying a distribution level. Hexagon therefore preserves an
  /// all-zero distribution level on that operation as an anchor marker.
  bool requiresUniqueRootAnchor = false;
  LoopTilingScope cacheParallel = LoopTilingScope::Unused;
  LoopTilingScope vectorCommonParallel = LoopTilingScope::Unused;
  LoopTilingScope vectorReduction = LoopTilingScope::Unused;
  LoopTilingScope vectorInnerParallel = LoopTilingScope::Unused;
  // Runs LLVMCPUTileToVectorSizePass
  bool runsTileToVectorSize = false;
  // Configured vector sizes are consumed by GenericVectorization.
  bool usesConfiguredVectorSizes = false;
  bool supportsIndependentNonRootComputeTiles = false;
  VTCMRequirement vtcmRequirement = VTCMRequirement::Unsupported;
  CacheTilingWithVTCM cacheTilingWithVTCM = CacheTilingWithVTCM::Preserve;
  LoopPeelingSupport loopPeeling = LoopPeelingSupport::Unsupported;
};

/// Returns the contract for a pipeline routed through the Hexagon backend.
FailureOr<PipelineContract>
getPipelineContract(IREE::CPU::LoweringPipeline pipeline);

/// Whether a downstream pass can still tile `op` to a configured compute tile
/// smaller than the extent it is fused into. When this is false the operation's
/// tile must match that extent exactly: a smaller configured vector shape is
/// rejected by vectorization and a larger one is masked.
bool canRefineComputeTileDownstream(const PipelineContract &contract,
                                    Operation *op);

} // namespace mlir::iree_compiler::hexagon::codegen::planning

#endif // ROOF_HEXAGON_CODEGEN_PLANNING_PIPELINECONTRACT_H_

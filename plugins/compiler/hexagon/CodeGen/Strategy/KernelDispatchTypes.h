// Copyright 2026 RooflineAI GmbH
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef ROOF_HEXAGON_CODEGEN_KERNELDISPATCHTYPES_H_
#define ROOF_HEXAGON_CODEGEN_KERNELDISPATCHTYPES_H_

#include "iree/compiler/Codegen/Dialect/CPU/IR/IREECPUTypes.h"
#include "llvm/ADT/SmallVector.h"

#include <optional>

namespace mlir::iree_compiler::hexagon::codegen {

// This header defines structs that are meant to interface between
// kernelDispatch heuristic selection and encoding.

/// Carries the tile sizes chosen for each lowering-config tiling
/// level before they are encoded as `#iree_cpu.lowering_config`.
struct TileLevelsPlan {
  // These represent different levels of tiling, copied from the LLVMCPU
  // pipeline.

  // Workgroup distribution tiles (not necessarily threads for this backend! At
  // least not for now)
  llvm::SmallVector<int64_t> distribution;
  // Tiles for cache size
  llvm::SmallVector<int64_t> cacheParallel;
  // Tiles for parallel dimension shared across multiple ops
  llvm::SmallVector<int64_t> vectorCommonParallel;
  // Tiles for reduction dimensions
  llvm::SmallVector<int64_t> vectorReduction;
  // Tiles for op-specific parallel dimensions (not necessarily captured
  // in the root). Currently unused.
  llvm::SmallVector<int64_t> vectorInnerParallel;
};

/// Describes the lowering plan for a single compute op.
struct OpLoweringPlan {
  TileLevelsPlan tileLevels;
};

/// Carries Hexagon-specific VTCM staging information before it is encoded as
/// `#iree_hexagon.vtcm_tiling_config`. Right now, only contains tileSizes, but
/// This can potentially contain more information in the future (ex:
/// double buffering)
struct VTCMTilingPlan {
  llvm::SmallVector<int64_t> tileSizes;
};

/// Describes the selected lowering plan for the dispatch root operation.
///
/// `opPlan` is empty when Hexagon intentionally falls back to the default CPU
/// pipeline instead of assigning a Hexagon-owned lowering strategy.
struct RootLoweringPlan {
  std::optional<OpLoweringPlan> opPlan;
  std::optional<VTCMTilingPlan> vtcmTiling;
  mlir::iree_compiler::IREE::CPU::LoweringPipeline pipeline =
      mlir::iree_compiler::IREE::CPU::LoweringPipeline::Default;
  bool enableLoopPeeling = false;
};

/// Hardware characteristics
struct TargetInfo {
  int64_t targetThreadCount = 8;
  int64_t usableVectorRegisterCount = 24;
  // This value is only used as a fallback when no information is included in
  // the target configuration attribute.
  int64_t nativeVectorBytes = 128;
};

/// These values act as caps on the heuristics, and are complementary to the
/// logic of the heuristics themselves (they are not translated directly into
/// tiling descriptions)
struct HeuristicCaps {
  int64_t reductionTile = 16;
  int64_t genericReductionTile = 8;
  int64_t contractionMTile = 8;
  int64_t contractionKTile = 8;
};

/// Collects the policy inputs used by Hexagon launch-configuration heuristics.
struct PolicyConfig {
  TargetInfo target;
  const HeuristicCaps caps;
};

} // namespace mlir::iree_compiler::hexagon::codegen

#endif // ROOF_HEXAGON_CODEGEN_KERNELDISPATCHTYPES_H_

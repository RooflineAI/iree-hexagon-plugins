// Copyright 2026 RooflineAI GmbH
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef ROOF_HEXAGON_CODEGEN_PLANNING_DISPATCHPLANTYPES_H_
#define ROOF_HEXAGON_CODEGEN_PLANNING_DISPATCHPLANTYPES_H_

#include "iree/compiler/Codegen/Dialect/CPU/IR/IREECPUTypes.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/IR/Operation.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "llvm/ADT/SmallVector.h"

#include <cstdint>
#include <optional>

namespace mlir::iree_compiler::hexagon::codegen::planning {

using GlobalDimId = uint32_t;

/// One tile decision before it is assigned to an encoded tiling level.
struct TileDecision {
  /// Zero means that this dimension has no explicit tile at the selected
  /// level.
  int64_t size = 0;
  /// Whether this size is fixed by the hardware and cannot be adjusted by
  /// later planning stages.
  bool hardwareFixed = false;
  /// Additional active constraints can be represented here when needed. For
  /// example, root/non-root negotiation may eventually need a
  /// `mustDivideEnclosingTile` field.
};

/// Dispatch-wide facts for a unified loop dimension.
///
/// Iterator type and static extent deliberately do not live here: one global
/// dimension may appear with different iterator types or bounds on different
/// operations.
struct GlobalDim {
  GlobalDimId id = 0;
  bool presentInRoot = false;
};

/// An operation's local presentation of one unified dispatch dimension.
struct LocalDim {
  uint32_t position = 0;
  GlobalDimId global = 0;
  int64_t staticExtent = ShapedType::kDynamic;
  utils::IteratorType iteratorType = utils::IteratorType::parallel;
};

/// Ordered dimension information for one compute operation.
struct OpShape {
  Operation *op = nullptr;
  // Index of an operation inside the dispatch after topological sorting.
  uint32_t ordinal = 0;
  llvm::SmallVector<LocalDim> dimensions;
};

/// Root selection plus the local/global dimension graph produced by dispatch
/// analysis.
struct DispatchShape {
  FunctionOpInterface entryPoint;
  Operation *root = nullptr;
  llvm::SmallVector<GlobalDim> globalDimensions;
  llvm::SmallVector<OpShape> operations;
};

enum class ComputeTileLevel { CommonParallel, Reduction, InnerParallel };

/// Returns the lowering-config spelling of a compute tile level.
StringRef stringifyComputeTileLevel(ComputeTileLevel level);

/// Mechanical level classification shared by verification and encoding. Tile
/// sizes remain policy decisions; this only partitions them by iterator type
/// and root coverage.
ComputeTileLevel classifyComputeTileLevel(const DispatchShape &dispatchShape,
                                          const LocalDim &dimension);

/// Finds the shape entry for `op`, or returns null if `op` is not analyzed.
const OpShape *findOpShape(const DispatchShape &shape, Operation *op);

/// Returns the analyzed root shape. Callers must already have established that
/// the dispatch has an analyzable root.
const OpShape &getRootShape(const DispatchShape &shape);

/// Hardware derived invariants. Right now, these are hardcoded.
/// Future improvement: read this from the executable target in the IR.
struct TargetInfo {
  /// Fallback HVX width when the executable target does not provide one.
  int64_t nativeVectorBytes = 128;
  int64_t architecturalVectorRegisterCount = 32;
  int64_t hmxM = 32;
  int64_t hmxN = 32;
};

/// User-controlled planning behavior.
struct PlanningOptions {
  bool enableVTCM = true;
  bool enableHmxMatmul = false;
};

class DecisionTrace;

/// Inputs shared by every planning stage.
struct PlanningContext {
  FunctionOpInterface entryPoint;
  TargetInfo target;
  const PlanningOptions options;
  DecisionTrace &trace;
};

/// Hexagon-specific outer tensor staging, encoded separately from the CPU
/// tiling levels and applied before them by the VTCM tiling pass.
struct VTCMPlan {
  llvm::SmallVector<TileDecision> tileSizes;
};

/// Root-only loop/staging decisions.
/// These affect the whole dispatch and there is therefore no per-op form of
/// this type.
struct RootTilingPlan {
  std::optional<VTCMPlan> vtcm;
  llvm::SmallVector<TileDecision> distributionTile;
  llvm::SmallVector<TileDecision> cacheTile;
  /// The root's compute/fusion tile in root-local loop order. This is the sole
  /// source of root compute tiling; root operations do not also receive an
  /// OpComputeTilePlan.
  llvm::SmallVector<TileDecision> computeTile;
};

/// One dispatch-level strategy and its root-only decisions.
struct DispatchStrategy {
  IREE::CPU::LoweringPipeline pipeline = IREE::CPU::LoweringPipeline::Default;
  /// Whether encoding should request the pipeline's translation-info-controlled
  /// loop peeling. This is a strategy policy, separate from whether the
  /// selected pipeline supports or unconditionally performs peeling.
  bool requestLoopPeeling = false;
  RootTilingPlan rootTiling;
};

/// One independently realizable compute tile for one non-root operation.
struct OpComputeTilePlan {
  Operation *op = nullptr;
  llvm::SmallVector<TileDecision> computeTile;
  // A future resource model may attach typed estimates here, such as the
  // expected number of vector registers used by this tile.
};

/// Complete, still-unencoded result of planning a dispatch.
struct DispatchPlan {
  DispatchStrategy strategy;
  llvm::SmallVector<OpComputeTilePlan> nonRootComputeTilePlans;
};

} // namespace mlir::iree_compiler::hexagon::codegen::planning

#endif // ROOF_HEXAGON_CODEGEN_PLANNING_DISPATCHPLANTYPES_H_

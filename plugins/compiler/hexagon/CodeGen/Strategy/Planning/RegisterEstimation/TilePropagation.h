// Copyright 2026 RooflineAI GmbH
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HEXAGON_CODEGEN_PLANNING_REGISTERESTIMATION_TILEPROPAGATION_H_
#define IREE_HEXAGON_CODEGEN_PLANNING_REGISTERESTIMATION_TILEPROPAGATION_H_

#include "DispatchRegisterGraph.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/AffineMap.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir::iree_compiler::hexagon::codegen::planning {

/// The coordinate system the caller's tile vector lives in, plus one map per
/// op that takes an anchor tile to that op's loop tile.
struct TilePropagationResult {
  SmallVector<AnchorDim> anchorDims;
  /// in same order as to the inputs op list
  /// The symbolic expression mapping the anchor dims to that op's loops.
  SmallVector<AffineMap> anchorToOpLoops;
};

/// Builds the anchor's dim list and every op's anchor-to-loops map by walking
/// producer/consumer edges out from `ops[anchorIndex]`.
///
/// `reportFailure` receives the reason whenever this returns failure.
FailureOr<TilePropagationResult>
propagateTiles(ArrayRef<linalg::LinalgOp> ops, unsigned anchorIndex,
               llvm::function_ref<LogicalResult(const Twine &)> reportFailure);

} // namespace mlir::iree_compiler::hexagon::codegen::planning

#endif // IREE_HEXAGON_CODEGEN_PLANNING_REGISTERESTIMATION_TILEPROPAGATION_H_

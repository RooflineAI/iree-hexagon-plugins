// Copyright 2026 RooflineAI GmbH
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "DispatchPlanTypes.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/ErrorHandling.h"

#include <cassert>

namespace mlir::iree_compiler::hexagon::codegen::planning {

StringRef stringifyComputeTileLevel(ComputeTileLevel level) {
  switch (level) {
  case ComputeTileLevel::CommonParallel:
    return "VectorCommonParallel";
  case ComputeTileLevel::Reduction:
    return "VectorReduction";
  case ComputeTileLevel::InnerParallel:
    return "VectorInnerParallel";
  }
  llvm_unreachable("unknown compute tile level");
}

ComputeTileLevel classifyComputeTileLevel(const DispatchShape &dispatchShape,
                                          const LocalDim &dimension) {
  if (dimension.iteratorType == utils::IteratorType::reduction)
    return ComputeTileLevel::Reduction;
  if (dispatchShape.globalDimensions[dimension.global].presentInRoot)
    return ComputeTileLevel::CommonParallel;
  return ComputeTileLevel::InnerParallel;
}

const OpShape *findOpShape(const DispatchShape &shape, Operation *op) {
  auto it = llvm::find_if(shape.operations,
                          [&](const OpShape &item) { return item.op == op; });
  return it == shape.operations.end() ? nullptr : &*it;
}

const OpShape &getRootShape(const DispatchShape &shape) {
  const OpShape *rootShape = findOpShape(shape, shape.root);
  assert(rootShape && "strategy selection requires an analyzable root");
  return *rootShape;
}

} // namespace mlir::iree_compiler::hexagon::codegen::planning

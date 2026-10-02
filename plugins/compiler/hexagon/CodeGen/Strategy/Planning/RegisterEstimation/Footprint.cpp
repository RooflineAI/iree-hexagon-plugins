// Copyright 2026 RooflineAI GmbH
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "Footprint.h"

#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/TypeUtilities.h"
#include "llvm/Support/MathExtras.h"

#include <cassert>

namespace mlir::iree_compiler::hexagon::codegen::planning {

StringRef stringifyRegClass(RegClass regClass) {
  switch (regClass) {
  case RegClass::Vector:
    return "vector";
  case RegClass::Predicate:
    return "predicate";
  case RegClass::None:
    return "none";
  }
  return "unknown";
}

RegClass classifyResultType(Type type) {
  if (!type)
    return RegClass::None;
  Type elementType = getElementTypeOrSelf(type);
  if (elementType.isInteger(1))
    return RegClass::Predicate;
  if (elementType.isIntOrFloat())
    return RegClass::Vector;
  return RegClass::None;
}

unsigned getElementBitWidth(Type type) {
  if (!type)
    return 0;
  Type elementType = getElementTypeOrSelf(type);
  return elementType.isIntOrFloat() ? elementType.getIntOrFloatBitWidth() : 0;
}

FailureOr<Extent> Extent::of(AffineExpr expr) {
  Extent extent;
  SmallVector<AffineExpr> worklist = {expr};
  while (!worklist.empty()) {
    AffineExpr term = worklist.pop_back_val();
    if (auto dim = dyn_cast<AffineDimExpr>(term)) {
      extent.terms.push_back(dim.getPosition());
      continue;
    }
    if (isa<AffineConstantExpr>(term)) {
      extent.terms.push_back(kConstant);
      continue;
    }
    auto binary = dyn_cast<AffineBinaryOpExpr>(term);
    if (!binary || binary.getKind() != AffineExprKind::Add)
      return failure();
    worklist.push_back(binary.getRHS());
    worklist.push_back(binary.getLHS());
  }
  return extent;
}

int64_t Extent::resolve(ArrayRef<int64_t> tile) const {
  // Index ranges [0, a-1] and [0, b-1] add up to a + b - 1 distinct values.
  int64_t span = 1;
  for (int term : terms)
    span += (term == kConstant ? 1 : tile[term]) - 1;
  return span;
}

static FailureOr<SmallVector<Extent>> getExtents(AffineMap map) {
  SmallVector<Extent> extents;
  for (AffineExpr result : map.getResults()) {
    FailureOr<Extent> extent = Extent::of(result);
    if (failed(extent))
      return failure();
    extents.push_back(std::move(*extent));
  }
  return extents;
}

FailureOr<Footprint> Footprint::resident(AffineMap map, unsigned bits) {
  FailureOr<SmallVector<Extent>> extents = getExtents(map);
  if (failed(extents))
    return failure();
  Footprint footprint;
  footprint.bits = bits;
  if (!extents->empty()) {
    footprint.lanes.push_back(extents->pop_back_val());
    footprint.rows = std::move(*extents);
  }
  return footprint;
}

FailureOr<Footprint> Footprint::flat(AffineMap map, unsigned bits) {
  FailureOr<SmallVector<Extent>> extents = getExtents(map);
  if (failed(extents))
    return failure();
  Footprint footprint;
  footprint.bits = bits;
  footprint.lanes = std::move(*extents);
  return footprint;
}

static int64_t product(ArrayRef<Extent> extents, ArrayRef<int64_t> tile) {
  int64_t count = 1;
  for (const Extent &extent : extents)
    count *= extent.resolve(tile);
  assert(count > 0 && "tile extents are positive");
  return count;
}

int64_t Footprint::registers(ArrayRef<int64_t> tile, int64_t vectorBits) const {
  if (bits == 0)
    return 0;
  return copies * product(rows, tile) *
         llvm::divideCeil(product(lanes, tile) * static_cast<int64_t>(bits),
                          vectorBits);
}

int64_t Footprint::usefulBits(ArrayRef<int64_t> tile,
                              int64_t vectorBits) const {
  if (bits == 0)
    return 0;
  if (lanes.empty())
    return registers(tile, vectorBits) * vectorBits;
  return copies * product(rows, tile) * product(lanes, tile) *
         static_cast<int64_t>(bits);
}

llvm::SmallBitVector getDimsOf(AffineMap map) {
  llvm::SmallBitVector dims(map.getNumDims());
  for (unsigned dim = 0, e = map.getNumDims(); dim < e; ++dim)
    if (map.isFunctionOfDim(dim))
      dims.set(dim);
  return dims;
}

llvm::SmallBitVector getLaneDims(AffineMap map) {
  llvm::SmallBitVector dims(map.getNumDims());
  if (map.getNumResults() == 0)
    return dims;
  AffineExpr innermost = map.getResult(map.getNumResults() - 1);
  for (unsigned dim = 0, e = map.getNumDims(); dim < e; ++dim)
    if (innermost.isFunctionOfDim(dim))
      dims.set(dim);
  return dims;
}

std::optional<unsigned> getLaneDim(AffineMap map) {
  if (map.getNumResults() == 0)
    return std::nullopt;
  auto dim = dyn_cast<AffineDimExpr>(map.getResult(map.getNumResults() - 1));
  if (!dim)
    return std::nullopt;
  return dim.getPosition();
}

} // namespace mlir::iree_compiler::hexagon::codegen::planning

// Copyright 2026 RooflineAI GmbH
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef ROOF_HEXAGON_CODEGEN_PLANNING_REGISTERESTIMATION_FOOTPRINT_H_
#define ROOF_HEXAGON_CODEGEN_PLANNING_REGISTERESTIMATION_FOOTPRINT_H_

#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Types.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallBitVector.h"
#include "llvm/ADT/SmallVector.h"

#include <cstdint>
#include <optional>

namespace mlir::iree_compiler::hexagon::codegen::planning {

/// Which register file a value occupies.
enum class RegClass {
  Vector, // HVX V registers
  Predicate,
  None, // scalar, or otherwise not held in an HVX register
};

StringRef stringifyRegClass(RegClass regClass);

/// The register class a value of `type` occupies: `i1` is a predicate, any
/// other int or float is a vector
RegClass classifyResultType(Type type);

/// Bit width of `type`'s elements, or 0 for anything that does not live in a
/// vector register.
unsigned getElementBitWidth(Type type);

/// How many elements one index expression of a tile spans, as a function of
/// the anchor tile
struct Extent {
  /// One entry per term: an anchor dim, or `kConstant` for a fixed index.
  static constexpr int kConstant = -1;
  SmallVector<int, 2> terms;

  static Extent ofDim(unsigned dim) { return Extent{{static_cast<int>(dim)}}; }
  /// Fails for anything but sums of dims and constants.
  static FailureOr<Extent> of(AffineExpr expr);

  int64_t resolve(ArrayRef<int64_t> tile) const;
};

/// What holding one value costs, as a function of the anchor tile:
struct Footprint {
  SmallVector<Extent> rows;
  SmallVector<Extent> lanes;
  /// Element bit width. 0 means the value holds no HVX register.
  unsigned bits = 0;
  int64_t copies = 1;

  /// A resident tile shaped `map` (anchor dims -> tile dims): every result
  /// but the innermost is a row. A rank-0 map is a splat of one register.
  static FailureOr<Footprint> resident(AffineMap map, unsigned bits);
  /// Every element of the tile shaped `map`, packed flat.
  static FailureOr<Footprint> flat(AffineMap map, unsigned bits);

  int64_t registers(ArrayRef<int64_t> tile, int64_t vectorBits) const;
  /// The bits of those registers that hold data: every element once, without
  /// padding. A splat fills each lane of its register with the one value, so
  /// all of its registers count as full.
  int64_t usefulBits(ArrayRef<int64_t> tile, int64_t vectorBits) const;
};

/// Anchor dims `map` depends on at all.
llvm::SmallBitVector getDimsOf(AffineMap map);

/// Anchor dims `map`'s innermost result - the one a register's lanes run
/// along - depends on. Empty for a rank-0 map.
llvm::SmallBitVector getLaneDims(AffineMap map);

/// The anchor dim `map`'s innermost result is, when it is a plain dim. A
/// window or a constant lane runs along no single dim.
std::optional<unsigned> getLaneDim(AffineMap map);

} // namespace mlir::iree_compiler::hexagon::codegen::planning

#endif // ROOF_HEXAGON_CODEGEN_PLANNING_REGISTERESTIMATION_FOOTPRINT_H_

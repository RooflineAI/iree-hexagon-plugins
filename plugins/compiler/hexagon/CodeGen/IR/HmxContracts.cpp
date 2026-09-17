// Copyright 2026 RooflineAI GmbH
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "hexagon/CodeGen/IR/HmxContracts.h"

#include "hexagon/CodeGen/IR/HexagonTypes.h"

#include "hexagon/Dialect/HexagonMem/IR/HexagonMemDialect.h"

#include "mlir/Analysis/DataFlow/IntegerDivisibilityAnalysis.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Operation.h"

#include <numeric>

namespace mlir::iree_compiler::IREE::Hexagon {
namespace {

LogicalResult verifyStaticGridCovers(Operation *op, ShapedType gridType,
                                     ShapedType matrixType, bool transpose,
                                     StringRef role) {
  int64_t rows = matrixType.getDimSize(transpose ? 1 : 0);
  int64_t columns = matrixType.getDimSize(transpose ? 0 : 1);
  if (!ShapedType::isDynamic(rows) &&
      gridType.getDimSize(0) < getHmxTileCount(rows)) {
    return op->emitOpError()
           << role << " does not cover ceil(rows/32) logical tiles";
  }
  if (!ShapedType::isDynamic(columns) &&
      gridType.getDimSize(1) < getHmxTileCount(columns)) {
    return op->emitOpError()
           << role << " does not cover ceil(columns/32) logical tiles";
  }
  return success();
}

// Fails closed for operations without an integer-divisibility model, including
// index casts. Add an upstream model if such offsets become part of the HMX
// lowering rather than guessing here.
bool isKnownMultipleOf(Value value, int64_t requiredMultiple,
                       DataFlowSolver &solver) {
  const auto *lattice =
      solver.lookupState<dataflow::IntegerDivisibilityLattice>(value);
  if (!lattice || lattice->getValue().isUninitialized()) {
    return false;
  }
  const ConstantIntDivisibility &divisibility = lattice->getValue().getValue();
  return divisibility.udiv() % requiredMultiple == 0 ||
         divisibility.sdiv() % requiredMultiple == 0;
}

// Byte offset that `subviewOp` adds to its source base address, reduced modulo
// `requiredAlignment`. Fails when the offset cannot be proven to preserve
// alignment.
FailureOr<int64_t> getSubViewByteOffsetModulo(memref::SubViewOp subviewOp,
                                              int64_t requiredAlignment,
                                              DataFlowSolver &solver) {
  auto sourceType = cast<MemRefType>(subviewOp.getSource().getType());
  SmallVector<int64_t> sourceStrides;
  int64_t sourceOffset;
  if (failed(sourceType.getStridesAndOffset(sourceStrides, sourceOffset))) {
    return failure();
  }
  int64_t elementBytes = getHmxElementSizeBytes(sourceType.getElementType());
  if (elementBytes == 0) {
    return failure();
  }

  int64_t byteOffsetModulo = 0;
  for (auto [offset, stride] :
       llvm::zip_equal(subviewOp.getMixedOffsets(), sourceStrides)) {
    if (ShapedType::isDynamic(stride)) {
      return failure();
    }
    int64_t strideBytes = stride * elementBytes;
    int64_t strideBytesModulo = strideBytes % requiredAlignment;
    std::optional<int64_t> constantOffset = getConstantIntValue(offset);
    if (!constantOffset) {
      // A dynamic offset preserves alignment when it is a multiple of
      // alignment/gcd(alignment, strideBytes). For example, a 32-column f32
      // tile offset is a multiple of 32; multiplied by its 4-byte innermost
      // stride it preserves 128-byte alignment.
      int64_t requiredOffsetMultiple =
          requiredAlignment /
          std::gcd(std::abs(strideBytes), requiredAlignment);
      if (!isKnownMultipleOf(cast<Value>(offset), requiredOffsetMultiple,
                             solver)) {
        return failure();
      }
      continue;
    }
    byteOffsetModulo +=
        ((*constantOffset % requiredAlignment) * strideBytesModulo) %
        requiredAlignment;
    byteOffsetModulo %= requiredAlignment;
  }
  return byteOffsetModulo;
}

} // namespace

bool hasStaticHmxTileSuffix(ShapedType type) {
  if (!type || type.getRank() < kHmxTileRank) {
    return false;
  }
  ArrayRef<int64_t> shape = type.getShape();
  return shape.take_back(kHmxTileRank) ==
         ArrayRef<int64_t>({kHmxTileRows, kHmxTileColumns, kHmxTileInterleave});
}

bool isHmxTileGrid(ShapedType type) {
  return type && type.getRank() == kHmxTileGridRank;
}

bool hasPositiveStaticTileGrid(ShapedType type) {
  for (unsigned dim : {0u, 1u}) {
    int64_t size = type.getDimSize(dim);
    if (ShapedType::isDynamic(size) || size <= 0) {
      return false;
    }
  }
  return true;
}

bool isHmxF16Tile(ShapedType type, bool requireGrid) {
  int64_t expectedRank = requireGrid ? kHmxTileGridRank : kHmxTileRank;
  return type && type.getRank() == expectedRank &&
         type.getElementType().isF16() && hasStaticHmxTileSuffix(type) &&
         (!requireGrid || hasPositiveStaticTileGrid(type));
}

bool isHmxAccumulator(Type type) {
  auto accType = dyn_cast<HmxAccType>(type);
  return accType &&
         accType.getShape() ==
             ArrayRef<int64_t>({kHmxLogicalTileSize, kHmxLogicalTileSize}) &&
         accType.getElementType().isF32();
}

LogicalResult verifyHmxF16Tile(Operation *op, ShapedType type, StringRef role,
                               bool requireGrid) {
  int64_t expectedRank = requireGrid ? kHmxTileGridRank : kHmxTileRank;
  if (type.getRank() != expectedRank || !type.getElementType().isF16()) {
    return op->emitOpError()
           << role << " must be a rank-" << expectedRank << " f16 HMX "
           << (requireGrid ? "tile grid" : "tile");
  }
  if (!hasStaticHmxTileSuffix(type)) {
    return op->emitOpError()
           << role << " must have the static physical suffix [16, 32, 2]";
  }
  if (requireGrid && !hasPositiveStaticTileGrid(type)) {
    return op->emitOpError()
           << role << " must have positive static tile-grid dimensions";
  }
  return success();
}

LogicalResult verifyHmxSingleF16Tile(Operation *op, ShapedType type,
                                     StringRef role, bool allowSingletonGrid) {
  bool grid = allowSingletonGrid && isHmxTileGrid(type);
  if (!isHmxF16Tile(type, /*requireGrid=*/grid) ||
      (grid && (type.getDimSize(0) != 1 || type.getDimSize(1) != 1))) {
    return op->emitOpError() << role << " must be one f16 [16, 32, 2] HMX tile";
  }
  return success();
}

LogicalResult verifyHmxPackContract(Operation *op, ShapedType sourceType,
                                    ShapedType destType, int64_t dim) {
  if (sourceType.getRank() != 2 || !sourceType.getElementType().isF16()) {
    return op->emitOpError("source must be a rank-2 f16 shaped value");
  }
  if (failed(verifyHmxF16Tile(op, destType, "destination",
                              /*requireGrid=*/true))) {
    return failure();
  }
  if (dim != 0 && dim != 1) {
    return op->emitOpError("dim must be 0 or 1");
  }
  return verifyStaticGridCovers(op, destType, sourceType, dim == 1,
                                "destination grid");
}

LogicalResult verifyHmxMatmulContract(Operation *op, ShapedType lhsType,
                                      ShapedType rhsType, ShapedType accType,
                                      bool allowSingleTileAccumulator) {
  if (failed(verifyHmxF16Tile(op, lhsType, "lhs", /*requireGrid=*/true)) ||
      failed(verifyHmxF16Tile(op, rhsType, "rhs", /*requireGrid=*/true))) {
    return failure();
  }
  bool singleTileAcc =
      allowSingleTileAccumulator && accType.getRank() == kHmxTileRank;
  if (failed(verifyHmxF16Tile(op, accType, "accumulator",
                              /*requireGrid=*/!singleTileAcc))) {
    return failure();
  }
  if (lhsType.getDimSize(1) != rhsType.getDimSize(0)) {
    return op->emitOpError("lhs and rhs K tile counts must match");
  }
  if (singleTileAcc) {
    if (lhsType.getDimSize(0) != 1 || rhsType.getDimSize(1) != 1) {
      return op->emitOpError(
          "a rank-3 accumulator requires singleton lhs M and rhs N grids");
    }
    return success();
  }
  if (lhsType.getDimSize(0) != accType.getDimSize(0) ||
      rhsType.getDimSize(1) != accType.getDimSize(1)) {
    return op->emitOpError(
        "accumulator grid must match lhs M and rhs N tile counts");
  }
  return success();
}

LogicalResult verifyHmxUnpackContract(Operation *op, ShapedType sourceType,
                                      ShapedType destType, int64_t dim) {
  // `verifyHmxF16Tile` checks the rank against the form selected here, so a
  // source that is neither a rank-3 tile nor a rank-5 grid is reported as a
  // rank-3 tile mismatch rather than through a separate pre-check.
  bool sourceIsGrid = isHmxTileGrid(sourceType);
  if (failed(verifyHmxF16Tile(op, sourceType, "source", sourceIsGrid))) {
    return failure();
  }
  if (destType.getRank() != 2 || (!destType.getElementType().isF16() &&
                                  !destType.getElementType().isF32())) {
    return op->emitOpError("destination must be a rank-2 f16 or f32 value");
  }
  if (dim != 0) {
    return op->emitOpError("dim must be 0");
  }
  if (sourceIsGrid) {
    return verifyStaticGridCovers(op, sourceType, destType,
                                  /*transpose=*/false, "source grid");
  }
  for (unsigned destDim : {0u, 1u}) {
    int64_t size = destType.getDimSize(destDim);
    if (!ShapedType::isDynamic(size) && size > kHmxLogicalTileSize) {
      return op->emitOpError(
          "a rank-3 source can unpack at most one logical 32x32 tile");
    }
  }
  return success();
}

Value getHmxMemRefViewSource(Value value) {
  if (auto castOp = value.getDefiningOp<memref::CastOp>()) {
    return castOp.getSource();
  }
  if (auto collapseOp = value.getDefiningOp<memref::CollapseShapeOp>()) {
    return collapseOp.getSrc();
  }
  if (auto expandOp = value.getDefiningOp<memref::ExpandShapeOp>()) {
    return expandOp.getSrc();
  }
  if (auto subviewOp = value.getDefiningOp<memref::SubViewOp>()) {
    return subviewOp.getSource();
  }
  return {};
}

Value getHmxAllocationBase(Value value) {
  while (Value source = getHmxMemRefViewSource(value)) {
    value = source;
  }
  return value;
}

std::optional<int64_t> getDeclaredHmxAlignment(Value value) {
  if (auto assumeOp = value.getDefiningOp<memref::AssumeAlignmentOp>()) {
    return static_cast<int64_t>(assumeOp.getAlignment());
  }
  if (auto allocOp = value.getDefiningOp<memref::AllocOp>()) {
    return static_cast<int64_t>(allocOp.getAlignment().value_or(0));
  }
  if (auto allocaOp = value.getDefiningOp<memref::AllocaOp>()) {
    return static_cast<int64_t>(allocaOp.getAlignment().value_or(0));
  }
  if (auto allocOp = value.getDefiningOp<hexagonmem::AllocOp>()) {
    return static_cast<int64_t>(allocOp.getAlignment());
  }
  return std::nullopt;
}

unsigned getHmxElementSizeBytes(Type elementType) {
  if (elementType.isF16() || elementType.isBF16()) {
    return 2;
  }
  if (elementType.isF32() || elementType.isInteger(32)) {
    return 4;
  }
  if (elementType.isInteger(8)) {
    return 1;
  }
  return 0;
}

bool providesHmxAlignment(Value value, int64_t requiredAlignment,
                          DataFlowSolver &solver) {
  if (std::optional<int64_t> declared = getDeclaredHmxAlignment(value)) {
    return *declared % requiredAlignment == 0;
  }

  auto subviewOp = value.getDefiningOp<memref::SubViewOp>();
  if (!subviewOp) {
    // Every other view forwards its source's base address unchanged.
    Value source = getHmxMemRefViewSource(value);
    return source && providesHmxAlignment(source, requiredAlignment, solver);
  }
  if (!providesHmxAlignment(subviewOp.getSource(), requiredAlignment, solver)) {
    return false;
  }
  FailureOr<int64_t> byteOffsetModulo =
      getSubViewByteOffsetModulo(subviewOp, requiredAlignment, solver);
  return succeeded(byteOffsetModulo) && *byteOffsetModulo == 0;
}
} // namespace mlir::iree_compiler::IREE::Hexagon

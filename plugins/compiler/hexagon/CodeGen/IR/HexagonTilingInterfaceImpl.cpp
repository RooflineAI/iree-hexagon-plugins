// Copyright 2026 RooflineAI GmbH
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "hexagon/CodeGen/IR/HexagonOps.h"

#include "hexagon/CodeGen/IR/HmxContracts.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Interfaces/TilingInterface.h"

namespace mlir::iree_compiler::IREE::Hexagon {

/// Scales an HMX tile-grid coordinate to a logical tensor coordinate.
static OpFoldResult multiplyByHmxTileSize(OpBuilder &builder, Location loc,
                                          OpFoldResult value) {
  AffineExpr d0;
  bindDims(builder.getContext(), d0);
  return affine::makeComposedFoldedAffineApply(builder, loc,
                                               d0 * kHmxLogicalTileSize, value);
}

/// Clips a logical HMX tile to the remaining extent of `tensor` at `offset`.
/// Physical HMX tiles are always 32x32, but a logical unpack destination is not
/// padded and its boundary slice must remain in bounds.
static OpFoldResult getClippedHmxTileSize(OpBuilder &builder, Location loc,
                                          Value tensor, unsigned dim,
                                          OpFoldResult offset) {
  auto type = cast<RankedTensorType>(tensor.getType());
  int64_t staticSize = type.getDimSize(dim);
  if (!ShapedType::isDynamic(staticSize) &&
      staticSize % kHmxLogicalTileSize == 0) {
    return builder.getIndexAttr(kHmxLogicalTileSize);
  }

  AffineExpr d0, d1;
  bindDims(builder.getContext(), d0, d1);
  AffineMap minMap = AffineMap::get(
      2, 0,
      {getAffineConstantExpr(kHmxLogicalTileSize, builder.getContext()),
       d0 - d1},
      builder.getContext());
  return affine::makeComposedFoldedAffineMin(
      builder, loc, minMap,
      {tensor::getMixedSize(builder, loc, tensor, dim), offset});
}

static SmallVector<OpFoldResult>
getClippedHmxTileSizes(OpBuilder &builder, Location loc, Value tensor,
                       ArrayRef<OpFoldResult> offsets) {
  assert(offsets.size() == 2 && "expected a rank-2 logical tile");
  return {getClippedHmxTileSize(builder, loc, tensor, 0, offsets[0]),
          getClippedHmxTileSize(builder, loc, tensor, 1, offsets[1])};
}

/// Returns whether a rank-5 slice contains exactly one physical HMX tile.
static bool isSingleFullHmxTileSlice(RankedTensorType type,
                                     ArrayRef<OpFoldResult> offsets,
                                     ArrayRef<OpFoldResult> sizes) {
  // The op verifiers guarantee the element type and physical tile suffix.
  if (type.getRank() != 5 || offsets.size() != 5 || sizes.size() != 5) {
    return false;
  }

  for (unsigned dim : {0u, 1u}) {
    auto size = getConstantIntValue(sizes[dim]);
    if (!size || *size != 1) {
      return false;
    }
  }

  // The trailing dimensions encode one indivisible physical HMX tile. The
  // fusion mappings project them away, so they must describe the full tile at
  // zero offset for the mapping to be exact.
  for (unsigned dim = 2; dim < 5; ++dim) {
    int64_t dimSize = type.getDimSize(dim);
    auto offset = getConstantIntValue(offsets[dim]);
    auto size = getConstantIntValue(sizes[dim]);
    if (!offset || *offset != 0 || !size || *size != dimSize) {
      return false;
    }
  }
  return true;
}

/// Conservatively proves that an offset is aligned to a logical HMX tile.
/// TilingInterface hooks have no pipeline DataFlowSolver to query, so this
/// recognizes the small set of expressions emitted by the tiling pipeline.
/// Runtime ABI validation uses IntegerDivisibilityAnalysis instead.
static bool isKnownHmxTileAligned(OpFoldResult offset) {
  if (std::optional<int64_t> constant = getConstantIntValue(offset)) {
    return *constant % kHmxLogicalTileSize == 0;
  }
  Value value = cast<Value>(offset);
  if (auto apply = value.getDefiningOp<affine::AffineApplyOp>()) {
    return apply.getAffineMap().getResult(0).isMultipleOf(kHmxLogicalTileSize);
  }
  if (auto mul = value.getDefiningOp<arith::MulIOp>()) {
    APInt constant;
    return (matchPattern(mul.getLhs(), m_ConstantInt(&constant)) &&
            constant.srem(kHmxLogicalTileSize) == 0) ||
           (matchPattern(mul.getRhs(), m_ConstantInt(&constant)) &&
            constant.srem(kHmxLogicalTileSize) == 0);
  }
  if (auto add = value.getDefiningOp<arith::AddIOp>()) {
    return isKnownHmxTileAligned(add.getLhs()) &&
           isKnownHmxTileAligned(add.getRhs());
  }
  if (auto sub = value.getDefiningOp<arith::SubIOp>()) {
    return isKnownHmxTileAligned(sub.getLhs()) &&
           isKnownHmxTileAligned(sub.getRhs());
  }
  if (auto blockArg = dyn_cast<BlockArgument>(value)) {
    Operation *parentOp = blockArg.getOwner()->getParentOp();
    if (auto forallOp = dyn_cast_or_null<scf::ForallOp>(parentOp)) {
      auto ivs = forallOp.getInductionVars();
      auto it = llvm::find(ivs, blockArg);
      if (it != ivs.end()) {
        unsigned index = it - ivs.begin();
        return isKnownHmxTileAligned(forallOp.getMixedLowerBound()[index]) &&
               isKnownHmxTileAligned(forallOp.getMixedStep()[index]);
      }
    }
    if (auto forOp = dyn_cast_or_null<scf::ForOp>(parentOp)) {
      if (blockArg == forOp.getInductionVar()) {
        return isKnownHmxTileAligned(forOp.getLowerBound()) &&
               isKnownHmxTileAligned(forOp.getStep());
      }
    }
  }
  return false;
}

/// Matches the boundary size emitted by the generic tiler for a static ragged
/// extent: `affine.min(extent - aligned_offset, 32)`. This mirrors the private
/// `getBoundedTileSize` helper in MLIR's TileUsingInterface.cpp. Accepting only
/// this exact form keeps interior partial tiles and arbitrary dynamic sizes
/// unfusable.
static bool isExactClippedHmxTileSize(OpFoldResult size, OpFoldResult offset,
                                      int64_t extent) {
  if (ShapedType::isDynamic(extent)) {
    return false;
  }
  auto sizeValue = dyn_cast<Value>(size);
  auto offsetValue = dyn_cast<Value>(offset);
  if (!sizeValue || !offsetValue) {
    return false;
  }
  auto minOp = sizeValue.getDefiningOp<affine::AffineMinOp>();
  if (!minOp) {
    return false;
  }
  AffineMap map = minOp.getAffineMap();
  auto operandIt = llvm::find(minOp.getOperands(), offsetValue);
  if (operandIt == minOp.getOperands().end()) {
    return false;
  }
  unsigned operandNumber = operandIt - minOp.getOperands().begin();
  if (operandNumber >= map.getNumDims()) {
    return false;
  }
  AffineExpr remainder = getAffineConstantExpr(extent, map.getContext()) -
                         getAffineDimExpr(operandNumber, map.getContext());
  bool hasFullTile = false;
  bool hasRemainder = false;
  for (AffineExpr result : map.getResults()) {
    hasFullTile |=
        result == getAffineConstantExpr(kHmxLogicalTileSize, map.getContext());
    hasRemainder |= result == remainder;
  }
  return hasFullTile && hasRemainder;
}

/// Checks that one logical dimension describes exactly one full or clipped
/// HMX tile at `offset`.
static bool isExactLogicalHmxTileSize(int64_t extent, OpFoldResult offset,
                                      OpFoldResult size) {
  if (!isKnownHmxTileAligned(offset)) {
    return false;
  }

  std::optional<int64_t> constantSize = getConstantIntValue(size);
  if (!constantSize) {
    return isExactClippedHmxTileSize(size, offset, extent);
  }
  if (*constantSize <= 0 || *constantSize > kHmxLogicalTileSize) {
    return false;
  }

  std::optional<int64_t> constantOffset = getConstantIntValue(offset);
  if (ShapedType::isDynamic(extent) || !constantOffset) {
    return *constantSize == kHmxLogicalTileSize &&
           (ShapedType::isDynamic(extent) || extent % kHmxLogicalTileSize == 0);
  }
  int64_t expectedSize =
      std::min(kHmxLogicalTileSize, extent - *constantOffset);
  return expectedSize > 0 && *constantSize == expectedSize;
}

/// Returns whether every logical destination extent is a whole number of 32x32
/// HMX tiles. Consumer fusion needs this: `getResultTilePosition` is invoked on
/// the *tiled* consumer, whose destination is already one clipped tile, so it
/// can no longer recover the untiled extent needed to clip a boundary tile.
/// With aligned extents every tile is a full 32x32 tile and no clipping is
/// required. Ragged destinations still tile and still fuse in the producer
/// direction, where the hook runs on the untiled op.
static bool hasTileAlignedLogicalDest(RankedTensorType destType) {
  if (destType.getRank() != 2) {
    return false;
  }
  return llvm::all_of(destType.getShape(), [](int64_t extent) {
    return !ShapedType::isDynamic(extent) && extent % kHmxLogicalTileSize == 0;
  });
}

/// Checks the subset of logical result tiles that maps to exactly one physical
/// tile. Dynamic/ragged result tiles are still supported by ordinary tiling;
/// consumer fusion is rejected unless this mapping can be proven exactly.
static bool isSingleLogicalHmxResultTile(RankedTensorType resultType,
                                         ArrayRef<OpFoldResult> offsets,
                                         ArrayRef<OpFoldResult> sizes) {
  if (resultType.getRank() != 2 || offsets.size() != 2 || sizes.size() != 2) {
    return false;
  }
  for (unsigned dim : {0u, 1u}) {
    if (!isExactLogicalHmxTileSize(resultType.getDimSize(dim), offsets[dim],
                                   sizes[dim])) {
      return false;
    }
  }
  return true;
}

// These methods implement TilingInterface. See
// mlir/Interfaces/TilingInterface.td for the generic method contracts; the
// comments below describe the HMX-specific iteration and tile mappings.

/// Uses the accumulator's (M tiles, N tiles) grid as the iteration domain.
SmallVector<Range> TensorHmxMatmulOp::getIterationDomain(OpBuilder &builder) {
  // The verifier guarantees a positive static rank-5 accumulator tile grid.
  Location loc = getLoc();
  OpFoldResult zero = builder.getIndexAttr(0);
  OpFoldResult one = builder.getIndexAttr(1);
  return {
      Range{zero, tensor::getMixedSize(builder, loc, getAcc(), 0), one},
      Range{zero, tensor::getMixedSize(builder, loc, getAcc(), 1), one},
  };
}

/// Exposes M and N tile traversal as parallel TilingInterface iterators.
SmallVector<utils::IteratorType> TensorHmxMatmulOp::getLoopIteratorTypes() {
  return {utils::IteratorType::parallel, utils::IteratorType::parallel};
}

/// Allows consumer fusion only when it requests one complete physical tile.
bool TensorHmxMatmulOp::isOpFusableWithConsumerSlice(
    unsigned resultNumber, ArrayRef<OpFoldResult> offsets,
    ArrayRef<OpFoldResult> sizes) {
  if (resultNumber != 0 || offsets.size() != 5 || sizes.size() != 5) {
    return false;
  }
  auto resultType =
      cast<RankedTensorType>(getOperation()->getResult(resultNumber).getType());
  return isSingleFullHmxTileSlice(resultType, offsets, sizes);
}

/// Materializes one output-grid tile while retaining the complete K tile grid.
FailureOr<TilingResult>
TensorHmxMatmulOp::getTiledImplementation(OpBuilder &builder,
                                          ArrayRef<OpFoldResult> offsets,
                                          ArrayRef<OpFoldResult> sizes) {
  if (offsets.size() != 2 || sizes.size() != 2) {
    return failure();
  }

  Location loc = getLoc();
  auto accType = cast<RankedTensorType>(getAcc().getType());

  if (auto mTileSize = getConstantIntValue(sizes[0]);
      !mTileSize || *mTileSize != 1) {
    return failure();
  }
  if (auto nTileSize = getConstantIntValue(sizes[1]);
      !nTileSize || *nTileSize != 1) {
    return failure();
  }

  OpFoldResult zero = builder.getIndexAttr(0);
  OpFoldResult one = builder.getIndexAttr(1);
  SmallVector<OpFoldResult> strides(5, one);

  OpFoldResult kTiles = tensor::getMixedSize(builder, loc, getLhs(), 1);
  // The verifier pins the trailing [16, 32, 2] physical tile on all three
  // operands, but each slice reads its extents from its own operand so the
  // sizes stay tied to the value being sliced.
  auto innerTileSizes = [&](Value operand) {
    return SmallVector<OpFoldResult>{
        tensor::getMixedSize(builder, loc, operand, 2),
        tensor::getMixedSize(builder, loc, operand, 3),
        tensor::getMixedSize(builder, loc, operand, 4)};
  };

  SmallVector<OpFoldResult> lhsSizes = {one, kTiles};
  llvm::append_range(lhsSizes, innerTileSizes(getLhs()));
  auto lhsSlice = tensor::ExtractSliceOp::create(
      builder, loc, getLhs(),
      SmallVector<OpFoldResult>{offsets[0], zero, zero, zero, zero}, lhsSizes,
      strides);

  SmallVector<OpFoldResult> rhsSizes = {kTiles, one};
  llvm::append_range(rhsSizes, innerTileSizes(getRhs()));
  auto rhsSlice = tensor::ExtractSliceOp::create(
      builder, loc, getRhs(),
      SmallVector<OpFoldResult>{zero, offsets[1], zero, zero, zero}, rhsSizes,
      strides);

  auto accTileType = RankedTensorType::get(
      {1, 1, kHmxTileRows, kHmxTileColumns, kHmxTileInterleave},
      accType.getElementType());
  SmallVector<OpFoldResult> accSizes = {one, one};
  llvm::append_range(accSizes, innerTileSizes(getAcc()));
  auto accSlice = tensor::ExtractSliceOp::create(
      builder, loc, accTileType, getAcc(),
      SmallVector<OpFoldResult>{offsets[0], offsets[1], zero, zero, zero},
      accSizes, strides);

  auto tiledOp =
      TensorHmxMatmulOp::create(builder, loc, accTileType, lhsSlice.getResult(),
                                rhsSlice.getResult(), accSlice.getResult());

  // No slices are reported to the fusion driver. `generatedSlices` seeds the
  // producer-fusion worklist in `scf::tileConsumerAndFuseProducersUsingSCF`,
  // and the producers here are the `hmx.tensor_pack` ops: an LHS panel is
  // reused by every N tile and an RHS panel by every M tile, so pulling them
  // into this loop would re-pack the same panel once per output tile. The
  // packs are deliberately hoisted and only sliced inside the loop.

  return TilingResult{
      {tiledOp},
      SmallVector<Value>(tiledOp->getResults()),
      {},
  };
}

/// Maps an (M, N) iteration tile to the packed rank-5 result coordinates.
LogicalResult TensorHmxMatmulOp::getResultTilePosition(
    OpBuilder &builder, unsigned resultNumber, ArrayRef<OpFoldResult> offsets,
    ArrayRef<OpFoldResult> sizes, SmallVector<OpFoldResult> &resultOffsets,
    SmallVector<OpFoldResult> &resultSizes) {
  if (resultNumber != 0 || offsets.size() != 2 || sizes.size() != 2) {
    return failure();
  }
  OpFoldResult zero = builder.getIndexAttr(0);
  OpFoldResult one = builder.getIndexAttr(1);
  resultOffsets = {offsets[0], offsets[1], zero, zero, zero};
  resultSizes = {one, one, builder.getIndexAttr(kHmxTileRows),
                 builder.getIndexAttr(kHmxTileColumns),
                 builder.getIndexAttr(kHmxTileInterleave)};
  return success();
}

/// Implements consumer fusion by mapping a result tile to an iteration tile.
FailureOr<TilingResult> TensorHmxMatmulOp::generateResultTileValue(
    OpBuilder &builder, unsigned resultNumber, ArrayRef<OpFoldResult> offsets,
    ArrayRef<OpFoldResult> sizes) {
  SmallVector<OpFoldResult> iterDomainOffsets;
  SmallVector<OpFoldResult> iterDomainSizes;
  if (failed(getIterationDomainTileFromResultTile(
          builder, resultNumber, offsets, sizes, iterDomainOffsets,
          iterDomainSizes))) {
    return failure();
  }
  return getTiledImplementation(builder, iterDomainOffsets, iterDomainSizes);
}

/// Projects a complete packed result tile onto the (M, N) iteration grid.
LogicalResult TensorHmxMatmulOp::getIterationDomainTileFromResultTile(
    OpBuilder &builder, unsigned resultNumber, ArrayRef<OpFoldResult> offsets,
    ArrayRef<OpFoldResult> sizes,
    SmallVectorImpl<OpFoldResult> &iterDomainOffsets,
    SmallVectorImpl<OpFoldResult> &iterDomainSizes) {
  if (resultNumber != 0) {
    return failure();
  }
  auto resultType =
      cast<RankedTensorType>(getOperation()->getResult(resultNumber).getType());
  if (!isSingleFullHmxTileSlice(resultType, offsets, sizes)) {
    return failure();
  }

  iterDomainOffsets.clear();
  iterDomainOffsets.append({offsets[0], offsets[1]});
  iterDomainSizes.clear();
  iterDomainSizes.append({sizes[0], sizes[1]});
  return success();
}

/// Uses the packed source's (M tiles, N tiles) grid as the iteration domain.
///
/// A rank-3 source is a single terminal physical tile with no outer grid left
/// to traverse, so it reports an empty domain. The TilingInterface has no way
/// to say "not tileable", but an empty domain is handled safely: `tileUsingSCF`
/// sees no tile sizes, takes its `all_of(tileSizes, isZeroInteger)` early
/// return, and leaves the op untouched instead of building a degenerate loop
/// nest.
SmallVector<Range> TensorHmxUnpackOp::getIterationDomain(OpBuilder &builder) {
  auto sourceType = cast<RankedTensorType>(getSource().getType());
  if (sourceType.getRank() != 5) {
    return {};
  }

  Location loc = getLoc();
  OpFoldResult zero = builder.getIndexAttr(0);
  OpFoldResult one = builder.getIndexAttr(1);
  return {
      Range{zero, tensor::getMixedSize(builder, loc, getSource(), 0), one},
      Range{zero, tensor::getMixedSize(builder, loc, getSource(), 1), one},
  };
}

/// Exposes M and N tile traversal as parallel TilingInterface iterators.
/// Stays in lockstep with `getIterationDomain`: a rank-3 terminal tile has an
/// empty domain and therefore no iterators.
SmallVector<utils::IteratorType> TensorHmxUnpackOp::getLoopIteratorTypes() {
  if (cast<RankedTensorType>(getSource().getType()).getRank() != 5) {
    return {};
  }
  return {utils::IteratorType::parallel, utils::IteratorType::parallel};
}

/// Allows producer fusion only through complete physical source tiles.
bool TensorHmxUnpackOp::isOpFusableWithProducerSlices(
    ArrayRef<unsigned> operandNumbers,
    ArrayRef<SmallVector<OpFoldResult>> allOffsets,
    ArrayRef<SmallVector<OpFoldResult>> allSizes) {
  if (operandNumbers.size() != allOffsets.size() ||
      operandNumbers.size() != allSizes.size()) {
    return false;
  }
  // Only the packed source can be fused through, so exactly one slice is
  // expected; the destination is filled in by the fusion driver.
  if (operandNumbers.size() != 1 || operandNumbers[0] != 0) {
    return false;
  }
  if (!hasTileAlignedLogicalDest(cast<RankedTensorType>(getDest().getType()))) {
    return false;
  }
  return isSingleFullHmxTileSlice(cast<RankedTensorType>(getSource().getType()),
                                  allOffsets[0], allSizes[0]);
}

/// Materializes one packed source tile and its corresponding, possibly partial,
/// logical destination tile.
FailureOr<TilingResult>
TensorHmxUnpackOp::getTiledImplementation(OpBuilder &builder,
                                          ArrayRef<OpFoldResult> offsets,
                                          ArrayRef<OpFoldResult> sizes) {
  if (offsets.size() != 2 || sizes.size() != 2) {
    return failure();
  }

  auto sourceType = cast<RankedTensorType>(getSource().getType());
  // Only a rank-5 source has an outer grid that can be tiled further. The
  // verifier also permits a terminal rank-3 physical tile.
  if (sourceType.getRank() != 5) {
    return failure();
  }
  if (auto mTileSize = getConstantIntValue(sizes[0]);
      !mTileSize || *mTileSize != 1) {
    return failure();
  }
  if (auto nTileSize = getConstantIntValue(sizes[1]);
      !nTileSize || *nTileSize != 1) {
    return failure();
  }

  Location loc = getLoc();
  OpFoldResult zero = builder.getIndexAttr(0);
  OpFoldResult one = builder.getIndexAttr(1);
  SmallVector<OpFoldResult> sourceStrides(5, one);

  OpFoldResult innerRows = tensor::getMixedSize(builder, loc, getSource(), 2);
  OpFoldResult innerCols = tensor::getMixedSize(builder, loc, getSource(), 3);
  OpFoldResult innerSpatial =
      tensor::getMixedSize(builder, loc, getSource(), 4);
  auto sourceTileType =
      RankedTensorType::get({kHmxTileRows, kHmxTileColumns, kHmxTileInterleave},
                            sourceType.getElementType());
  auto sourceSlice = tensor::ExtractSliceOp::create(
      builder, loc, sourceTileType, getSource(),
      SmallVector<OpFoldResult>{offsets[0], offsets[1], zero, zero, zero},
      SmallVector<OpFoldResult>{one, one, innerRows, innerCols, innerSpatial},
      sourceStrides);

  OpFoldResult logicalMOffset = multiplyByHmxTileSize(builder, loc, offsets[0]);
  OpFoldResult logicalNOffset = multiplyByHmxTileSize(builder, loc, offsets[1]);
  SmallVector<OpFoldResult> destOffsets = {logicalMOffset, logicalNOffset};
  SmallVector<OpFoldResult> destSizes =
      getClippedHmxTileSizes(builder, loc, getDest(), destOffsets);
  SmallVector<OpFoldResult> destStrides(2, one);
  auto destSlice = tensor::ExtractSliceOp::create(
      builder, loc, getDest(), destOffsets, destSizes, destStrides);

  auto tiledOp = TensorHmxUnpackOp::create(builder, loc, destSlice.getType(),
                                           sourceSlice.getResult(),
                                           destSlice.getResult(), getDimAttr());
  return TilingResult{
      {tiledOp},
      SmallVector<Value>(tiledOp->getResults()),
      {sourceSlice, destSlice},
  };
}

/// Builds the consumer tile that uses the supplied fused producer tile.
FailureOr<TilingResult>
TensorHmxUnpackOp::getTiledImplementationFromOperandTiles(
    OpBuilder &builder, ArrayRef<unsigned> operandNumbers,
    ArrayRef<SmallVector<OpFoldResult>> allOffsets,
    ArrayRef<SmallVector<OpFoldResult>> allSizes) {
  SmallVector<OpFoldResult> iterDomainOffsets;
  SmallVector<OpFoldResult> iterDomainSizes;
  if (failed(getIterationDomainTileFromOperandTiles(
          builder, operandNumbers, allOffsets, allSizes, iterDomainOffsets,
          iterDomainSizes))) {
    return failure();
  }

  Location loc = getLoc();
  OpFoldResult logicalMOffset =
      multiplyByHmxTileSize(builder, loc, iterDomainOffsets[0]);
  OpFoldResult logicalNOffset =
      multiplyByHmxTileSize(builder, loc, iterDomainOffsets[1]);
  SmallVector<OpFoldResult> destOffsets = {logicalMOffset, logicalNOffset};
  SmallVector<OpFoldResult> destSizes =
      getClippedHmxTileSizes(builder, loc, getDest(), destOffsets);
  OpFoldResult one = builder.getIndexAttr(1);
  SmallVector<OpFoldResult> destStrides(2, one);
  auto destSlice = tensor::ExtractSliceOp::create(
      builder, loc, getDest(), destOffsets, destSizes, destStrides);

  Type resultType = destSlice.getType();
  // Keep the original source operand here. The SCF consumer-fusion utility
  // replaces this operand with the already-computed producer tile after this
  // hook returns.
  auto tiledOp =
      TensorHmxUnpackOp::create(builder, loc, resultType, getSource(),
                                destSlice.getResult(), getDimAttr());
  return TilingResult{
      {tiledOp},
      SmallVector<Value>(tiledOp->getResults()),
      {destSlice},
  };
}

/// Projects a complete packed source tile onto the (M, N) iteration grid.
LogicalResult TensorHmxUnpackOp::getIterationDomainTileFromOperandTiles(
    OpBuilder &builder, ArrayRef<unsigned> operandNumbers,
    ArrayRef<SmallVector<OpFoldResult>> allOffsets,
    ArrayRef<SmallVector<OpFoldResult>> allSizes,
    SmallVectorImpl<OpFoldResult> &iterDomainOffsets,
    SmallVectorImpl<OpFoldResult> &iterDomainSizes) {
  if (operandNumbers.size() != allOffsets.size() ||
      operandNumbers.size() != allSizes.size()) {
    return failure();
  }
  // The packed source is the only operand that can carry a fused producer
  // tile, so a well-formed request names exactly one slice of operand 0.
  if (operandNumbers.size() != 1 || operandNumbers[0] != 0) {
    return failure();
  }
  if (!hasTileAlignedLogicalDest(cast<RankedTensorType>(getDest().getType()))) {
    return failure();
  }

  auto sourceType = cast<RankedTensorType>(getSource().getType());
  if (!isSingleFullHmxTileSlice(sourceType, allOffsets[0], allSizes[0])) {
    return failure();
  }

  iterDomainOffsets.assign({allOffsets[0][0], allOffsets[0][1]});
  iterDomainSizes.assign({allSizes[0][0], allSizes[0][1]});
  return success();
}

/// Maps an (M, N) iteration tile to its in-bounds logical result coordinates.
LogicalResult TensorHmxUnpackOp::getResultTilePosition(
    OpBuilder &builder, unsigned resultNumber, ArrayRef<OpFoldResult> offsets,
    ArrayRef<OpFoldResult> sizes, SmallVector<OpFoldResult> &resultOffsets,
    SmallVector<OpFoldResult> &resultSizes) {
  if (resultNumber != 0 || offsets.size() != 2 || sizes.size() != 2) {
    return failure();
  }

  // The verifier ties the result to the rank-2 destination, so the extents are
  // read from the `dest` operand rather than from this op's own result.
  Location loc = getLoc();
  resultOffsets = {multiplyByHmxTileSize(builder, loc, offsets[0]),
                   multiplyByHmxTileSize(builder, loc, offsets[1])};
  resultSizes = getClippedHmxTileSizes(builder, loc, getDest(), resultOffsets);
  return success();
}

/// Maps a logical result tile back to the packed source tile grid.
FailureOr<TilingResult> TensorHmxUnpackOp::generateResultTileValue(
    OpBuilder &builder, unsigned resultNumber, ArrayRef<OpFoldResult> offsets,
    ArrayRef<OpFoldResult> sizes) {
  if (resultNumber != 0 || offsets.size() != 2 || sizes.size() != 2) {
    return failure();
  }

  auto resultType = cast<RankedTensorType>(getResult().getType());
  if (!isSingleLogicalHmxResultTile(resultType, offsets, sizes)) {
    return failure();
  }

  Location loc = getLoc();
  AffineExpr d0;
  bindDims(builder.getContext(), d0);
  auto toTileOffset = [&](OpFoldResult value) {
    return affine::makeComposedFoldedAffineApply(
        builder, loc, d0.floorDiv(kHmxLogicalTileSize), value);
  };

  SmallVector<OpFoldResult> iterDomainOffsets = {toTileOffset(offsets[0]),
                                                 toTileOffset(offsets[1])};
  SmallVector<OpFoldResult> iterDomainSizes = {builder.getIndexAttr(1),
                                               builder.getIndexAttr(1)};
  return getTiledImplementation(builder, iterDomainOffsets, iterDomainSizes);
}

} // namespace mlir::iree_compiler::IREE::Hexagon

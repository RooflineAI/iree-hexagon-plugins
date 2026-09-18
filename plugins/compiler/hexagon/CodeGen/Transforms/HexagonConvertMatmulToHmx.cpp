// Copyright 2026 RooflineAI GmbH
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Converts eligible f16 linalg.matmul operations to tensor-level HMX layout
// and compute operations. This pass runs after VTCM tiling and before LLVMCPU
// inner tiling
//
// Each operand is packed into a grid of complete 32x32 HMX tiles. Within a
// tile, the row dimension is interleaved as:
//   packed[..., p, c, s] = source[..., 2p + s, c]
//
// Static upper bounds determine the physical tile grids, so ragged and bounded
// dynamic shapes are accepted. Runtime pack kernels zero-fill partial tiles;
// the unpack preserves the logical output shape. The unpack receives a
// two-dimensional lowering config and becomes the root that tiles and fuses the
// HMX matmul producer.

#include "hexagon/CodeGen/IR/HexagonOps.h"
#include "hexagon/CodeGen/IR/HmxContracts.h"
#include "hexagon/CodeGen/Passes.h"

#include "iree/compiler/Codegen/Dialect/CPU/IR/IREECPUTypes.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/IR/LinalgInterfaces.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/ValueBoundsOpInterface.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/SmallPtrSet.h"

#include <optional>

namespace mlir::iree_compiler::hexagon::codegen {

#define GEN_PASS_DEF_HEXAGONCONVERTMATMULTOHMXPASS
#include "hexagon/CodeGen/Passes.h.inc"

namespace {

namespace linalg = mlir::linalg;

// Rounds `dim` up to the next multiple of the 32x32x32 HMX tile size.
static int64_t alignToHmxTile(int64_t dim) {
  return IREE::Hexagon::getHmxElementCapacity(
      IREE::Hexagon::getHmxTileCount(dim));
}

// Static upper bound of `value`'s dimension `dim`, rounded up to a whole HMX
// tile. For a static dim this is just the dim rounded up; for a dynamic dim
// (e.g. a VTCM boundary tile bounded by an affine.min) it uses the analyzable
// upper bound. This is the physical size of the padded tile-major buffer; the
// runtime only ever fills the valid `value`-sized region of it.
static FailureOr<int64_t> alignedUpperBound(Value value, int64_t dim) {
  // Look through VTCM staging markers: `stage_to_vtcm` preserves the operand
  // shape (AllTypesMatch) but is an opaque op with no ValueBoundsOpInterface,
  // so the bound analysis cannot see the `affine.min` tile bound on the staged
  // slice through it. Its source carries the same (analyzable) bound.
  while (auto stage = value.getDefiningOp<IREE::Hexagon::StageToVTCMOp>()) {
    value = stage.getSource();
  }
  FailureOr<int64_t> ub = ValueBoundsConstraintSet::computeConstantBound(
      presburger::BoundType::UB, {value, dim}, /*stopCondition=*/nullptr,
      ValueBoundsOptions{/*closedUB=*/true});
  if (failed(ub)) {
    return failure();
  }
  return alignToHmxTile(*ub);
}

// Matches the element types and contraction structure that the HMX path can
// handle, independent of how many batch dimensions are present: two f16 inputs,
// an f16 or f32 output, and exactly one M, N, and K iterator.
static FailureOr<linalg::ContractionDimensions>
matchHmxContractionShape(linalg::LinalgOp linalgOp) {
  if (!linalgOp.hasPureTensorSemantics() || linalgOp.getNumDpsInputs() != 2 ||
      linalgOp.getNumDpsInits() != 1) {
    return failure();
  }
  for (Value input : linalgOp.getDpsInputs()) {
    if (!getElementTypeOrSelf(input.getType()).isF16()) {
      return failure();
    }
  }
  Type outputElementType =
      getElementTypeOrSelf(linalgOp.getDpsInitOperand(0)->get().getType());
  if (!outputElementType.isF16() && !outputElementType.isF32()) {
    return failure();
  }

  FailureOr<linalg::ContractionDimensions> contractionDims =
      linalg::inferContractionDims(linalgOp);
  if (failed(contractionDims) || contractionDims->m.size() != 1 ||
      contractionDims->n.size() != 1 || contractionDims->k.size() != 1) {
    return failure();
  }
  return contractionDims;
}

// Return the operand tensor dimension that carries iterator `iterDim`, or -1 if
// the iterator does not appear (as a plain dim) in the operand's indexing map.
static int64_t operandDimForIterator(linalg::LinalgOp linalgOp,
                                     OpOperand *operand, unsigned iterDim) {
  AffineMap map = linalgOp.getMatchingIndexingMap(operand);
  for (unsigned i = 0, e = map.getNumResults(); i < e; ++i) {
    if (auto dim = dyn_cast<AffineDimExpr>(map.getResult(i))) {
      if (dim.getPosition() == iterDim) {
        return static_cast<int64_t>(i);
      }
    }
  }
  return -1;
}

enum class VtcmProvenance { StagedValue, EmptyDestination };

struct VtcmProvenanceResult {
  std::optional<VtcmProvenance> provenance;
  Operation *unsupportedCarrier = nullptr;
};

/// Conservatively proves the VTCM marker that owns a tensor value. Every
/// terminal path must resolve to the same marker. Loop recurrences are allowed
/// only through operations whose result is tied to the same destination. This
/// is stricter than a generic reverse alias walk because both a loop's initial
/// value and every yielded recurrence must preserve the marker.
class VtcmProvenanceAnalysis {
public:
  VtcmProvenanceResult get(Value value) {
    Summary summary = trace(value);
    VtcmProvenanceResult result;
    result.unsupportedCarrier = summary.unsupportedCarrier;
    // Exactly one marker kind must reach every terminal path. Seeing both
    // stage_to_vtcm and vtcm_empty is conflicting provenance.
    if (!summary.unknown && summary.sawStaged != summary.sawEmpty) {
      result.provenance = summary.sawStaged ? VtcmProvenance::StagedValue
                                            : VtcmProvenance::EmptyDestination;
    }
    return result;
  }

private:
  struct Summary {
    bool sawStaged = false;
    bool sawEmpty = false;
    bool unknown = false;
    Operation *unsupportedCarrier = nullptr;
  };

  static Summary unknown(Operation *carrier = nullptr) {
    return {.unknown = true, .unsupportedCarrier = carrier};
  }

  static void merge(Summary &into, const Summary &other) {
    into.sawStaged |= other.sawStaged;
    into.sawEmpty |= other.sawEmpty;
    into.unknown |= other.unknown;
    if (!into.unsupportedCarrier) {
      into.unsupportedCarrier = other.unsupportedCarrier;
    }
  }

  Summary traceForCarriedValue(scf::ForOp forOp, unsigned index) {
    Summary summary = trace(forOp.getInitArgs()[index]);
    auto yield = cast<scf::YieldOp>(forOp.getBody()->getTerminator());
    merge(summary, trace(yield.getResults()[index]));
    return summary;
  }

  Summary trace(Value value) {
    if (!value) {
      return unknown();
    }
    if (!active.insert(value).second) {
      // A recurrence edge contributes no new terminal. Its initialization path
      // must still provide a marker before the overall proof can succeed.
      return {};
    }
    Summary result = traceImpl(value);
    active.erase(value);
    return result;
  }

  Summary traceImpl(Value value) {
    if (value.getDefiningOp<IREE::Hexagon::StageToVTCMOp>()) {
      return {.sawStaged = true};
    }
    if (value.getDefiningOp<IREE::Hexagon::VTCMEmptyOp>()) {
      return {.sawEmpty = true};
    }
    if (auto collapse = value.getDefiningOp<tensor::CollapseShapeOp>()) {
      return trace(collapse.getSrc());
    }
    if (auto expand = value.getDefiningOp<tensor::ExpandShapeOp>()) {
      return trace(expand.getSrc());
    }
    if (auto slice = value.getDefiningOp<tensor::ExtractSliceOp>()) {
      return trace(slice.getSource());
    }
    if (auto castOp = value.getDefiningOp<tensor::CastOp>()) {
      return trace(castOp.getSource());
    }

    if (auto result = dyn_cast<OpResult>(value)) {
      if (auto dps = dyn_cast<DestinationStyleOpInterface>(result.getOwner())) {
        OpOperand *tied = dps.getTiedOpOperand(result);
        return tied ? trace(tied->get()) : unknown(result.getOwner());
      }
      if (auto forOp = dyn_cast<scf::ForOp>(result.getOwner())) {
        return traceForCarriedValue(forOp, result.getResultNumber());
      }
      if (auto forallOp = dyn_cast<scf::ForallOp>(result.getOwner())) {
        OpOperand *tied = forallOp.getTiedOpOperand(result);
        return tied ? trace(tied->get()) : unknown(result.getOwner());
      }
      return unknown(result.getOwner());
    }

    auto blockArg = dyn_cast<BlockArgument>(value);
    if (!blockArg) {
      return unknown();
    }
    Operation *parentOp = blockArg.getOwner()->getParentOp();
    if (auto forOp = dyn_cast_or_null<scf::ForOp>(parentOp)) {
      auto iterArgs = forOp.getRegionIterArgs();
      auto it = llvm::find(iterArgs, blockArg);
      if (it == iterArgs.end()) {
        return unknown(parentOp);
      }
      unsigned index = it - iterArgs.begin();
      return traceForCarriedValue(forOp, index);
    }
    if (auto forallOp = dyn_cast_or_null<scf::ForallOp>(parentOp)) {
      OpOperand *tied = forallOp.getTiedOpOperand(blockArg);
      return tied ? trace(tied->get()) : unknown(parentOp);
    }
    return unknown(parentOp);
  }

  llvm::SmallPtrSet<Value, 16> active;
};

struct HmxMatmulInfo {
  linalg::MatmulOp op;
  int64_t lhsInterleaveDim;
  int64_t rhsInterleaveDim;
  int64_t outputInterleaveDim;
  int64_t mUpperBound;
  int64_t nUpperBound;
  int64_t kUpperBound;
};

// Analyzes a named matmul. An empty optional means that the operation is not an
// HMX candidate. A failure means that an HMX candidate violates a pipeline
// invariant and conversion cannot safely continue.
static FailureOr<std::optional<HmxMatmulInfo>>
analyzeHmxMatmul(linalg::MatmulOp matmulOp) {
  FailureOr<linalg::ContractionDimensions> contractionDims =
      matchHmxContractionShape(matmulOp);
  if (failed(contractionDims) || !contractionDims->batch.empty()) {
    return std::optional<HmxMatmulInfo>{};
  }

  auto outputType = dyn_cast<RankedTensorType>(
      matmulOp.getDpsInitOperand(0)->get().getType());
  if (!outputType || outputType.getRank() != 2) {
    return std::optional<HmxMatmulInfo>{};
  }

  unsigned mDim = contractionDims->m.front();
  unsigned kDim = contractionDims->k.front();
  OpOperand *lhsOperand = matmulOp.getDpsInputOperand(0);
  OpOperand *rhsOperand = matmulOp.getDpsInputOperand(1);
  OpOperand *outputOperand = matmulOp.getDpsInitOperand(0);
  int64_t lhsInterleaveDim = operandDimForIterator(matmulOp, lhsOperand, mDim);
  int64_t rhsInterleaveDim = operandDimForIterator(matmulOp, rhsOperand, kDim);
  int64_t outputInterleaveDim =
      operandDimForIterator(matmulOp, outputOperand, mDim);
  if (lhsInterleaveDim < 0 || lhsInterleaveDim > 1 || rhsInterleaveDim < 0 ||
      rhsInterleaveDim > 1 || outputInterleaveDim != 0) {
    matmulOp.emitWarning("unsupported HMX matmul operand layout (rank-2 "
                         "operands with a canonical MxN output required)");
    return failure();
  }

  // HMX layout conversion and its runtime kernels require every operand to be
  // VTCM-resident. This is an early tensor-level pipeline check: rewrite only
  // values whose VTCM staging marker is preserved through supported tensor and
  // loop carriers. HexagonLowerHmxToCalls separately validates the concrete
  // memref memory space, layout, and alignment after bufferization at the
  // runtime ABI boundary. Inputs must originate at stage_to_vtcm; the
  // destination may pass through destination-style operations, but must
  // originate at vtcm_empty.
  VtcmProvenanceAnalysis provenanceAnalysis;
  auto checkProvenance = [&](Value value, VtcmProvenance expected,
                             StringRef role) -> LogicalResult {
    VtcmProvenanceResult result = provenanceAnalysis.get(value);
    if (result.provenance == expected) {
      return success();
    }
    InFlightDiagnostic diagnostic = matmulOp.emitError()
                                    << "could not prove " << role
                                    << " has the required VTCM provenance";
    if (result.unsupportedCarrier) {
      diagnostic.attachNote(result.unsupportedCarrier->getLoc())
          << "provenance stopped at unsupported carrier operation '"
          << result.unsupportedCarrier->getName() << "'";
    } else {
      diagnostic << "; all provenance paths must resolve to "
                 << (expected == VtcmProvenance::StagedValue
                         ? "iree_hexagon.stage_to_vtcm"
                         : "iree_hexagon.vtcm_empty");
    }
    return failure();
  };
  if (failed(checkProvenance(lhsOperand->get(), VtcmProvenance::StagedValue,
                             "lhs")) ||
      failed(checkProvenance(rhsOperand->get(), VtcmProvenance::StagedValue,
                             "rhs")) ||
      failed(checkProvenance(outputOperand->get(),
                             VtcmProvenance::EmptyDestination,
                             "output initializer"))) {
    return failure();
  }

  Value lhs = lhsOperand->get();
  Value rhs = rhsOperand->get();
  FailureOr<int64_t> mUpperBound = alignedUpperBound(lhs, lhsInterleaveDim);
  FailureOr<int64_t> kUpperBound = alignedUpperBound(lhs, 1 - lhsInterleaveDim);
  FailureOr<int64_t> nUpperBound = alignedUpperBound(rhs, 1 - rhsInterleaveDim);
  if (failed(mUpperBound) || failed(kUpperBound) || failed(nUpperBound)) {
    matmulOp.emitWarning("could not bound HMX matmul shapes for packing");
    return failure();
  }
  if (*mUpperBound <= 0 || *nUpperBound <= 0 || *kUpperBound <= 0) {
    matmulOp.emitError("HMX matmul requires positive bounded M, N, and K");
    return failure();
  }

  return std::optional<HmxMatmulInfo>{HmxMatmulInfo{
      matmulOp, lhsInterleaveDim, rhsInterleaveDim, outputInterleaveDim,
      *mUpperBound, *nUpperBound, *kUpperBound}};
}

// True for a named f16 batch matmul that would be HMX-eligible after removing
// its batch dimensions. The batch dim should have been tiled to 1 before this
// pass so the rank-reducing patterns above could collapse it to a plain matmul;
// a leftover batch dim means that tiling did not happen.
static bool isUntiledBatchedHmxMatmul(linalg::LinalgOp linalgOp) {
  if (!isa<linalg::BatchMatmulOp>(linalgOp.getOperation())) {
    return false;
  }
  FailureOr<linalg::ContractionDimensions> contractionDims =
      matchHmxContractionShape(linalgOp);
  if (failed(contractionDims) || contractionDims->batch.empty()) {
    return false;
  }

  SmallVector<int64_t> loopRanges = linalgOp.getStaticLoopRanges();
  for (unsigned dim : {contractionDims->m.front(), contractionDims->n.front(),
                       contractionDims->k.front()}) {
    // Ragged (non-multiple-of-32) static dims are accepted; the pack pass pads
    // them up to a whole tile. Only dynamic dims are rejected for now.
    if (dim >= loopRanges.size() || loopRanges[dim] == ShapedType::kDynamic) {
      return false;
    }
  }

  return true;
}

static IREE::CPU::LoweringConfigAttr
getHmxUnpackLoweringConfig(MLIRContext *context) {
  llvm::SmallVector<NamedAttribute> items;
  // Keep the same root-anchor marker used by the CPU double-tiling pipeline,
  // but expressed in the HMX unpack domain: [m_tile, n_tile].
  items.emplace_back(
      IREE::CPU::getTilingLevelName(IREE::CPU::TilingLevel::DistributionTiles),
      IREE::CPU::LoweringConfigAttr::getTilingLevelAttr(context, {0, 0}));
  items.emplace_back(
      IREE::CPU::getTilingLevelName(
          IREE::CPU::TilingLevel::VectorCommonParallelTiles),
      IREE::CPU::LoweringConfigAttr::getTilingLevelAttr(context, {1, 1}));
  return IREE::CPU::LoweringConfigAttr::get(context, items);
}

// Returns reassociation indices for collapsing/expanding a tensor of rank
// `rank` at position `pos`.
static SmallVector<ReassociationIndices>
getReassociationForReshapeAtDim(int64_t rank, int64_t pos) {
  SmallVector<ReassociationIndices> reassociation(rank - 1, {0, 1});
  bool lastDim = pos == rank - 1;
  if (rank > 2) {
    for (int64_t i = 0; i < rank - 1; i++) {
      if (i == pos || (lastDim && i == pos - 1))
        reassociation[i] = ReassociationIndices{i, i + 1};
      else if (i < pos)
        reassociation[i] = ReassociationIndices{i};
      else
        reassociation[i] = ReassociationIndices{i + 1};
    }
  }
  return reassociation;
}

static FailureOr<Value> collapseSingletonDimAt(PatternRewriter &rewriter,
                                               Value value, int64_t pos) {
  auto type = dyn_cast<RankedTensorType>(value.getType());
  if (!type || pos < 0 || pos >= type.getRank() || type.getDimSize(pos) != 1) {
    return failure();
  }

  SmallVector<int64_t> collapsedShape(type.getShape());
  collapsedShape.erase(collapsedShape.begin() + pos);
  auto collapsedType =
      RankedTensorType::get(collapsedShape, type.getElementType());
  return tensor::CollapseShapeOp::create(
             rewriter, value.getLoc(), collapsedType, value,
             getReassociationForReshapeAtDim(type.getRank(), pos))
      .getResult();
}

static FailureOr<AffineMap> dropUnitBatchDim(AffineMap map,
                                             unsigned batchIteratorDim,
                                             int64_t operandBatchDim) {
  if (map.getNumSymbols() != 0 || map.getNumDims() == 0 ||
      operandBatchDim < 0 ||
      operandBatchDim >= static_cast<int64_t>(map.getNumResults())) {
    return failure();
  }

  auto batchExpr = dyn_cast<AffineDimExpr>(map.getResult(operandBatchDim));
  if (!batchExpr || batchExpr.getPosition() != batchIteratorDim) {
    return failure();
  }

  MLIRContext *context = map.getContext();
  SmallVector<AffineExpr> dimReplacements;
  dimReplacements.reserve(map.getNumDims());
  for (unsigned i = 0, e = map.getNumDims(); i < e; ++i) {
    if (i < batchIteratorDim) {
      dimReplacements.push_back(getAffineDimExpr(i, context));
    } else if (i > batchIteratorDim) {
      dimReplacements.push_back(getAffineDimExpr(i - 1, context));
    } else {
      // This replacement must not be used by any remaining result expression:
      // the batch result is dropped below.
      dimReplacements.push_back(getAffineDimExpr(0, context));
    }
  }

  SmallVector<AffineExpr> results;
  results.reserve(map.getNumResults() - 1);
  for (auto [i, expr] : llvm::enumerate(map.getResults())) {
    if (static_cast<int64_t>(i) == operandBatchDim) {
      continue;
    }
    if (expr.isFunctionOfDim(batchIteratorDim)) {
      return failure();
    }
    results.push_back(expr.replaceDimsAndSymbols(dimReplacements, {}));
  }

  return AffineMap::get(map.getNumDims() - 1, /*symbolCount=*/0, results,
                        context);
}

// MLIR's generic contraction rank-reduction rejects named contractions with
// user-defined maps. The HMX path supports rank-2 transpose-a/transpose-b
// matmuls, so locally rank-reduce the common unit-batch batch_matmul case while
// preserving the non-batch maps.
struct RankReduceUnitBatchMatmulWithUserMaps final
    : public OpRewritePattern<linalg::BatchMatmulOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::BatchMatmulOp batchMatmulOp,
                                PatternRewriter &rewriter) const override {
    if (!batchMatmulOp.hasPureTensorSemantics() ||
        batchMatmulOp.getNumDpsInputs() != 2 ||
        batchMatmulOp.getNumDpsInits() != 1 ||
        batchMatmulOp.getResultTensors().size() != 1 ||
        !batchMatmulOp.hasUserDefinedMaps()) {
      return failure();
    }

    FailureOr<linalg::ContractionDimensions> contractionDims =
        linalg::inferContractionDims(batchMatmulOp);
    if (failed(contractionDims) || contractionDims->batch.size() != 1 ||
        contractionDims->m.size() != 1 || contractionDims->n.size() != 1 ||
        contractionDims->k.size() != 1) {
      return failure();
    }

    unsigned batchDim = contractionDims->batch.front();
    OpOperand *lhsOperand = batchMatmulOp.getDpsInputOperand(0);
    OpOperand *rhsOperand = batchMatmulOp.getDpsInputOperand(1);
    OpOperand *accOperand = batchMatmulOp.getDpsInitOperand(0);
    SmallVector<OpOperand *> operands = {lhsOperand, rhsOperand, accOperand};

    SmallVector<int64_t> operandBatchDims;
    operandBatchDims.reserve(operands.size());
    for (OpOperand *operand : operands) {
      int64_t operandBatchDim =
          operandDimForIterator(batchMatmulOp, operand, batchDim);
      if (operandBatchDim < 0) {
        return failure();
      }
      auto type = dyn_cast<RankedTensorType>(operand->get().getType());
      if (!type || type.getDimSize(operandBatchDim) != 1) {
        return failure();
      }
      operandBatchDims.push_back(operandBatchDim);
    }

    // Validate every indexing map before creating reshape operations. Pattern
    // rewrites must not mutate the IR and then return failure: another pattern
    // may continue from the partially rewritten operation.
    SmallVector<AffineMap> collapsedMaps;
    collapsedMaps.reserve(3);
    for (auto [map, dim] : llvm::zip_equal(batchMatmulOp.getIndexingMapsArray(),
                                           operandBatchDims)) {
      FailureOr<AffineMap> collapsedMap = dropUnitBatchDim(map, batchDim, dim);
      if (failed(collapsedMap)) {
        return failure();
      }
      collapsedMaps.push_back(*collapsedMap);
    }

    SmallVector<Value> collapsedOperands;
    collapsedOperands.reserve(operands.size());
    for (auto [operand, dim] : llvm::zip_equal(operands, operandBatchDims)) {
      // Operand types and singleton dimensions were checked above, so these
      // reshapes cannot fail after mutation begins.
      collapsedOperands.push_back(
          *collapseSingletonDimAt(rewriter, operand->get(), dim));
    }

    Location loc = batchMatmulOp.getLoc();
    auto collapsedInitType =
        cast<RankedTensorType>(collapsedOperands[2].getType());
    auto matmulOp = linalg::MatmulOp::create(
        rewriter, loc, TypeRange{collapsedInitType},
        ValueRange{collapsedOperands[0], collapsedOperands[1]},
        ValueRange{collapsedOperands[2]});
    matmulOp.setIndexingMapsAttr(rewriter.getAffineMapArrayAttr(collapsedMaps));
    for (NamedAttribute attr : batchMatmulOp->getAttrs()) {
      if (attr.getName() ==
              linalg::LinalgDialect::kMemoizedIndexingMapsAttrName ||
          attr.getName() == "indexing_maps") {
        continue;
      }
      matmulOp->setAttr(attr.getName(), attr.getValue());
    }

    Value result = matmulOp.getResultTensors().front();
    auto expandedType = cast<RankedTensorType>(
        batchMatmulOp.getResultTensors().front().getType());
    Value expanded = tensor::ExpandShapeOp::create(
        rewriter, loc, expandedType, result,
        getReassociationForReshapeAtDim(expandedType.getRank(),
                                        operandBatchDims[2]));
    rewriter.replaceOp(batchMatmulOp, expanded);
    return success();
  }
};

// `rows` and `cols` are already whole multiples of the logical tile size, so
// the tile counts divide exactly.
static RankedTensorType getHmxTileGridType(int64_t rows, int64_t cols,
                                           Type elementType) {
  return RankedTensorType::get(
      {IREE::Hexagon::getHmxTileCount(rows),
       IREE::Hexagon::getHmxTileCount(cols), IREE::Hexagon::kHmxTileRows,
       IREE::Hexagon::kHmxTileColumns, IREE::Hexagon::kHmxTileInterleave},
      elementType);
}

static void rewriteMatmulToHmx(IRRewriter &rewriter,
                               const HmxMatmulInfo &info) {
  linalg::MatmulOp matmulOp = info.op;
  Location loc = matmulOp.getLoc();
  rewriter.setInsertionPoint(matmulOp);

  auto createVtcmEmpty = [&](RankedTensorType type) -> Value {
    return IREE::Hexagon::VTCMEmptyOp::create(rewriter, loc, type,
                                              /*dynamicSizes=*/ValueRange{})
        .getResult();
  };
  auto packTiles = [&](Value source, int64_t rows, int64_t cols,
                       int64_t interleaveDim) -> Value {
    auto sourceType = cast<RankedTensorType>(source.getType());
    Value destination = createVtcmEmpty(
        getHmxTileGridType(rows, cols, sourceType.getElementType()));
    return IREE::Hexagon::TensorHmxPackOp::create(
               rewriter, loc, TypeRange{destination.getType()}, source,
               destination, rewriter.getI64IntegerAttr(interleaveDim))
        .getResult();
  };

  Value lhs = matmulOp.getDpsInputOperand(0)->get();
  Value rhs = matmulOp.getDpsInputOperand(1)->get();
  Value output = matmulOp.getDpsInitOperand(0)->get();

  // The packed buffers cover complete 32x32 tiles. Pack kernels zero-fill the
  // parts outside the logical source extents, which supports ragged and bounded
  // dynamic shapes without tensor.pad operations in this pass.
  Value packedLhs =
      packTiles(lhs, info.mUpperBound, info.kUpperBound, info.lhsInterleaveDim);
  Value packedRhs =
      packTiles(rhs, info.kUpperBound, info.nUpperBound, info.rhsInterleaveDim);

  // The v79 target intrinsics and HexKL currently expose only an f16
  // accumulator conversion/read sequence.
  // TODO: Preserve full accumulator precision for f32 outputs.
  Value accumulatorRead = createVtcmEmpty(getHmxTileGridType(
      info.mUpperBound, info.nUpperBound, rewriter.getF16Type()));
  auto hmxMatmul = IREE::Hexagon::TensorHmxMatmulOp::create(
      rewriter, loc, accumulatorRead.getType(), packedLhs, packedRhs,
      accumulatorRead);

  // Preserve the logical output extent while the physical HMX grid remains
  // padded. TensorHmxUnpackOp clips boundary destination slices, and runtime
  // lowering passes those logical row and column counts to the unpack kernel.
  auto unpack = IREE::Hexagon::TensorHmxUnpackOp::create(
      rewriter, loc, TypeRange{output.getType()}, hmxMatmul.getResult(), output,
      rewriter.getI64IntegerAttr(info.outputInterleaveDim));
  setLoweringConfig(unpack, getHmxUnpackLoweringConfig(rewriter.getContext()));

  rewriter.replaceOp(matmulOp, unpack.getResult());
}

struct HexagonConvertMatmulToHmxPass final
    : public impl::HexagonConvertMatmulToHmxPassBase<
          HexagonConvertMatmulToHmxPass> {
  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry
        .insert<linalg::LinalgDialect, tensor::TensorDialect,
                arith::ArithDialect, IREE::Codegen::IREECodegenDialect,
                IREE::CPU::IREECPUDialect, IREE::Hexagon::IREEHexagonDialect>();
  }

  void runOnOperation() override {
    mlir::FunctionOpInterface funcOp = getOperation();

    // Reduce named batch_matmul operations whose batch dimension has already
    // been tiled to one into linalg.matmul. The standard patterns are kept for
    // canonicalization consistency and may also rank-reduce contractions that
    // are not ultimately eligible for HMX; that broader simplification is
    // intentional for now.
    {
      RewritePatternSet patterns(&getContext());
      patterns.add<RankReduceUnitBatchMatmulWithUserMaps>(&getContext());
      linalg::populateContractionOpRankReducingPatterns(patterns);
      if (failed(applyPatternsGreedily(funcOp, std::move(patterns)))) {
        return signalPassFailure();
      }
    }

    // linalg.generic contractions are excluded in this pass. Supporting them
    // later requires validating their region semantics, not only their maps and
    // iterator types.
    SmallVector<linalg::MatmulOp> matmuls;
    funcOp.walk(
        [&](linalg::MatmulOp matmulOp) { matmuls.push_back(matmulOp); });

    // This is checking that the pass is being used the way it is expected
    bool foundUntiledHmxBatchMatmul = false;
    funcOp.walk([&](linalg::LinalgOp linalgOp) {
      if (isUntiledBatchedHmxMatmul(linalgOp)) {
        foundUntiledHmxBatchMatmul = true;
        linalgOp.emitError()
            << "HMX-eligible batch matmul still has a batch dimension; "
               "expected the batch dimension to have been tiled to 1 before "
               "this pass so "
               "it could be rank-reduced to a plain matmul (see the "
               "cache-parallel batch tiling in "
               "addHexagonHmxMatmulExpertPassPipeline)";
      }
    });
    if (foundUntiledHmxBatchMatmul) {
      return signalPassFailure();
    }

    SmallVector<HmxMatmulInfo> eligibleMatmuls;
    for (linalg::MatmulOp matmulOp : matmuls) {
      FailureOr<std::optional<HmxMatmulInfo>> info = analyzeHmxMatmul(matmulOp);
      if (failed(info)) {
        return signalPassFailure();
      }
      if (*info) {
        eligibleMatmuls.push_back(**info);
      }
    }

    IRRewriter rewriter(&getContext());
    for (const HmxMatmulInfo &info : eligibleMatmuls) {
      rewriteMatmulToHmx(rewriter, info);
    }
  }
};

} // namespace

} // namespace mlir::iree_compiler::hexagon::codegen

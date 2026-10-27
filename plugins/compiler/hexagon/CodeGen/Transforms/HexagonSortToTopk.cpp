// Copyright 2026 RooflineAI GmbH
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "hexagon/CodeGen/Passes.h"

#include "iree/compiler/Dialect/HAL/IR/HALTypes.h"
#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtDialect.h"
#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "iree/compiler/Dialect/Util/IR/UtilTypes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::iree_compiler::hexagon::codegen {

#define GEN_PASS_DEF_HEXAGONSORTTOTOPKPASS
#include "hexagon/CodeGen/Passes.h.inc"

namespace {

namespace LinalgExt = IREE::LinalgExt;

// The comparison a topk region is built from: the cmpf/cmpi of the sort
// comparator with its predicate made strict.
struct TopkComparison {
  Operation *cmpOp;
  // Strict form of the predicate, as the arith enum value of cmpOp's kind.
  uint64_t strictPredicate;
  // Whether the comparison selects the largest values (gt) or the smallest.
  bool selectsLargest;
  bool isUnsigned = false;
};

// Matches a comparator that yields `cmp(lhs, rhs)` of the first two block
// arguments (the sorted values). Returns the comparison with the predicate
// made strict, so topk keeps the earlier of two equal values.
static std::optional<TopkComparison> matchComparator(Region &region) {
  Block &block = region.front();
  if (!llvm::hasSingleElement(block.without_terminator()))
    return std::nullopt;
  Operation *cmpOp = &block.front();
  auto yieldOp = cast<LinalgExt::YieldOp>(block.getTerminator());
  if (yieldOp.getNumOperands() != 1 ||
      yieldOp.getOperand(0) != cmpOp->getResult(0) ||
      cmpOp->getNumOperands() != 2 ||
      cmpOp->getOperand(0) != block.getArgument(0) ||
      cmpOp->getOperand(1) != block.getArgument(1))
    return std::nullopt;

  if (auto cmpf = dyn_cast<arith::CmpFOp>(cmpOp)) {
    using P = arith::CmpFPredicate;
    switch (cmpf.getPredicate()) {
    case P::OGT:
    case P::OGE:
      return TopkComparison{cmpOp, static_cast<uint64_t>(P::OGT), true};
    case P::OLT:
    case P::OLE:
      return TopkComparison{cmpOp, static_cast<uint64_t>(P::OLT), false};
    default:
      return std::nullopt;
    }
  }
  if (auto cmpi = dyn_cast<arith::CmpIOp>(cmpOp)) {
    using P = arith::CmpIPredicate;
    switch (cmpi.getPredicate()) {
    case P::sgt:
    case P::sge:
      return TopkComparison{cmpOp, static_cast<uint64_t>(P::sgt), true};
    case P::slt:
    case P::sle:
      return TopkComparison{cmpOp, static_cast<uint64_t>(P::slt), false};
    case P::ugt:
    case P::uge:
      return TopkComparison{cmpOp, static_cast<uint64_t>(P::ugt), true, true};
    case P::ult:
    case P::ule:
      return TopkComparison{cmpOp, static_cast<uint64_t>(P::ult), false, true};
    default:
      return std::nullopt;
    }
  }
  return std::nullopt;
}

// Returns the value every input beats or ties under `comparison`: -inf / the
// minimum integer for gt, +inf / the maximum integer for lt.
static std::optional<TypedAttr> getTopkInitValue(OpBuilder &b, Type elemType,
                                                 TopkComparison comparison) {
  bool negative = comparison.selectsLargest;
  if (auto floatType = dyn_cast<FloatType>(elemType)) {
    const llvm::fltSemantics &semantics = floatType.getFloatSemantics();
    if (!APFloat::semanticsHasInf(semantics))
      return std::nullopt;
    return b.getFloatAttr(floatType, APFloat::getInf(semantics, negative));
  }
  auto intType = dyn_cast<IntegerType>(elemType);
  if (!intType)
    return std::nullopt;
  unsigned width = intType.getWidth();
  APInt init =
      comparison.isUnsigned
          ? (negative ? APInt::getMinValue(width) : APInt::getMaxValue(width))
          : (negative ? APInt::getSignedMinValue(width)
                      : APInt::getSignedMaxValue(width));
  return b.getIntegerAttr(intType, init);
}

// Whether `value` is `linalg.index dim`, cast to an integer, written by an
// otherwise input-free elementwise generic.
static bool isIotaAlongDim(Value value, int64_t dim) {
  auto genericOp = value.getDefiningOp<linalg::GenericOp>();
  if (!genericOp || genericOp.getNumDpsInputs() != 0 ||
      genericOp.getNumDpsInits() != 1 ||
      genericOp.getNumParallelLoops() != genericOp.getNumLoops() ||
      !genericOp.getIndexingMapsArray().front().isIdentity())
    return false;
  auto yieldOp =
      cast<linalg::YieldOp>(genericOp.getRegion().front().getTerminator());
  Value yielded = yieldOp.getOperand(0);
  if (auto castOp = yielded.getDefiningOp<arith::IndexCastOp>())
    yielded = castOp.getIn();
  auto indexOp = yielded.getDefiningOp<linalg::IndexOp>();
  return indexOp && static_cast<int64_t>(indexOp.getDim()) == dim;
}

// Returns k if `sliceOp` takes the leading k elements of its source along
// `dim` and the full extent of every other dimension.
static std::optional<int64_t>
getLeadingSliceSize(tensor::ExtractSliceOp sliceOp, int64_t dim) {
  RankedTensorType sourceType = sliceOp.getSourceType();
  if (sliceOp.getResultType().getRank() != sourceType.getRank() ||
      !llvm::all_of(sliceOp.getStaticOffsets(),
                    [](int64_t offset) { return offset == 0; }) ||
      !llvm::all_of(sliceOp.getStaticStrides(),
                    [](int64_t stride) { return stride == 1; }))
    return std::nullopt;
  ArrayRef<int64_t> sizes = sliceOp.getStaticSizes();
  for (auto [i, size] : llvm::enumerate(sizes)) {
    if (static_cast<int64_t>(i) != dim && size != sourceType.getDimSize(i))
      return std::nullopt;
  }
  if (ShapedType::isDynamic(sizes[dim]))
    return std::nullopt;
  return sizes[dim];
}

// Converts topk's i32 indices to the element type of the sort's iota.
static Value castIndices(OpBuilder &b, Location loc, Value indices,
                         Type elemType) {
  if (elemType.isInteger(32))
    return indices;
  auto indicesType = cast<RankedTensorType>(indices.getType());
  int64_t rank = indicesType.getRank();
  Value empty =
      tensor::EmptyOp::create(b, loc, indicesType.getShape(), elemType);
  SmallVector<AffineMap> maps(2, b.getMultiDimIdentityMap(rank));
  SmallVector<utils::IteratorType> iteratorTypes(rank,
                                                 utils::IteratorType::parallel);
  auto genericOp = linalg::GenericOp::create(
      b, loc, empty.getType(), indices, empty, maps, iteratorTypes,
      [&](OpBuilder &nested, Location nestedLoc, ValueRange args) {
        // Indices are non-negative, so sign extension is exact.
        Value cast =
            elemType.isIndex()
                ? arith::IndexCastOp::create(nested, nestedLoc, elemType,
                                             args[0])
                      .getResult()
                : arith::ExtSIOp::create(nested, nestedLoc, elemType, args[0])
                      .getResult();
        linalg::YieldOp::create(nested, nestedLoc, cast);
      });
  return genericOp.getResult(0);
}

struct SortToTopk : OpRewritePattern<LinalgExt::SortOp> {
  using Base::Base;

  LogicalResult matchAndRewrite(LinalgExt::SortOp sortOp,
                                PatternRewriter &rewriter) const override {
    if (!sortOp.hasPureTensorSemantics())
      return rewriter.notifyMatchFailure(sortOp, "expected tensor semantics");
    int64_t dim = sortOp.getDimension();
    ValueRange outputs = sortOp.getOutputs();
    if (outputs.empty() || outputs.size() > 2)
      return rewriter.notifyMatchFailure(sortOp,
                                         "expected values and optional iota");
    Value values = outputs[0];
    auto valuesType = cast<RankedTensorType>(values.getType());
    if (!valuesType.hasStaticShape())
      return rewriter.notifyMatchFailure(sortOp, "expected a static shape");
    int64_t n = valuesType.getDimSize(dim);
    // topk's implicit indices are i32.
    if (n > std::numeric_limits<int32_t>::max())
      return rewriter.notifyMatchFailure(sortOp, "sort dimension too large");
    if (outputs.size() == 2 && !isIotaAlongDim(outputs[1], dim))
      return rewriter.notifyMatchFailure(sortOp,
                                         "indices are not an iota along dim");

    std::optional<TopkComparison> comparison =
        matchComparator(sortOp.getRegion());
    if (!comparison)
      return rewriter.notifyMatchFailure(sortOp, "unsupported comparator");
    Type elemType = valuesType.getElementType();
    std::optional<TypedAttr> initAttr =
        getTopkInitValue(rewriter, elemType, *comparison);
    if (!initAttr)
      return rewriter.notifyMatchFailure(sortOp, "no init value for type");

    // Every use must read the same leading k elements.
    SmallVector<tensor::ExtractSliceOp> slices;
    std::optional<int64_t> k;
    for (OpOperand &use : sortOp->getUses()) {
      auto sliceOp = dyn_cast<tensor::ExtractSliceOp>(use.getOwner());
      if (!sliceOp)
        return rewriter.notifyMatchFailure(sortOp, "use is not a slice");
      std::optional<int64_t> sliceK = getLeadingSliceSize(sliceOp, dim);
      if (!sliceK || (k && *k != *sliceK))
        return rewriter.notifyMatchFailure(sortOp,
                                           "slices differ or are not leading");
      k = sliceK;
      slices.push_back(sliceOp);
    }
    if (!k || *k >= n)
      return rewriter.notifyMatchFailure(sortOp, "no strict prefix is read");

    Location loc = sortOp.getLoc();
    SmallVector<int64_t> outShape(valuesType.getShape());
    outShape[dim] = *k;
    // Same init as StableHLO's topk lowering: an input that ties the init
    // value still replaces the index, since its index is smaller.
    Value initValue = arith::ConstantOp::create(rewriter, loc, *initAttr);
    Value outValues =
        linalg::FillOp::create(
            rewriter, loc, initValue,
            tensor::EmptyOp::create(rewriter, loc, outShape, elemType)
                .getResult())
            .result();
    Value initIndex = arith::ConstantOp::create(
        rewriter, loc,
        rewriter.getI32IntegerAttr(std::numeric_limits<int32_t>::max()));
    Value outIndices =
        linalg::FillOp::create(rewriter, loc, initIndex,
                               tensor::EmptyOp::create(rewriter, loc, outShape,
                                                       rewriter.getI32Type())
                                   .getResult())
            .result();

    auto topkOp = LinalgExt::TopkOp::create(
        rewriter, loc, TypeRange{outValues.getType(), outIndices.getType()},
        values, /*indices=*/Value(), outValues, outIndices, dim);
    Block *block = rewriter.createBlock(&topkOp.getRegion(), {},
                                        {elemType, elemType}, {loc, loc});
    {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(block);
      Operation *cmpOp = rewriter.clone(*comparison->cmpOp);
      cmpOp->setOperands(block->getArguments());
      if (auto cmpf = dyn_cast<arith::CmpFOp>(cmpOp))
        cmpf.setPredicate(
            static_cast<arith::CmpFPredicate>(comparison->strictPredicate));
      else
        cast<arith::CmpIOp>(cmpOp).setPredicate(
            static_cast<arith::CmpIPredicate>(comparison->strictPredicate));
      LinalgExt::YieldOp::create(rewriter, loc, cmpOp->getResult(0));
    }

    Value topkIndices;
    for (tensor::ExtractSliceOp sliceOp : slices) {
      unsigned resultNumber =
          cast<OpResult>(sliceOp.getSource()).getResultNumber();
      if (resultNumber == 0) {
        rewriter.replaceOp(sliceOp, topkOp.getResult(0));
        continue;
      }
      if (!topkIndices) {
        rewriter.setInsertionPointAfter(topkOp);
        topkIndices = castIndices(
            rewriter, loc, topkOp.getResult(1),
            cast<RankedTensorType>(outputs[1].getType()).getElementType());
      }
      rewriter.replaceOp(sliceOp, topkIndices);
    }
    rewriter.eraseOp(sortOp);
    return success();
  }
};

// Whether the module targets hexagon devices only. Device globals are
// assigned before preprocessing.
static bool targetsOnlyHexagon(ModuleOp moduleOp) {
  bool sawHexagon = false;
  bool sawOther = false;
  for (auto globalOp : moduleOp.getOps<IREE::Util::GlobalOpInterface>()) {
    Attribute initialValue = globalOp.getGlobalInitialValue();
    if (!initialValue)
      continue;
    initialValue.walk([&](IREE::HAL::DeviceTargetAttr targetAttr) {
      if (targetAttr.getDeviceID().getValue() == "hexagon")
        sawHexagon = true;
      else
        sawOther = true;
    });
  }
  return sawHexagon && !sawOther;
}

struct HexagonSortToTopkPass
    : impl::HexagonSortToTopkPassBase<HexagonSortToTopkPass> {
  using Base::Base;

  void runOnOperation() override {
    ModuleOp moduleOp = getOperation();
    if (!targetsOnlyHexagon(moduleOp))
      return;
    RewritePatternSet patterns(&getContext());
    patterns.add<SortToTopk>(&getContext());
    if (failed(applyPatternsGreedily(moduleOp, std::move(patterns))))
      return signalPassFailure();
  }
};

} // namespace
} // namespace mlir::iree_compiler::hexagon::codegen

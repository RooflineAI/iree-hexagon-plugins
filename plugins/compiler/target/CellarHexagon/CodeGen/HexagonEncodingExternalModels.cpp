// Copyright 2025 RooflineAI GmbH
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// TODO: This file currently contains a lot of copy pasted code from LLVMCPU
// needed to lower matmul operations. I did not check that code yet
// TODO: Remove, keep, modify or whatever:
// This is the description of the CPUEncodingExternalModels.cpp file that this
// one is modeled after
//===- CPUEncodingExternalModels.cpp --------------------------------------===//
//
// This file implements the following interfaces for CPU backends and the VMVX
// backend:
//
// - IREE::Encoding::LayoutResolverAttr
// - IREE::Encoding::SerializableAttr
// - IREE::Encoding::LayoutMaterializerAttr
// - IREE::Codegen::PackedLayoutMaterializerAttr
//
// In these backends, we transpose narrow-N into narrow-M
// for a combination of reasons:
//
//   1. As linalg.matmul materializes into linalg.mmt4d, which has a transposed
//      RHS and therefore LHS<->RHS symmetry, transposeNarrowN is easy to
//      implement at that level.
//   2. We use ukernels, and this allows writing 2x fewer narrow ukernels.
//   3. Heuristics for cache-friendly dispatch tiling can get complex on CPU,
//      so it is nice that they have fewer narrow cases to consider.
//
// This transposition is made easier by (and was all along part of the idea in)
// the RHS-transposition in mmt4d (the t in mmt4d), as generally with matrix
// multiplication
//
//   B * Transpose(A) == Transpose( A * Transpose(B) )
//
// so in mmt4d terms
//
//   mmt4d(B, A) == Transpose(mmt4d(A, B))
//
//===---------------------------------------------------------------------===//

#include "target/CellarHexagon/CodeGen/HexagonEncodingExternalModels.h"
#include "target/CellarHexagon/CodeGen/IR/HexagonEncodingAttrs.h"
#include "target/CellarHexagon/CodeGen/IR/HexagonEncodingDialect.h"

#include "iree/compiler/Codegen/Dialect/Codegen/Utils/Utils.h"
#include "iree/compiler/Codegen/ExternalInterfaces/Utils.h"
#include "iree/compiler/Codegen/Utils/Utils.h"
#include "iree/compiler/Dialect/Encoding/IR/EncodingTypes.h"
#include "iree/compiler/Dialect/Encoding/Utils/Utils.h"
#include "mlir/IR/BuiltinAttributes.h"

#define DEBUG_TYPE "iree-hexagon-encoding"

namespace mlir::iree_compiler::hexagon {

using namespace IREE::Hexagon;

namespace {

// Fixed 4x4x4 tiling keeps the prototype simple while still exercising the
// encoding plumbing in the same way as CPU/GPU backends.
constexpr IREE::Codegen::TileMxNxK kHexagonPrototypeTile{4, 4, 4};

// TODO: Copy pasted function, come back to it later. It might not even be
// useful for Hexagon, this probably needs a lot of rethinking
static RankedTensorType
getExpandedType(RankedTensorType type, bool isBatched, bool isTransposed,
                SmallVectorImpl<ReassociationIndices> &ri) {
  if (!isBatched) {
    ri.assign({{0, 1}, {2, 3}});
    if (!isTransposed) {
      return RankedTensorType::get(
          {1, type.getDimSize(0), 1, type.getDimSize(1)},
          type.getElementType());
    }
    return RankedTensorType::get({type.getDimSize(0), 1, type.getDimSize(1), 1},
                                 type.getElementType());
  }

  ri.assign({{0}, {1, 2}, {3, 4}});
  if (!isTransposed) {
    return RankedTensorType::get(
        {type.getDimSize(0), 1, type.getDimSize(1), 1, type.getDimSize(2)},
        type.getElementType());
  }
  return RankedTensorType::get(
      {type.getDimSize(0), type.getDimSize(1), 1, type.getDimSize(2), 1},
      type.getElementType());
}

// TODO: Copy pasted function, come back to it later. It might not even be
// useful for Hexagon, this probably needs a lot of rethinking
static Value createElementWiseExtUIOp(OpBuilder &builder, Value input,
                                      Location loc, Type outElemType) {
  auto inputType = cast<RankedTensorType>(input.getType());
  if (inputType.getElementType() == outElemType) {
    return input;
  }
  SmallVector<AffineMap> maps(
      2, builder.getMultiDimIdentityMap(inputType.getRank()));
  SmallVector<utils::IteratorType> iteratorTypes(inputType.getRank(),
                                                 utils::IteratorType::parallel);
  auto castedType = inputType.clone(outElemType);
  SmallVector<OpFoldResult> inputMixedSizes =
      tensor::getMixedSizes(builder, loc, input);
  Value init =
      tensor::EmptyOp::create(builder, loc, inputMixedSizes, outElemType);
  return builder
      .create<linalg::GenericOp>(
          loc, castedType, input, init, maps, iteratorTypes,
          [&](OpBuilder &b, Location nestedLoc, ValueRange args) {
            Value castRes =
                arith::ExtUIOp::create(b, nestedLoc, outElemType, args[0])
                    ->getResult(0);
            linalg::YieldOp::create(b, nestedLoc, castRes);
          })
      .getResult(0);
}

// TODO: Copy pasted function, come back to it later. It might not even be
// useful for Hexagon, this probably needs a lot of rethinking
static Value getMmt4dOperand(Value value, linalg::LinalgOp linalgOp,
                             bool transpose, OpBuilder &builder,
                             SmallVectorImpl<ReassociationIndices> &ri,
                             ArrayRef<Type> elemTypes, int operandIdx) {
  assert(linalgOp.getNumDpsInputs() == 2);
  assert(linalgOp.getNumDpsInits() == 1);
  auto cDims = linalg::inferContractionDims(linalgOp);
  Location loc = linalgOp->getLoc();
  Value expandedValue = value;
  // If vecmat with non-rhs operandIdx or matvec with non-lhs operandIdx, the
  // operand is a vector and must be extended
  if ((cDims->m.empty() && operandIdx != 1) ||
      (cDims->n.empty() && operandIdx != 0)) {
    auto type = cast<RankedTensorType>(value.getType());
    RankedTensorType newType = getExpandedType(
        type, /*isBatched=*/!cDims->batch.empty(),
        /*isTransposed=*/operandIdx == 2 && (transpose ^ cDims->n.empty()), ri);
    expandedValue =
        tensor::ExpandShapeOp::create(builder, loc, newType, value, ri);
  }
  if (elemTypes[operandIdx].isUnsignedInteger()) {
    return createElementWiseExtUIOp(builder, expandedValue, loc,
                                    elemTypes.back());
  }
  return expandedValue;
}

// TODO: Copy pasted function, come back to it later. It might not even be
// useful for Hexagon, this probably needs a lot of rethinking
FailureOr<Operation *> lowerContractionOpWithEncoding(
    OpBuilder &builder, linalg::LinalgOp linalgOp, ValueRange operands,
    IREE::Encoding::LayoutMaterializerAttr layoutAttr) {

  // Verification that this is one of the operations we actually want to lower.
  if (!linalgOp.hasPureTensorSemantics()) {
    return failure();
  }

  auto inputs = linalgOp.getDpsInputOperands();
  auto outputs = linalgOp.getDpsInits();

  auto lhsType = cast<RankedTensorType>(inputs[0]->get().getType());
  auto rhsType = cast<RankedTensorType>(inputs[1]->get().getType());
  auto resultType = cast<RankedTensorType>(outputs[0].getType());
  auto lhsEncoding = IREE::Encoding::getEncodingAttr(lhsType);
  auto rhsEncoding = IREE::Encoding::getEncodingAttr(rhsType);
  auto resultEncoding = IREE::Encoding::getEncodingAttr(resultType);
  if (!lhsEncoding || !rhsEncoding || !resultEncoding) {
    return failure();
  }

  if (lhsEncoding.getOperandIndex().getValue() != IREE::Encoding::MATMUL_LHS ||
      rhsEncoding.getOperandIndex().getValue() != IREE::Encoding::MATMUL_RHS ||
      resultEncoding.getOperandIndex().getValue() !=
          IREE::Encoding::MATMUL_RESULT) {
    return failure();
  }

  // It is one of the operations we want to lower, let's try to do so.
  IREE::Codegen::MaterializeEncodingInfo encodingInfo = {};
  if (auto packedLayoutAttr =
          dyn_cast<IREE::Codegen::PackedLayoutMaterializerAttr>(layoutAttr)) {
    encodingInfo = packedLayoutAttr.getEncodingInfo(
        cast<RankedTensorType>(linalgOp->getResultTypes()[0]));
  }

  // Small check for identity layout
  if (isIdentityLayout(encodingInfo)) {
    return dropEncodingAndCloneOp(builder, linalgOp,
                                  operands.take_front(inputs.size()),
                                  operands.drop_front(inputs.size()));
  }

  // Lower to a mmt4
  bool transpose = isNarrowNResult(resultEncoding);
  SmallVector<Type> elemTypes = lhsEncoding.getElementTypesArray();
  SmallVector<ReassociationIndices> ri;
  Value newLhs = getMmt4dOperand(operands[0], linalgOp, transpose, builder, ri,
                                 elemTypes, /*operandIdx=*/0);
  Value newRhs = getMmt4dOperand(operands[1], linalgOp, transpose, builder, ri,
                                 elemTypes, /*operandIdx=*/1);
  Value newResult = getMmt4dOperand(operands[2], linalgOp, transpose, builder,
                                    ri, elemTypes, /*operandIdx=*/2);
  if (transpose) {
    std::swap(newLhs, newRhs);
  }
  Type newResultType = newResult.getType();
  auto cDims = IREE::Encoding::getEncodingContractionDims(lhsEncoding);
  Operation *result;
  if (cDims->batch.empty()) {
    result = linalg::Mmt4DOp::create(builder, linalgOp.getLoc(), newResultType,
                                     ValueRange{newLhs, newRhs},
                                     ValueRange{newResult});
  } else {
    result = linalg::BatchMmt4DOp::create(
        builder, linalgOp.getLoc(), newResultType, ValueRange{newLhs, newRhs},
        ValueRange{newResult});
  }
  if (!ri.empty()) {
    result = tensor::CollapseShapeOp::create(builder, linalgOp->getLoc(),
                                             operands[2].getType(),
                                             result->getResult(0), ri);
  }
  return result;
}

/// TODO:
/// This is an attempt at a very simple resolver for tiling. No arch specific
/// choices yet, just setting the skeleton up for it.
/// Most of this file is copy pasted for the time being.
struct HexagonPackedLayoutMaterializerAttr
    : public ::mlir::iree_compiler::IREE::
          PackedLayoutMaterializerAttrExternalModelBase<
              HexagonPackedLayoutMaterializerAttr,
              HexagonEncodingResolverAttr> {

  DictionaryAttr getConfiguration(Attribute attr) const {
    auto resolver = cast<HexagonEncodingResolverAttr>(attr);
    return resolver.getConfiguration();
  }

  IREE::Codegen::MaterializeEncodingInfo
  getEncodingInfoImpl(Attribute attr, RankedTensorType type) const {
    // Mirror the CPU/GPU flow: look at the tensor's encoding and request a
    // materialization recipe. This prototype always selects the 4x4x4 tile.
    (void)attr;
    IREE::Codegen::MaterializeEncodingInfo info;
    auto encoding =
        dyn_cast_or_null<IREE::Encoding::EncodingAttr>(type.getEncoding());
    if (!encoding) {
      return info;
    }

    FailureOr<IREE::Codegen::MaterializeEncodingInfo> maybeInfo =
        IREE::Codegen::getEncodingInfoForMatmul(encoding,
                                                kHexagonPrototypeTile);
    if (failed(maybeInfo)) {
      llvm::errs() << "Unexpected miss when retrieving tiling information\n";
      return info;
    }
    return std::move(maybeInfo.value());
  }
};

// If you are looking at LLVMCPU as reference, note that this one is named
// differently, but I think they just messed up the naming
struct HexagonLayoutMaterializerAttr
    : public ::mlir::iree_compiler::IREE::
          EncodingLayoutMaterializerAttrExternalModelBase<
              HexagonLayoutMaterializerAttr, HexagonEncodingResolverAttr> {
  Operation *lowerOp(Attribute attr, OpBuilder &builder, Operation *op,
                     TypeRange convertedResults,
                     ValueRange convertedOperands) const {
    (void)convertedResults;
    auto linalgOp = dyn_cast<linalg::LinalgOp>(op);
    if (!linalgOp) {
      return nullptr;
    }

    auto layoutAttr = cast<IREE::Encoding::LayoutMaterializerAttr>(attr);
    FailureOr<Operation *> lowered = lowerContractionOpWithEncoding(
        builder, linalgOp, convertedOperands, layoutAttr);
    return succeeded(lowered) ? *lowered : nullptr;
  }
};

struct HexagonLayoutResolverAttr
    : public IREE::Encoding::LayoutResolverAttr::ExternalModel<
          HexagonLayoutResolverAttr, HexagonEncodingResolverAttr> {
  Attribute cloneWithSimplifiedConfig(Attribute attr,
                                      DictionaryAttr config) const {
    (void)config;
    // TODO: filter configuration entries that Hexagon needs once defined.
    // I do not understand this function yet
    return attr;
  }

  Attribute getLayout(Attribute attr, RankedTensorType type) const {
    (void)type;
    // TODO: synthesize layout attributes once tiling strategy is available.
    // I do not understand this function yet
    return attr;
  }
};

struct HexagonSerializableAttr
    : public IREE::Encoding::SerializableAttr::ExternalModel<
          HexagonSerializableAttr, HexagonEncodingResolverAttr> {

  Value calculateStorageSizeInBytes(Attribute attr, Location loc,
                                    OpBuilder &builder, RankedTensorType type,
                                    ValueRange dynamicDims) const {
    // I hope to be able to reuse the builtin functionality for this.
    return IREE::calculatePackedStorageSizeInBytesImpl(attr, loc, builder, type,
                                                       dynamicDims);
  }
};

} // namespace

void registerHexagonEncodingExternalModels(DialectRegistry &registry) {
  registry.addExtension(
      +[](MLIRContext *ctx, IREEHexagonEncodingDialect *dialect) {
        (void)dialect;
        HexagonEncodingResolverAttr::attachInterface<
            HexagonPackedLayoutMaterializerAttr, HexagonLayoutMaterializerAttr,
            HexagonLayoutResolverAttr, HexagonSerializableAttr>(*ctx);
      });
}

} // namespace mlir::iree_compiler::hexagon

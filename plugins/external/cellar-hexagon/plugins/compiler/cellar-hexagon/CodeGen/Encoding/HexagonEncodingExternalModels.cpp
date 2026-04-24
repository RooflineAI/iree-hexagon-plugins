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

#include "cellar-hexagon/CodeGen/Encoding/HexagonEncodingExternalModels.h"
#include "cellar-hexagon/CodeGen/Encoding/IR/HexagonEncodingAttrs.h"
#include "cellar-hexagon/CodeGen/Encoding/IR/HexagonEncodingDialect.h"

#include "iree/compiler/Codegen/Dialect/Codegen/Utils/Utils.h"
#include "iree/compiler/Codegen/ExternalInterfaces/Utils.h"
#include "iree/compiler/Codegen/Utils/Utils.h"
#include "iree/compiler/Dialect/Encoding/IR/EncodingTypes.h"
#include "iree/compiler/Dialect/Encoding/Utils/Utils.h"
#include "mlir/IR/BuiltinAttributes.h"

#define DEBUG_TYPE "iree-hexagon-encoding"

namespace mlir::iree_compiler::cellar_hexagon::codegen {

namespace Hexagon = IREE::Hexagon;
namespace Codegen = IREE::Codegen;
namespace Encoding = IREE::Encoding;

namespace {

// Fixed 4x4x4 tiling keeps the prototype simple while still exercising the
// encoding plumbing in the same way as CPU/GPU backends.
constexpr Codegen::TileMxNxK kHexagonPrototypeTile{4, 4, 4};

// TODO: Copy pasted function, come back to it later. It might not even be
// useful for Hexagon, this probably needs a lot of rethinking
static void transposeInPlace(IREE::Codegen::MaterializeEncodingInfo &info) {
  // Vector cases: nothing to do.
  if (info.innerTileSizes.size() < 2) {
    return;
  }
  // In non-vector cases, the last two entries of each array are M and N.
  auto transpose = [](SmallVector<int64_t> &a) {
    std::swap(a[a.size() - 2], a[a.size() - 1]);
  };
  transpose(info.innerDimsPos);
  transpose(info.innerTileSizes);
  transpose(info.outerDimsPerm);
}

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
  return linalg::GenericOp::create(
             builder, loc, castedType, input, init, maps, iteratorTypes,
             [&](OpBuilder &b, Location nestedLoc, ValueRange args) {
               Value castRes =
                   arith::ExtUIOp::create(b, nestedLoc, outElemType, args[0])
                       ->getResult(0);
               linalg::YieldOp::create(b, nestedLoc, castRes);
             })
      ->getResult(0);
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
Operation *
lowerContractionOpWithEncoding(OpBuilder &builder, linalg::LinalgOp linalgOp,
                               ValueRange operands,
                               Encoding::LayoutMaterializerAttr layoutAttr) {

  // Verification that this is one of the operations we actually want to lower.
  if (!linalgOp.hasPureTensorSemantics()) {
    return nullptr;
  }

  auto inputs = linalgOp.getDpsInputOperands();
  auto outputs = linalgOp.getDpsInits();

  auto lhsType = cast<RankedTensorType>(inputs[0]->get().getType());
  auto rhsType = cast<RankedTensorType>(inputs[1]->get().getType());
  auto resultType = cast<RankedTensorType>(outputs[0].getType());
  auto lhsEncoding = Encoding::getEncodingAttr(lhsType);
  auto rhsEncoding = Encoding::getEncodingAttr(rhsType);
  auto resultEncoding = Encoding::getEncodingAttr(resultType);
  if (!lhsEncoding || !rhsEncoding || !resultEncoding) {
    return nullptr;
  }

  if (lhsEncoding.getOperandIndex().getValue() != Encoding::MATMUL_LHS ||
      rhsEncoding.getOperandIndex().getValue() != Encoding::MATMUL_RHS ||
      resultEncoding.getOperandIndex().getValue() != Encoding::MATMUL_RESULT) {
    return nullptr;
  }

  // It is one of the operations we want to lower, let's try to do so.
  Codegen::MaterializeEncodingInfo encodingInfo = {};
  if (auto packedLayoutAttr =
          dyn_cast<Codegen::PackedLayoutMaterializerAttr>(layoutAttr)) {
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
  auto cDims = Encoding::getEncodingContractionDims(lhsEncoding);
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
              Hexagon::HexagonEncodingResolverAttr> {

  DictionaryAttr getConfiguration(Attribute attr) const {
    auto resolver = cast<Hexagon::HexagonEncodingResolverAttr>(attr);
    return resolver.getConfiguration();
  }

  Codegen::MaterializeEncodingInfo
  getEncodingInfoImpl(Attribute attr, RankedTensorType type) const {
    // Mirror the CPU/GPU flow: look at the tensor's encoding and request a
    // materialization recipe. This prototype always selects the 4x4x4 tile.
    Codegen::MaterializeEncodingInfo info;
    auto encoding =
        dyn_cast_or_null<Encoding::EncodingAttr>(type.getEncoding());
    if (!encoding)
      return info;

    FailureOr<Codegen::MaterializeEncodingInfo> maybeInfo =
        Codegen::getEncodingInfoForMatmul(encoding, kHexagonPrototypeTile);
    if (failed(maybeInfo)) {
      return info;
    }

    info = std::move(maybeInfo.value());

    // Narrow-N cases are handled by transposing the materialized layout so that
    // the narrow dimension maps to M. Copy pasted from LLVMCPU materializer
    if (IREE::Encoding::isNarrowNResult(encoding)) {
      transposeInPlace(info);
    }
    return info;
  }
};

// If you are looking at LLVMCPU as reference, note that this one is named
// differently, but I think they just messed up the naming
struct HexagonLayoutMaterializerAttr
    : public ::mlir::iree_compiler::IREE::
          EncodingLayoutMaterializerAttrExternalModelBase<
              HexagonLayoutMaterializerAttr,
              Hexagon::HexagonEncodingResolverAttr> {
  Operation *lowerOp(Attribute attr, OpBuilder &builder, Operation *op,
                     TypeRange convertedResTypes,
                     ValueRange convertedOperands) const {
    auto layoutAttr = cast<Hexagon::HexagonEncodingResolverAttr>(attr);
    auto linalgOp = dyn_cast<linalg::LinalgOp>(op);
    if (!linalgOp) {
      return nullptr;
    }

    if (auto fillOp = dyn_cast<linalg::FillOp>(op)) {
      return IREE::lowerFillOpWithResolvedLayouts(
          builder, fillOp, convertedResTypes, convertedOperands);
    }
    if (linalg::isaContractionOpInterface(linalgOp)) {
      return lowerContractionOpWithEncoding(
          builder, linalgOp, convertedOperands,
          cast<Encoding::LayoutMaterializerAttr>(layoutAttr));
    }

    // TODO: I have not tested this code path, copy pasted from LLVMCPU's
    // upstream bump
    if (auto genericOp = dyn_cast<linalg::GenericOp>(op)) {
      return IREE::lowerGenericOpWithResolvedLayouts(
          builder, genericOp, convertedResTypes, convertedOperands,
          cast<Encoding::LayoutMaterializerAttr>(attr));
    }
    return nullptr;
  }
};

struct HexagonLayoutResolverAttr
    : public Encoding::LayoutResolverAttr::ExternalModel<
          HexagonLayoutResolverAttr, Hexagon::HexagonEncodingResolverAttr> {
  Attribute cloneWithSimplifiedConfig(Attribute attr,
                                      DictionaryAttr config) const {
    MLIRContext *ctx = attr.getContext();
    SmallVector<NamedAttribute> configItems;

    // We could add more information to this config. As an example,
    // LLVMCPUTarget adds information about ukernels, cpu features and target
    // triple.
    // TODO: There is some weird behavior with this.The idea is to simplify the
    // verbosity of the IR according to the doc. Nevertheless, even in the CPU
    // pipeline the information is not added to the attribute, nor used
    // apparently. Can investigate later, maybe I am misunderstanding something.

    return Hexagon::HexagonEncodingResolverAttr::get(
        ctx, DictionaryAttr::get(ctx, configItems));
  }

  Attribute getLayout(Attribute attr, RankedTensorType type) const {
    MLIRContext *ctx = attr.getContext();

    DictionaryAttr config = IREE::getPackedLayoutImpl(attr, type);
    return Hexagon::HexagonEncodingResolverAttr::get(ctx, config);
  }
};

struct HexagonSerializableAttr
    : public Encoding::SerializableAttr::ExternalModel<
          HexagonSerializableAttr, Hexagon::HexagonEncodingResolverAttr> {

  // This get called from the encodeHostTensorPass, and that pass will fail if
  // this function returns false, since it requires the information we are
  // providing here
  bool isSerialized(Attribute attr) const {
    auto configuration =
        cast<Hexagon::HexagonEncodingResolverAttr>(attr).getConfiguration();
    return configuration && configuration.contains(IREE::kEncodingInfoAttrName);
  }

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
      +[](MLIRContext *ctx, Hexagon::IREEHexagonEncodingDialect *dialect) {
        (void)dialect;
        Hexagon::HexagonEncodingResolverAttr::attachInterface<
            HexagonPackedLayoutMaterializerAttr, HexagonLayoutMaterializerAttr,
            HexagonLayoutResolverAttr, HexagonSerializableAttr>(*ctx);
      });
}

} // namespace mlir::iree_compiler::cellar_hexagon::codegen

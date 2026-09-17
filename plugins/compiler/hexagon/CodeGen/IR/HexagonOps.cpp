// Copyright 2026 RooflineAI GmbH
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "hexagon/CodeGen/IR/HexagonOps.h"

#include "hexagon/CodeGen/IR/HexagonDialect.h"
#include "hexagon/CodeGen/IR/HmxContracts.h"

#include "mlir/Interfaces/SideEffectInterfaces.h"

#define GET_OP_CLASSES
#include "hexagon/CodeGen/IR/HexagonOps.cpp.inc"

namespace mlir::iree_compiler::IREE::Hexagon {
namespace {

void addReadEffect(
    OpOperand *operand,
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), operand,
                       SideEffects::DefaultResource::get());
}

void addWriteEffect(
    OpOperand *operand,
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Write::get(), operand,
                       SideEffects::DefaultResource::get());
}

} // namespace

LogicalResult VTCMEmptyOp::verify() {
  auto resultType = cast<RankedTensorType>(getResult().getType());
  if (getDynamicSizes().size() != resultType.getNumDynamicDims()) {
    return emitOpError("expected ")
           << resultType.getNumDynamicDims()
           << " dynamic size operands for the result type, got "
           << getDynamicSizes().size();
  }
  return success();
}

LogicalResult HmxPackOp::verify() {
  return verifyHmxPackContract(getOperation(), getSource().getType(),
                               getDest().getType(), getDim());
}

LogicalResult TensorHmxPackOp::verify() {
  return verifyHmxPackContract(getOperation(), getSource().getType(),
                               getDest().getType(), getDim());
}

LogicalResult HmxMatmulOp::verify() {
  return verifyHmxMatmulContract(getOperation(), getLhs().getType(),
                                 getRhs().getType(), getAcc().getType(),
                                 /*allowSingleTileAccumulator=*/true);
}

LogicalResult TensorHmxMatmulOp::verify() {
  return verifyHmxMatmulContract(getOperation(), getLhs().getType(),
                                 getRhs().getType(), getAcc().getType(),
                                 /*allowSingleTileAccumulator=*/false);
}

LogicalResult HmxUnpackOp::verify() {
  return verifyHmxUnpackContract(getOperation(), getSource().getType(),
                                 getDest().getType(), getDim());
}

LogicalResult TensorHmxUnpackOp::verify() {
  return verifyHmxUnpackContract(getOperation(), getSource().getType(),
                                 getDest().getType(), getDim());
}

LogicalResult HmxAccSetupReadOp::verify() {
  auto configType = dyn_cast<MemRefType>(getConfig().getType());
  if (!configType || configType.getRank() != 1 ||
      !configType.getElementType().isInteger(8) ||
      ShapedType::isDynamic(configType.getDimSize(0)) ||
      configType.getDimSize(0) < kHmxConfigBytes) {
    return emitOpError("config must be a static rank-1 i8 memref containing at "
                       "least 256 bytes");
  }
  return success();
}

LogicalResult HmxMmaOp::verify() {
  auto verifySide = [&](Value value, StringRef role) {
    // Tiling reduces a grid to one tile without dropping its rank, so both the
    // rank-3 tile and a singleton rank-5 grid reach the MMA.
    return verifyHmxSingleF16Tile(getOperation(),
                                  cast<MemRefType>(value.getType()), role,
                                  /*allowSingletonGrid=*/true);
  };
  if (failed(verifySide(getLhs(), "lhs"))) {
    return failure();
  }
  if (failed(verifySide(getRhs(), "rhs"))) {
    return failure();
  }
  // Keep operand/accumulator compatibility separate from type legality so new
  // hardware accumulator modes cannot be paired with the f16 MMA by accident.
  // Only `accIn` needs checking: `AllTypesMatch<["accIn", "accOut"]>` in the
  // operation definition already ties the result accumulator to it.
  if (!isHmxAccumulator(getAccIn().getType())) {
    return emitOpError("input and result accumulators must have type "
                       "!iree_hexagon.hmx.acc<32x32xf32>");
  }
  return success();
}

LogicalResult HmxAccReadOp::verify() {
  auto destType = cast<MemRefType>(getDest().getType());
  // The legal accumulator set may grow independently of this f16 read format.
  if (!isHmxAccumulator(getAcc().getType())) {
    return emitOpError(
        "accumulator must have type !iree_hexagon.hmx.acc<32x32xf32>");
  }
  return verifyHmxSingleF16Tile(getOperation(), destType, "destination",
                                /*allowSingletonGrid=*/false);
}

void HmxPackOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  addReadEffect(&getSourceMutable(), effects);
  addWriteEffect(&getDestMutable(), effects);
}

void HmxMatmulOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  addReadEffect(&getLhsMutable(), effects);
  addReadEffect(&getRhsMutable(), effects);
  addWriteEffect(&getAccMutable(), effects);
}

void HmxUnpackOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  addReadEffect(&getSourceMutable(), effects);
  addReadEffect(&getDestMutable(), effects);
  addWriteEffect(&getDestMutable(), effects);
}

/// Registers the operations generated from HexagonOps.td with the dialect.
void IREEHexagonDialect::registerOperations() {
  addOperations<
#define GET_OP_LIST
#include "hexagon/CodeGen/IR/HexagonOps.cpp.inc"
      >();
}

} // namespace mlir::iree_compiler::IREE::Hexagon

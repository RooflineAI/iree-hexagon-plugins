// Copyright 2026 RooflineAI GmbH
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "hexagon/Target/LLVMPasses/DynamicInsertElementToSelect.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/Analysis.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/Support/MathExtras.h"

namespace mlir::iree_compiler::hexagon::target {

// Whether the backend lowers a variable-index insert into `type` through a
// stack slot. Measured with llc (v79, 128- and 64-byte HVX) over i8, i16, i32,
// f16, f32, i64 and f64 elements and 1/4 to 4 HVX vectors:
// - Vectors of HVX element types wider than one HVX vector (register pairs,
//   and vectors split into several): through the stack.
// - Vectors of f32 from half an HVX vector up, with 128-byte HVX: through the
//   stack. They are widened to, or are, v32f32, for which the HVX lowering
//   leaves INSERT_VECTOR_ELT to the generic expansion.
// - Up to one HVX vector of i8, i16, i32 or f16: in registers (vror/vinsert).
// - i64, f64, and vectors shorter than half an HVX vector: no HVX store.
static bool isInsertedThroughStack(llvm::FixedVectorType *type,
                                   unsigned hvxVectorBits, bool hvxFloat) {
  llvm::Type *elementType = type->getElementType();
  bool isFloat = elementType->isFloatTy();
  bool isHvxElement =
      isFloat || elementType->isHalfTy() || elementType->isIntegerTy(8) ||
      elementType->isIntegerTy(16) || elementType->isIntegerTy(32);
  if (!isHvxElement) {
    return false;
  }
  uint64_t bits = type->getPrimitiveSizeInBits().getFixedValue();
  if (bits > hvxVectorBits) {
    return true;
  }
  return isFloat && hvxFloat && bits >= hvxVectorBits / 2;
}

// The narrowest of i8, i16 and i32 that is at least as wide as the element
// (so the compare mask has the data's lane layout) and holds every lane index.
static llvm::IntegerType *getLaneIndexType(llvm::FixedVectorType *type) {
  unsigned width = type->getScalarSizeInBits();
  unsigned laneBits = llvm::Log2_64_Ceil(type->getNumElements());
  for (unsigned candidate : {8u, 16u, 32u}) {
    if (candidate >= width && candidate >= laneBits) {
      return llvm::IntegerType::get(type->getContext(), candidate);
    }
  }
  return llvm::IntegerType::get(type->getContext(), 64);
}

llvm::PreservedAnalyses
DynamicInsertElementToSelectPass::run(llvm::Function &function,
                                      llvm::FunctionAnalysisManager &) {
  if (hvxVectorBits == 0) {
    return llvm::PreservedAnalyses::all();
  }
  bool changed = false;
  for (llvm::Instruction &instruction :
       llvm::make_early_inc_range(llvm::instructions(function))) {
    auto *insert = llvm::dyn_cast<llvm::InsertElementInst>(&instruction);
    if (!insert || llvm::isa<llvm::Constant>(insert->getOperand(2))) {
      continue;
    }
    auto *vectorType = llvm::dyn_cast<llvm::FixedVectorType>(insert->getType());
    if (!vectorType ||
        !isInsertedThroughStack(vectorType, hvxVectorBits, hvxFloat)) {
      continue;
    }

    unsigned numElements = vectorType->getNumElements();
    llvm::IntegerType *laneType = getLaneIndexType(vectorType);
    llvm::SmallVector<llvm::Constant *> lanes;
    lanes.reserve(numElements);
    for (unsigned lane = 0; lane < numElements; ++lane) {
      lanes.push_back(llvm::ConstantInt::get(laneType, lane));
    }

    // An index that does not fit the lane type is out of range, so the insert
    // is poison and any lane may be selected.
    llvm::IRBuilder<> builder(insert);
    llvm::Value *index =
        builder.CreateZExtOrTrunc(insert->getOperand(2), laneType);
    llvm::Value *mask =
        builder.CreateICmpEQ(builder.CreateVectorSplat(numElements, index),
                             llvm::ConstantVector::get(lanes));
    llvm::Value *element =
        builder.CreateVectorSplat(numElements, insert->getOperand(1));
    llvm::Value *select =
        builder.CreateSelect(mask, element, insert->getOperand(0));
    select->takeName(insert);
    insert->replaceAllUsesWith(select);
    insert->eraseFromParent();
    changed = true;
  }
  if (!changed) {
    return llvm::PreservedAnalyses::all();
  }
  llvm::PreservedAnalyses preserved;
  preserved.preserveSet<llvm::CFGAnalyses>();
  return preserved;
}

} // namespace mlir::iree_compiler::hexagon::target

// Copyright 2026 RooflineAI GmbH
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "hexagon/Target/LLVMPasses/SplatSelectConditionToI1Vector.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/IR/Analysis.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"

namespace mlir::iree_compiler::hexagon::target {

llvm::PreservedAnalyses
SplatSelectConditionToI1VectorPass::run(llvm::Function &function,
                                        llvm::FunctionAnalysisManager &) {
  bool changed = false;
  for (llvm::Instruction &instruction :
       llvm::make_early_inc_range(llvm::instructions(function))) {
    auto *select = llvm::dyn_cast<llvm::SelectInst>(&instruction);
    if (!select) {
      continue;
    }
    auto *vectorType = llvm::dyn_cast<llvm::VectorType>(select->getType());
    if (!vectorType || !vectorType->getElementType()->isIntegerTy(1) ||
        select->getCondition()->getType()->isVectorTy()) {
      continue;
    }
    llvm::IRBuilder<> builder(select);
    llvm::Value *condition = builder.CreateVectorSplat(
        vectorType->getElementCount(), select->getCondition());
    select->setCondition(condition);
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

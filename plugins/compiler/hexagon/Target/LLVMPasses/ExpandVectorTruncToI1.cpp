// Copyright 2025 RooflineAI GmbH
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "hexagon/Target/LLVMPasses/ExpandVectorTruncToI1.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/IR/Analysis.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"

namespace mlir::iree_compiler::hexagon::target {

llvm::PreservedAnalyses
ExpandVectorTruncToI1Pass::run(llvm::Function &function,
                               llvm::FunctionAnalysisManager &) {
  bool changed = false;
  for (llvm::Instruction &instruction :
       llvm::make_early_inc_range(llvm::instructions(function))) {
    auto *trunc = llvm::dyn_cast<llvm::TruncInst>(&instruction);
    if (!trunc || !trunc->getType()->isVectorTy() ||
        !trunc->getType()->getScalarType()->isIntegerTy(1) ||
        trunc->getSrcTy()->getScalarType()->isIntegerTy(8)) {
      continue;
    }
    llvm::IRBuilder<> builder(trunc);
    llvm::Value *source = trunc->getOperand(0);
    llvm::Value *isSet = builder.CreateIsNotNull(builder.CreateAnd(
        source, llvm::ConstantInt::get(source->getType(), 1)));
    isSet->takeName(trunc);
    trunc->replaceAllUsesWith(isSet);
    trunc->eraseFromParent();
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

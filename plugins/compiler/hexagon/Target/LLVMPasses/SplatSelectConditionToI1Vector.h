// Copyright 2026 RooflineAI GmbH
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef HEXAGON_TARGET_LLVMPASSES_SPLATSELECTCONDITIONTOI1VECTOR_H_
#define HEXAGON_TARGET_LLVMPASSES_SPLATSELECTCONDITIONTOI1VECTOR_H_

#include "llvm/IR/Function.h"
#include "llvm/IR/PassManager.h"

namespace mlir::iree_compiler::hexagon::target {

// FIXME: workaround for https://github.com/llvm/llvm-project/issues/239550
// Instruction selection crashes in vector legalization on a `select` with a
// scalar i1 condition between two vectors of i1, e.g.
// `select i1 %c, <8 x i1> %a, <8 x i1> %b`. Without HVX every width crashes;
// with 128-byte HVX the widths below 16 do. Masked loop remainders over a
// dynamic dimension produce this pattern.
//
// This pass splats the condition, `select <N x i1> splat(%c), %a, %b`, which
// selects the same lanes and which the backend handles. It must run after the
// last InstCombine of the middle-end pipeline, which folds a splat condition
// back into a scalar one.
struct SplatSelectConditionToI1VectorPass
    : llvm::RequiredPassInfoMixin<SplatSelectConditionToI1VectorPass> {
  llvm::PreservedAnalyses run(llvm::Function &function,
                              llvm::FunctionAnalysisManager &);
};

} // namespace mlir::iree_compiler::hexagon::target

#endif // HEXAGON_TARGET_LLVMPASSES_SPLATSELECTCONDITIONTOI1VECTOR_H_

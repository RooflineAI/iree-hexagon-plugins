// Copyright 2025 RooflineAI GmbH
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef HEXAGON_TARGET_LLVMPASSES_EXPANDVECTORTRUNCTOI1_H_
#define HEXAGON_TARGET_LLVMPASSES_EXPANDVECTORTRUNCTOI1_H_

#include "llvm/IR/Function.h"
#include "llvm/IR/PassManager.h"

namespace mlir::iree_compiler::hexagon::target {

// FIXME: workaround for https://github.com/llvm/llvm-project/issues/237238
// The Hexagon backend miscompiles a vector `trunc` to i1 on HVX when the
// source lanes are wider than a byte: it tests bit 0 of every byte, so a
// `select` on the result mixes bytes of both operands.
//
// This pass rewrites each such `trunc` as `icmp ne (and %x, 1), 0`, which
// computes the same lanes. It must run after the last InstCombine of the
// middle-end pipeline, which would fold the comparison back into a `trunc`.
// Sources with i8 lanes are left alone, as testing bit 0 of each byte is
// correct for them.
struct ExpandVectorTruncToI1Pass
    : llvm::RequiredPassInfoMixin<ExpandVectorTruncToI1Pass> {
  llvm::PreservedAnalyses run(llvm::Function &function,
                              llvm::FunctionAnalysisManager &);
};

} // namespace mlir::iree_compiler::hexagon::target

#endif // HEXAGON_TARGET_LLVMPASSES_EXPANDVECTORTRUNCTOI1_H_

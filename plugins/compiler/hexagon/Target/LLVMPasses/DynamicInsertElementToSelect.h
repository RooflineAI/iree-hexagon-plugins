// Copyright 2026 RooflineAI GmbH
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef HEXAGON_TARGET_LLVMPASSES_DYNAMICINSERTELEMENTTOSELECT_H_
#define HEXAGON_TARGET_LLVMPASSES_DYNAMICINSERTELEMENTTOSELECT_H_

#include "llvm/IR/Function.h"
#include "llvm/IR/PassManager.h"

namespace mlir::iree_compiler::hexagon::target {

// FIXME: workaround for a Hexagon VLIW packetizer bug, not yet filed upstream.
// An `insertelement` with a variable index into a vector that the backend does
// not insert into in registers is lowered through a stack slot: spill the
// vector with an HVX store, store the element with a scalar store, reload.
// The packetizer may put the spill and the element store in one packet. An HVX
// store can only go in slot 0, and the slot-0 store takes effect last, so the
// spill overwrites the element: the insert is lost. Strided copies, which
// gather elements into a vector one by one, then write zeros.
//
// This pass rewrites such an insert into
// `select (icmp eq (splat %idx), <0, 1, ..., N-1>), (splat %x), %v`, which the
// backend lowers in registers (a compare and a vmux). It is the same value for
// every in-range index; for an out-of-range index `insertelement` is poison.
// It must run after the last InstCombine of the middle-end pipeline, which
// folds the select back into an `insertelement`.
//
// Only the vector types that are lowered through the stack are rewritten
// (measured with llc on v79; see the .cpp). A variable-index `extractelement`
// is also lowered through the stack, but as a vector store followed by a
// scalar load, which the packetizer never puts in one packet.
struct DynamicInsertElementToSelectPass
    : llvm::RequiredPassInfoMixin<DynamicInsertElementToSelectPass> {
  // `hvxVectorBits`: the HVX vector length in bits (1024 or 512), or 0 when
  // HVX is not enabled, in which case the pass does nothing.
  // `hvxFloat`: whether vectors of f32 are HVX types (128-byte HVX on v68 or
  // later), whose variable-index insert the backend expands through the stack.
  DynamicInsertElementToSelectPass(unsigned hvxVectorBits, bool hvxFloat)
      : hvxVectorBits(hvxVectorBits), hvxFloat(hvxFloat) {}

  llvm::PreservedAnalyses run(llvm::Function &function,
                              llvm::FunctionAnalysisManager &);

  unsigned hvxVectorBits;
  bool hvxFloat;
};

} // namespace mlir::iree_compiler::hexagon::target

#endif // HEXAGON_TARGET_LLVMPASSES_DYNAMICINSERTELEMENTTOSELECT_H_

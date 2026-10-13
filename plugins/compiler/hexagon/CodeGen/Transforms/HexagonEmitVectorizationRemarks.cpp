// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Derived from Codegen/LLVMCPU/LLVMCPUEmitVectorizationRemarks.cpp at IREE
// revision a45adeaa6115e446c898e6eb21fb6edc0e65ddc4.

#include "hexagon/CodeGen/Passes.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Pass/Pass.h"

namespace mlir::iree_compiler::hexagon::codegen {

#define GEN_PASS_DEF_HEXAGONEMITVECTORIZATIONREMARKSPASS
#include "hexagon/CodeGen/Passes.h.inc"

namespace {
struct HexagonEmitVectorizationRemarksPass
    : impl::HexagonEmitVectorizationRemarksPassBase<
          HexagonEmitVectorizationRemarksPass> {
  void runOnOperation() override;
};
} // namespace

void HexagonEmitVectorizationRemarksPass::runOnOperation() {
  mlir::FunctionOpInterface funcOp = getOperation();
  bool dump = false;
  funcOp.walk([&](linalg::LinalgOp op) {
    op.emitWarning("op is not vectorized");
    dump = true;
  });
  if (dump) {
    funcOp.emitWarning("found one or more ops not vectorized");
  }
}
} // namespace mlir::iree_compiler::hexagon::codegen

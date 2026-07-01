// Copyright 2026 RooflineAI GmbH
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "cellar-hexagon/CodeGen/Passes.h"
#include "cellar-hexagon/CodeGen/Strategy/KernelDispatch.h"

#include "iree/compiler/Codegen/Dialect/CPU/IR/IREECPUDialect.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenDialect.h"

namespace mlir::iree_compiler::cellar_hexagon::codegen {

#define GEN_PASS_DEF_HEXAGONSELECTLOWERINGSTRATEGYPASS
#include "cellar-hexagon/CodeGen/Passes.h.inc"

namespace {

class HexagonSelectLoweringStrategyPass final
    : public impl::HexagonSelectLoweringStrategyPassBase<
          HexagonSelectLoweringStrategyPass> {
public:
  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::iree_compiler::IREE::CPU::IREECPUDialect,
                    mlir::iree_compiler::IREE::Codegen::IREECodegenDialect>();
  }

  void runOnOperation() override {
    for (mlir::FunctionOpInterface funcOp :
         getOperation().getOps<mlir::FunctionOpInterface>()) {
      if (mlir::failed(initHexagonLaunchConfig(funcOp))) {
        funcOp.emitOpError("failed to set Hexagon lowering configuration");
        signalPassFailure();
        return;
      }
    }
  }
};

} // namespace

} // namespace mlir::iree_compiler::cellar_hexagon::codegen

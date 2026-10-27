// Copyright 2026 RooflineAI GmbH
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "hexagon/CodeGen/Passes.h"

#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

namespace mlir::iree_compiler::hexagon::codegen {

#define GEN_PASS_DEF_HEXAGONMARKTHREADFORALLSPASS
#include "hexagon/CodeGen/Passes.h.inc"

bool isThreadForall(Operation *op) {
  auto forallOp = dyn_cast_if_present<scf::ForallOp>(op);
  if (!forallOp)
    return false;
  std::optional<ArrayAttr> mapping = forallOp.getMapping();
  return mapping && !mapping->empty() &&
         llvm::all_of(*mapping, llvm::IsaPred<IREE::Codegen::LocalMappingAttr>);
}

namespace {

struct HexagonMarkThreadForallsPass
    : impl::HexagonMarkThreadForallsPassBase<HexagonMarkThreadForallsPass> {
  using Base::Base;

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    getOperation()->walk([&](scf::ForallOp forallOp) {
      if (forallOp.getMapping())
        return;
      SmallVector<Attribute> mapping;
      for (int64_t i = 0, e = forallOp.getRank(); i < e; ++i)
        mapping.push_back(IREE::Codegen::LocalMappingAttr::get(context, i));
      forallOp.setMappingAttr(ArrayAttr::get(context, mapping));
    });
  }
};

} // namespace
} // namespace mlir::iree_compiler::hexagon::codegen

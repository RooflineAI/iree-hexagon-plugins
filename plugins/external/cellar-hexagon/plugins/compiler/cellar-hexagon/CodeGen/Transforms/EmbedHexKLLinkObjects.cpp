// Copyright 2025 RooflineAI GmbH
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "cellar-hexagon/CodeGen/Passes.h"

#include "hexagon/Dialect/HexKL/IR/HexKLDialect.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SetVector.h"

namespace mlir::iree_compiler::cellar_hexagon::codegen {

#define GEN_PASS_DEF_EMBEDHEXKLLINKOBJECTSPASS
#include "cellar-hexagon/CodeGen/Passes.h.inc"

namespace {

constexpr llvm::StringLiteral kHexKLRuntimeLib = "libruntime_lib.a";
constexpr llvm::StringLiteral kHexKLMicroLib = "libhexkl_micro.a";

static llvm::ArrayRef<llvm::StringLiteral> getRequiredHexKLLinkObjects() {
  static constexpr llvm::StringLiteral kObjectNames[] = {kHexKLRuntimeLib,
                                                         kHexKLMicroLib};
  return kObjectNames;
}

struct EmbedHexKLLinkObjectsPass
    : public impl::EmbedHexKLLinkObjectsPassBase<EmbedHexKLLinkObjectsPass> {
  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::hexkl::HexKLDialect>();
  }

  void runOnOperation() override {
    mlir::ModuleOp moduleOp = getOperation();

    // TODO: Ideally I would remove the pass altogether and manage this
    // dynamically from the runtime, but for the time being I am removing this
    // that was making it slightly cleaner
    // bool requiresHexKLLinkObjects = false; moduleOp.walk(
    //     [&](mlir::hexkl::MatmulOp) { requiresHexKLLinkObjects = true; });
    // if (!requiresHexKLLinkObjects)
    //   return;

    auto variantOp = moduleOp->getParentOfType<
        mlir::iree_compiler::IREE::HAL::ExecutableVariantOp>();
    if (!variantOp)
      return;

    if (variantOp.getTarget().getBackend().getValue() != "hexagon")
      return;

    mlir::Builder builder(&getContext());
    llvm::SetVector<mlir::Attribute> objectAttrs;
    if (auto existingObjectsAttr = variantOp.getObjectsAttr()) {
      objectAttrs.insert(existingObjectsAttr.begin(),
                         existingObjectsAttr.end());
    }

    for (llvm::StringRef objectName : getRequiredHexKLLinkObjects()) {
      auto objectAttr =
          builder.getAttr<mlir::iree_compiler::IREE::HAL::ExecutableObjectAttr>(
              builder.getStringAttr(objectName), nullptr);
      objectAttrs.insert(objectAttr);
    }

    variantOp.setObjectsAttr(builder.getArrayAttr(objectAttrs.getArrayRef()));
  }
};

} // namespace

} // namespace mlir::iree_compiler::cellar_hexagon::codegen

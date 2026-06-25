// Copyright 2026 RooflineAI GmbH
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "cellar-hexagon/CodeGen/Passes.h"

#include "cellar-hexagon/CodeGen/Encoding/IR/HexagonEncodingDialect.h"
#include "cellar-hexagon/CodeGen/Encoding/IR/HexagonEncodingOps.h"
#include "mlir/Dialect/LLVMIR/FunctionCallUtils.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/SymbolTable.h"
#include "llvm/ADT/StringMap.h"

namespace mlir::iree_compiler::cellar_hexagon::codegen {
namespace IREE = mlir::iree_compiler::IREE;

#define GEN_PASS_DEF_LOWERPROFILINGMARKERSPASS
#include "cellar-hexagon/CodeGen/Passes.h.inc"

namespace {

constexpr llvm::StringLiteral kProfilingZoneBeginFn =
    "hexagon_runtime_profiling_zone_begin";
constexpr llvm::StringLiteral kProfilingZoneEndFn =
    "hexagon_runtime_profiling_zone_end";
constexpr llvm::StringLiteral kCStringGlobalPrefix =
    "__hexagon_profiling_marker";

std::string getUniqueSymbolName(ModuleOp moduleOp, StringRef prefix) {
  unsigned counter = 0;
  SmallString<128> name = SymbolTable::generateSymbolName<128>(
      prefix,
      [&](StringRef candidate) { return moduleOp.lookupSymbol(candidate); },
      counter);
  return name.str().str();
}

Value getOrCreateCStringPtr(ModuleOp moduleOp, OpBuilder &builder, Location loc,
                            StringRef value,
                            llvm::StringMap<LLVM::GlobalOp> &globalCache) {
  MLIRContext *ctx = moduleOp.getContext();
  auto ptrType = LLVM::LLVMPointerType::get(ctx);
  if (value.empty()) {
    return LLVM::ZeroOp::create(builder, loc, ptrType).getResult();
  }

  LLVM::GlobalOp global = globalCache.lookup(value);
  LLVM::LLVMArrayType stringType =
      LLVM::LLVMArrayType::get(builder.getI8Type(), value.size() + 1);
  if (!global) {
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(moduleOp.getBody());

    std::string storage = value.str();
    storage.push_back('\0');
    global = LLVM::GlobalOp::create(
        builder, loc, stringType, /*isConstant=*/true, LLVM::Linkage::Internal,
        getUniqueSymbolName(moduleOp, kCStringGlobalPrefix),
        builder.getStringAttr(storage));
    globalCache[value] = global;
  }

  Value address = LLVM::AddressOfOp::create(builder, loc, global).getResult();
  return LLVM::GEPOp::create(builder, loc, ptrType, stringType, address,
                             ArrayRef<LLVM::GEPArg>{0, 0})
      .getResult();
}

struct LowerProfilingMarkersPass final
    : public impl::LowerProfilingMarkersPassBase<LowerProfilingMarkersPass> {
  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry
        .insert<IREE::Hexagon::IREEHexagonEncodingDialect, LLVM::LLVMDialect>();
  }

  void runOnOperation() override {
    ModuleOp moduleOp = getOperation();
    MLIRContext *ctx = &getContext();
    OpBuilder builder(ctx);
    llvm::SmallVector<Operation *> markers;
    moduleOp.walk([&](Operation *op) {
      if (isa<IREE::Hexagon::ProfilingBeginOp, IREE::Hexagon::ProfilingEndOp>(
              op)) {
        markers.push_back(op);
      }
    });
    if (markers.empty()) {
      return;
    }

    auto voidType = LLVM::LLVMVoidType::get(ctx);
    auto i32Type = builder.getI32Type();
    auto ptrType = LLVM::LLVMPointerType::get(ctx);

    FailureOr<LLVM::LLVMFuncOp> beginFn = LLVM::lookupOrCreateFn(
        builder, moduleOp, kProfilingZoneBeginFn, {i32Type, ptrType}, voidType);
    FailureOr<LLVM::LLVMFuncOp> endFn = LLVM::lookupOrCreateFn(
        builder, moduleOp, kProfilingZoneEndFn, {}, voidType);
    if (failed(beginFn) || failed(endFn))
      return signalPassFailure();

    llvm::StringMap<LLVM::GlobalOp> stringGlobals;

    for (Operation *marker : markers) {
      builder.setInsertionPoint(marker);
      Location loc = marker->getLoc();

      if (auto beginOp = dyn_cast<IREE::Hexagon::ProfilingBeginOp>(marker)) {
        StringRef extraInfo = beginOp.getExtraInfo().value_or(StringRef{});

        Value zoneType = LLVM::ConstantOp::create(
                             builder, loc, i32Type,
                             builder.getI32IntegerAttr(beginOp.getZoneType()))
                             .getResult();
        Value extraInfoPtr = getOrCreateCStringPtr(moduleOp, builder, loc,
                                                   extraInfo, stringGlobals);
        LLVM::CallOp::create(builder, loc, beginFn.value(),
                             ValueRange{zoneType, extraInfoPtr});
      } else {
        LLVM::CallOp::create(builder, loc, endFn.value(), ValueRange{});
      }

      marker->erase();
    }
  }
};

} // namespace

} // namespace mlir::iree_compiler::cellar_hexagon::codegen

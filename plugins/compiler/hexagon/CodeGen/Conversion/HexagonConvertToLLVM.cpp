// Copyright 2025 RooflineAI GmbH
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "hexagon/CodeGen/Conversion/HexagonConvertToLLVM.h"

#include "hexagon/CodeGen/Conversion/HexagonABIToLLVM.h"
#include "hexagon/CodeGen/Conversion/HexagonRuntimeLinking.h"
#include "hexagon/CodeGen/IR/HexagonDialect.h"
#include "hexagon/CodeGen/IR/HexagonOps.h"
#include "hexagon/CodeGen/Passes.h"

#include "hexagon/Common/Common.h"
#include "hexagon/Conversion/DMAToLLVM/DMAToLLVM.h"
#include "hexagon/Conversion/HexKLToLLVM/HexKLToLLVM.h"
#include "hexagon/Conversion/HexagonMemToLLVM/HexagonMemToLLVM.h"
#include "hexagon/Dialect/HexKL/IR/HexKLDialect.h"
#include "hexagon/Dialect/HexagonMem/IR/HexagonMemDialect.h"
#include "iree/compiler/Codegen/Common/PassUtils.h"
#include "iree/compiler/Codegen/Common/Transforms.h"
#include "iree/compiler/Codegen/Utils/Utils.h"
#include "iree/compiler/Dialect/HAL/IR/HALDialect.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Dialect/Util/IR/UtilDialect.h"
#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
#include "mlir/Analysis/DataLayoutAnalysis.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ComplexToLLVM/ComplexToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
#include "mlir/Conversion/IndexToLLVM/IndexToLLVM.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/LoweringOptions.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Conversion/MathToLLVM/MathToLLVM.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Conversion/TosaToArith/TosaToArith.h"
#include "mlir/Conversion/UBToLLVM/UBToLLVM.h"
#include "mlir/Conversion/VectorToLLVM/ConvertVectorToLLVM.h"
#include "mlir/Conversion/VectorToSCF/VectorToSCF.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/Passes.h"
#include "mlir/Dialect/LLVMIR/FunctionCallUtils.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Math/Transforms/Passes.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/Transforms/Transforms.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Transforms/LoweringPatterns.h"
#include "mlir/Dialect/Vector/Transforms/VectorRewritePatterns.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/SmallString.h"

namespace mlir::iree_compiler::hexagon::codegen {
namespace IREE = mlir::iree_compiler::IREE;

#define GEN_PASS_DEF_HEXAGONCONVERTTOLLVMPASS
#include "hexagon/CodeGen/Passes.h.inc"

namespace {

class HexagonConvertToLLVMPass
    : public impl::HexagonConvertToLLVMPassBase<HexagonConvertToLLVMPass> {
public:
  using Base::Base;
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, math::MathDialect, func::FuncDialect,
                    memref::MemRefDialect, linalg::LinalgDialect,
                    tosa::TosaDialect, scf::SCFDialect, vector::VectorDialect,
                    LLVM::LLVMDialect, IREE::Hexagon::IREEHexagonDialect,
                    hexagonmem::HexagonMemDialect, hexkl::HexKLDialect>();
  }
  void runOnOperation() override;
};

} // namespace

static std::string getDataLayoutString(ModuleOp module) {
  auto targetAttr = IREE::HAL::ExecutableTargetAttr::lookup(module);
  if (!targetAttr) {
    return "";
  }
  std::optional<StringRef> stringAttr =
      getConfigDataLayout(targetAttr.getConfiguration());
  return stringAttr ? stringAttr.value().str() : std::string("");
}

static std::string getTargetTripleString(ModuleOp module) {
  auto targetAttr = IREE::HAL::ExecutableTargetAttr::lookup(module);
  if (!targetAttr) {
    return "";
  }
  std::optional<StringRef> stringAttr =
      getConfigTargetTriple(targetAttr.getConfiguration());
  return stringAttr ? stringAttr.value().str() : std::string("");
}

static void populateTanhPatterns(RewritePatternSet &p) {
  StringRef fname = math::TanhOp::getOperationName();
  StringRef opName =
      fname.drop_front(math::MathDialect::getDialectNamespace().size() + 1);
  math::populateExpansionPatterns(p, /*OpMnemonics=*/{opName});
};

namespace {

//===----------------------------------------------------------------------===//
// Hexagon runtime lowering.
//===----------------------------------------------------------------------===//

constexpr llvm::StringLiteral kProfilerZoneBeginFn =
    "hexagon_runtime_profiler_zone_begin";
constexpr llvm::StringLiteral kProfilerZoneEndFn =
    "hexagon_runtime_profiler_zone_end";
constexpr llvm::StringLiteral kCStringGlobalPrefix =
    "__hexagon_profiler_marker";

static std::string getUniqueSymbolName(ModuleOp moduleOp, StringRef prefix) {
  unsigned counter = 0;
  SmallString<128> name = SymbolTable::generateSymbolName<128>(
      prefix,
      [&](StringRef candidate) { return moduleOp.lookupSymbol(candidate); },
      counter);
  return name.str().str();
}

// Materializes a pointer to a NUL-terminated constant string, reusing an
// existing internal global with the same contents when one is already present.
static Value getOrCreateCStringPtr(ModuleOp moduleOp, OpBuilder &builder,
                                   Location loc, StringRef value) {
  MLIRContext *ctx = moduleOp.getContext();
  auto ptrType = LLVM::LLVMPointerType::get(ctx);
  if (value.empty()) {
    return LLVM::ZeroOp::create(builder, loc, ptrType).getResult();
  }

  std::string storage = value.str();
  storage.push_back('\0');
  auto storageAttr = builder.getStringAttr(storage);
  LLVM::LLVMArrayType stringType =
      LLVM::LLVMArrayType::get(builder.getI8Type(), value.size() + 1);

  // Deduplicate identical marker strings (e.g. many ops sharing the same
  // "hexagonmem.copy" tag) by reusing a matching global instead of caching op
  // pointers across conversion-pattern invocations.
  LLVM::GlobalOp global;
  for (auto candidate : moduleOp.getOps<LLVM::GlobalOp>()) {
    if (candidate.getSymName().starts_with(kCStringGlobalPrefix) &&
        candidate.getValueOrNull() == storageAttr) {
      global = candidate;
      break;
    }
  }
  if (!global) {
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(moduleOp.getBody());
    global = LLVM::GlobalOp::create(
        builder, loc, stringType, /*isConstant=*/true, LLVM::Linkage::Internal,
        getUniqueSymbolName(moduleOp, kCStringGlobalPrefix), storageAttr);
  }

  Value address = LLVM::AddressOfOp::create(builder, loc, global).getResult();
  return LLVM::GEPOp::create(builder, loc, ptrType, stringType, address,
                             ArrayRef<LLVM::GEPArg>{0, 0})
      .getResult();
}

/// Lowers `iree_hexagon.get_runtime_state` by delegating the entry argument and
/// extended dispatch-state layout to HexagonDispatchABI.
struct ConvertGetRuntimeStateOp
    : public ConvertOpToLLVMPattern<IREE::Hexagon::GetRuntimeStateOp> {
  ConvertGetRuntimeStateOp(HexagonDispatchABI &abi,
                           LLVMTypeConverter &typeConverter)
      : ConvertOpToLLVMPattern(typeConverter), abi(abi) {}

  LogicalResult
  matchAndRewrite(IREE::Hexagon::GetRuntimeStateOp getStateOp, OpAdaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto funcOp = getStateOp->getParentOfType<LLVM::LLVMFuncOp>();
    if (!funcOp || funcOp.getNumArguments() < 2) {
      return rewriter.notifyMatchFailure(
          getStateOp,
          "expected an enclosing function with a dispatch state argument");
    }
    rewriter.replaceOp(getStateOp, abi.loadRuntimeState(getStateOp, rewriter));
    return success();
  }

  HexagonDispatchABI &abi;
};

/// Lowers `iree_hexagon.profiler.begin` to a call to the
/// `hexagon_runtime_profiler_zone_begin` runtime entry point.
struct ConvertProfilerBeginOp
    : public ConvertOpToLLVMPattern<IREE::Hexagon::ProfilerBeginOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;
  LogicalResult
  matchAndRewrite(IREE::Hexagon::ProfilerBeginOp beginOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto moduleOp = beginOp->getParentOfType<ModuleOp>();
    Location loc = beginOp.getLoc();
    auto i32Type = rewriter.getI32Type();
    auto ptrType = LLVM::LLVMPointerType::get(rewriter.getContext());

    FailureOr<LLVM::LLVMFuncOp> beginFn =
        LLVM::lookupOrCreateFn(rewriter, moduleOp, kProfilerZoneBeginFn,
                               {ptrType, i32Type, ptrType}, ptrType);
    if (failed(beginFn)) {
      return rewriter.notifyMatchFailure(
          beginOp, "failed to declare profiler zone begin runtime helper");
    }

    // The zone type enum values match the runtime's
    // iree_hal_hexagon_profiler_zone_types_t, so the enum is passed on as its
    // underlying integer value.
    Value zoneType = LLVM::ConstantOp::create(
                         rewriter, loc, i32Type,
                         rewriter.getI32IntegerAttr(
                             static_cast<int32_t>(beginOp.getZoneType())))
                         .getResult();
    Value extraInfoPtr = getOrCreateCStringPtr(
        moduleOp, rewriter, loc, beginOp.getExtraInfo().value_or(StringRef{}));
    // `adaptor.getState()` is the `!llvm.ptr` from the `runtime_state` operand.
    auto callOp = LLVM::CallOp::create(
        rewriter, loc, beginFn.value(),
        ValueRange{adaptor.getState(), zoneType, extraInfoPtr});
    // The result replaces the typed profiler_record value with an `!llvm.ptr`;
    // the matching profiler.end pattern reads it through its adaptor.
    rewriter.replaceOp(beginOp, callOp.getResult());
    return success();
  }
};

/// Lowers `iree_hexagon.profiler.end` to a call to the
/// `hexagon_runtime_profiler_zone_end` runtime entry point.
struct ConvertProfilerEndOp
    : public ConvertOpToLLVMPattern<IREE::Hexagon::ProfilerEndOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;
  LogicalResult
  matchAndRewrite(IREE::Hexagon::ProfilerEndOp endOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto moduleOp = endOp->getParentOfType<ModuleOp>();
    Location loc = endOp.getLoc();
    auto voidType = LLVM::LLVMVoidType::get(rewriter.getContext());
    auto ptrType = LLVM::LLVMPointerType::get(rewriter.getContext());

    FailureOr<LLVM::LLVMFuncOp> endFn = LLVM::lookupOrCreateFn(
        rewriter, moduleOp, kProfilerZoneEndFn, {ptrType}, voidType);
    if (failed(endFn)) {
      return rewriter.notifyMatchFailure(
          endOp, "failed to declare profiler zone end runtime helper");
    }

    // The record operand was replaced with the `!llvm.ptr` returned by the
    // lowered profiler.begin call, so read it generically via the adaptor.
    LLVM::CallOp::create(rewriter, loc, endFn.value(),
                         ValueRange{adaptor.getRecord()});
    rewriter.eraseOp(endOp);
    return success();
  }
};

static void
populateHexagonRuntimeToLLVMConversionPatterns(HexagonDispatchABI &abi,
                                               LLVMTypeConverter &typeConverter,
                                               RewritePatternSet &patterns) {
  // Runtime state and profiler records are opaque runtime values represented
  // as plain pointers in LLVM dialect.
  typeConverter.addConversion(
      [](IREE::Hexagon::RuntimeStateType type) -> std::optional<Type> {
        return LLVM::LLVMPointerType::get(type.getContext());
      });
  typeConverter.addConversion(
      [](IREE::Hexagon::ProfilerRecordType type) -> std::optional<Type> {
        return LLVM::LLVMPointerType::get(type.getContext());
      });
  patterns.addWithLabel<ConvertGetRuntimeStateOp>({"hexagon-runtime-to-llvm"},
                                                  abi, typeConverter);
  patterns.addWithLabel<ConvertProfilerBeginOp, ConvertProfilerEndOp>(
      {"hexagon-runtime-to-llvm"}, typeConverter);
}

} // namespace

void HexagonConvertToLLVMPass::runOnOperation() {
  mlir::ModuleOp moduleOp = getOperation();
  std::string dataLayoutStr = targetDataLayout;
  if (targetDataLayout.empty()) {
    dataLayoutStr = getDataLayoutString(moduleOp);
  }
  std::string targetTripleStr = targetTriple;
  if (targetTripleStr.empty()) {
    targetTripleStr = getTargetTripleString(moduleOp);
  }
  // Add required attributes to the module so that the lowering knows how to
  // handle structs and data layouts.
  moduleOp->setAttr(LLVM::LLVMDialect::getTargetTripleAttrName(),
                    StringAttr::get(moduleOp->getContext(), targetTripleStr));
  moduleOp->setAttr(LLVM::LLVMDialect::getDataLayoutAttrName(),
                    StringAttr::get(moduleOp->getContext(), dataLayoutStr));

  // Run Vector -> Vector transformations ahead of conversion to LLVM.
  {
    RewritePatternSet patterns(&getContext());
    vector::populateVectorToVectorCanonicalizationPatterns(patterns);
    vector::populateBubbleVectorBitCastOpPatterns(patterns);
    vector::populateVectorBroadcastLoweringPatterns(patterns);
    vector::populateVectorGatherToConditionalLoadPatterns(patterns);
    vector::populateVectorInterleaveLoweringPatterns(patterns);
    // TODO: doubtful that the "default" does what one want here, it is likely
    // better to use outerproduct.
    vector::VectorTransformsOptions defaultOptions;
    vector::populateVectorContractLoweringPatterns(
        patterns, defaultOptions.vectorContractLowering);
    vector::populateVectorMaskMaterializationPatterns(
        patterns, /*force32BitVectorIndices=*/false);
    vector::populateVectorMaskOpLoweringPatterns(patterns);
    // FIXME: This pass is run twice unneccessarily, once here and another time
    // in addHexagonLowerToLLVMPasses in LoweringPipelines.cpp. Related to
    // ROO-1458.
    vector::populateVectorShapeCastLoweringPatterns(patterns);
    // vector::populateVectorFromElementsLoweringPatterns(patterns);
    // vector::populateVectorToElementsLoweringPatterns(patterns);
    vector::populateVectorFromElementsUnrollPatterns(patterns);
    vector::populateVectorToElementsUnrollPatterns(patterns);
    // TODO: doubtful that the "default" does what one want here, it is likely
    // better to use shuffle.
    vector::populateVectorTransposeLoweringPatterns(
        patterns, defaultOptions.vectorTransposeLowering);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      return signalPassFailure();
    }
  }
  {
    RewritePatternSet vectorToLoopsPatterns(&getContext());
    populateVectorToSCFConversionPatterns(
        vectorToLoopsPatterns, VectorTransferToSCFOptions().enableFullUnroll());
    if (failed(applyPatternsGreedily(getOperation(),
                                     std::move(vectorToLoopsPatterns)))) {
      return signalPassFailure();
    }
  }

  const auto &dataLayoutAnalysis = getAnalysis<DataLayoutAnalysis>();
  LowerToLLVMOptions options(&getContext(),
                             dataLayoutAnalysis.getAtOrAbove(moduleOp));
  options.dataLayout = llvm::DataLayout(dataLayoutStr);
  // LLVMCPU enforces the index bitwidth to be the same as the pointer bitwidth.
  // Hexagon-mlir expects the default bitwidth of 64 though, so skipping it.
  // options.overrideIndexBitwidth(options.dataLayout.getPointerSizeInBits());
  LLVMTypeConverter typeConverter(&getContext(), options, &dataLayoutAnalysis);

  // Hexagon's LLVM backend represents both DDR and VTCM pointers in address
  // space zero. Configure that mapping on the one type converter shared by
  // every pattern family.
  ::mlir::hexagon::addTypeConversions(&getContext(), typeConverter);

  RewritePatternSet patterns(&getContext());

  tosa::populateTosaRescaleToArithConversionPatterns(&patterns,
                                                     /*use32BitImpl=*/false);
  LLVMConversionTarget target(getContext());
  populateAffineToStdConversionPatterns(patterns);
  populateSCFToControlFlowConversionPatterns(patterns);
  cf::populateControlFlowToLLVMConversionPatterns(typeConverter, patterns);
  // TODO: This should be revisited, hexagon-mlir already has decompositions for
  // specific operations. Need to check which ones are more efficient.
  populateTanhPatterns(patterns);

  populateComplexToLLVMConversionPatterns(typeConverter, patterns);
  // TODO: This should be revisited, hexagon-mlir already has decompositions for
  // specific operations. Need to check which ones are more efficient.
  populateMathToLLVMConversionPatterns(typeConverter, patterns);
  // Note: workaround needed due to `memref.subview` returned from an `if`.
  memref::populateExpandStridedMetadataPatterns(patterns);
  iree_compiler::populateIREEResolveExtractStridedMetadataPatterns(patterns);
  populateFinalizeMemRefToLLVMConversionPatterns(typeConverter, patterns);
  populateFuncToLLVMConversionPatterns(typeConverter, patterns);
  arith::populateArithToLLVMConversionPatterns(typeConverter, patterns);
  index::populateIndexToLLVMConversionPatterns(typeConverter, patterns);
  populateVectorToSCFConversionPatterns(patterns);
  // Some n-D vectors are generated by EmulateNarrowType pass, so we need to
  // unroll them to 1-D before converting to the LLVM dialect.
  vector::populateVectorBitCastLoweringPatterns(patterns);
  vector::populateVectorRankReducingFMAPattern(patterns);
  vector::populateVectorInsertExtractStridedSliceTransforms(patterns);
  vector::populateVectorStepLoweringPatterns(patterns);
  // TODO: Revisit this in the future when performance becomes relevant
  // LLVMCPU uses data type alignment. Non ideal vector memory
  // access patterns when specifying this alignment for this backend (the llvm
  // backend needs to be a lot more conservative when it comes to unaligned
  // accesses) have been observed. In theory, tiling should guarantee proper
  // alignment. Nevertheless, there are some dispatches, like "slow copies" into
  // padded buffers that will fail when using vector alignment here.
  populateVectorToLLVMConversionPatterns(typeConverter, patterns,
                                         reassociateFpReductions, false, false);
  // vector::populateVectorFromElementsLoweringPatterns(patterns);
  // vector::populateVectorToElementsLoweringPatterns(patterns);
  vector::populateVectorFromElementsUnrollPatterns(patterns);
  vector::populateVectorToElementsUnrollPatterns(patterns);
  ub::populateUBToLLVMConversionPatterns(typeConverter, patterns);
  vector::populateVectorTransferLoweringPatterns(patterns,
                                                 /*maxTransferRank=*/1);

  HexagonDispatchABI abi(&typeConverter);
  populateHexagonABIToLLVMConversionPatterns(abi, typeConverter, patterns);
  populateHexagonInstrumentationToLLVMConversionPatterns(abi, typeConverter,
                                                         patterns);
  populateHexagonRuntimeToLLVMConversionPatterns(abi, typeConverter, patterns);

  // Target-specific lowerings participate in the same conversion transaction
  // as the standard and HAL ABI lowerings. Pattern labels supplied by these
  // helpers preserve per-family observability with MLIR pattern debugging.
  hexagonmem::populateHexagonMemToLLVMConversionPatterns(typeConverter,
                                                         patterns);
  ::mlir::hexagon::populateDMAToLLVMConversionPatterns(typeConverter, patterns);
  hexkl::populateHexKLToLLVMConversionPatterns(typeConverter, patterns);

  target.addLegalOp<ModuleOp, IREE::Codegen::DispatchConfigOp>();
  target.markOpRecursivelyLegal<IREE::Codegen::DispatchConfigOp>();
  target.addIllegalDialect<
      func::FuncDialect, mlir::arith::ArithDialect, IREE::Util::UtilDialect,
      IREE::HAL::HALDialect, math::MathDialect, tosa::TosaDialect,
      hexagonmem::HexagonMemDialect, hexkl::HexKLDialect>();
  target.addIllegalOp<
      IREE::Hexagon::GetRuntimeStateOp, IREE::Hexagon::ProfilerBeginOp,
      IREE::Hexagon::ProfilerEndOp, memref::DmaStartOp, memref::DmaWaitOp>();

  if (failed(applyPartialConversion(moduleOp, target, std::move(patterns)))) {
    signalPassFailure();
    return;
  }

  // Materializations are allowed while the conversion is in flight, but must
  // not become another pipeline seam. Reconcile them here and fail if a cast
  // cannot be eliminated.
  SmallVector<UnrealizedConversionCastOp> casts;
  moduleOp.walk(
      [&](UnrealizedConversionCastOp cast) { casts.push_back(cast); });
  SmallVector<UnrealizedConversionCastOp> remainingCasts;
  reconcileUnrealizedCasts(casts, &remainingCasts);
  if (!remainingCasts.empty()) {
    remainingCasts.front().emitError(
        "unreconciled materialization after Hexagon LLVM conversion");
    return signalPassFailure();
  }

  // Helper externs such as free/memrefCopy are introduced during memref
  // finalization. Canonicalize and mark them for static/native DSP linking.
  for (auto funcOp : moduleOp.getOps<LLVM::LLVMFuncOp>()) {
    if (failed(renameAndTagNativeRuntimeLinkedFunc(moduleOp, funcOp))) {
      return signalPassFailure();
    }
  }

  // If we were supporting dynamic library imports, we would need to add a
  // conversion pattern here to rewrite any remaining external calls to the
  // appropriate runtime entry point. However, Hexagon does not support dynamic
  // imports, so any remaining external calls are invalid and should be reported
  // as an error.

  if (failed(validateHexagonExternalCalls(moduleOp))) {
    return signalPassFailure();
  }

  // No target-specific post conversion patterns for Hexagon.
}

std::unique_ptr<OperationPass<ModuleOp>>
createHexagonConvertToLLVMPass(bool reassociateFpReductions) {
  HexagonConvertToLLVMPassOptions options;
  options.reassociateFpReductions = reassociateFpReductions;
  return std::make_unique<HexagonConvertToLLVMPass>(options);
}

} // namespace mlir::iree_compiler::hexagon::codegen

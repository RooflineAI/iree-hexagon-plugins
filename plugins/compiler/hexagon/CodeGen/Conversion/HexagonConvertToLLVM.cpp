// Copyright 2025 RooflineAI GmbH
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "hexagon/CodeGen/Conversion/HexagonConvertToLLVM.h"

#include "hexagon/CodeGen/Conversion/HexagonABIToLLVM.h"
#include "hexagon/CodeGen/Conversion/HexagonHmxToLLVM.h"
#include "hexagon/CodeGen/Conversion/HexagonRuntimeLinking.h"
#include "hexagon/CodeGen/IR/HexagonDialect.h"
#include "hexagon/CodeGen/IR/HexagonOps.h"
#include "hexagon/CodeGen/Passes.h"

#include "hexagon/Common/Common.h"
#include "hexagon/Conversion/DMAToLLVM/DMAToLLVM.h"
#include "hexagon/Conversion/HexKLToLLVM/HexKLExternalFnNames.h"
#include "hexagon/Conversion/HexKLToLLVM/HexKLToLLVM.h"
#include "hexagon/Conversion/HexagonMemToLLVM/HexagonMemExternalFnNames.h"
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
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Transforms/Passes.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
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
#include "mlir/IR/Matchers.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/MathExtras.h"

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
    // The coroutines of the async threads are outlined from the dispatch and
    // have other arguments; they get the state as a captured value instead.
    auto funcOp = getStateOp->getParentOfType<LLVM::LLVMFuncOp>();
    if (!funcOp || funcOp.getNumArguments() < 2 ||
        funcOp.walk([](LLVM::CoroIdOp) { return WalkResult::interrupt(); })
            .wasInterrupted()) {
      return rewriter.notifyMatchFailure(
          getStateOp,
          "expected an enclosing function with a dispatch state argument");
    }
    rewriter.replaceOp(getStateOp,
                       abi.getRuntimeStateAtEntry(getStateOp, rewriter));
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

//===----------------------------------------------------------------------===//
// VTCM partition allocations.
//===----------------------------------------------------------------------===//

// HexKL's matmul places its VTCM tiles at HEXKL_HMX_ACTIVATION_ALIGNMENT.
constexpr uint64_t kHexKLScratchAlignment = 2048;

/// Whether |funcOp| has the signature of a dispatch entry point as produced by
/// the HAL ABI lowering: i32 (environment*, dispatch_state*, workgroup_state*).
/// Only entry points receive the dispatch state
static bool hasDispatchEntryPointSignature(LLVM::LLVMFuncOp funcOp) {
  LLVM::LLVMFunctionType type = funcOp.getFunctionType();
  return type.getNumParams() == 3 && type.getReturnType().isInteger(32) &&
         llvm::all_of(type.getParams(), [](Type type) {
           return isa<LLVM::LLVMPointerType>(type);
         });
}

static Value castToIntType(OpBuilder &builder, Location loc, Value value,
                           IntegerType type) {
  auto valueType = cast<IntegerType>(value.getType());
  if (valueType == type)
    return value;
  if (valueType.getWidth() < type.getWidth())
    return LLVM::ZExtOp::create(builder, loc, type, value);
  return LLVM::TruncOp::create(builder, loc, type, value);
}

/// Returns true for the `index -> i64` casts that the async-to-LLVM lowering
/// emits for the fixed 64 bit arguments of the AsyncRuntime (group sizes).
static bool isIndexToI64Cast(UnrealizedConversionCastOp castOp) {
  return castOp.getNumOperands() == 1 && castOp.getNumResults() == 1 &&
         castOp.getOperand(0).getType().isIndex() &&
         castOp.getResult(0).getType().isInteger(64);
}

/// `index` is 32 bit on Hexagon, so the `index -> i64` casts above can't be
/// reconciled away and widen the converted operand instead.
struct IndexToI64CastLowering
    : public OpConversionPattern<UnrealizedConversionCastOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(UnrealizedConversionCastOp castOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (!isIndexToI64Cast(castOp))
      return failure();
    rewriter.replaceOp(castOp, castToIntType(rewriter, castOp.getLoc(),
                                             adaptor.getOperands().front(),
                                             rewriter.getI64Type()));
    return success();
  }
};

/// The async lowering asserts that the awaited thread group is not in an error
/// state. For Hexagon, the RT handles this case and aborts instead, so no need
/// to codegen it
struct DropAssertLowering
    : public mlir::OpConversionPattern<mlir::cf::AssertOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(mlir::cf::AssertOp assertOp, OpAdaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.eraseOp(assertOp);
    return success();
  }
};

/// Returns the constant |value| holds, looking through memref descriptor
/// insertvalue/extractvalue chains, which are not folded yet after conversion.
static std::optional<int64_t> getConstantIntValue(Value value) {
  while (auto extractOp = value.getDefiningOp<LLVM::ExtractValueOp>()) {
    LLVM::ExtractValueOp::FoldAdaptor adaptor({Attribute()}, extractOp);
    auto folded = dyn_cast_if_present<Value>(extractOp.fold(adaptor));
    if (!folded || folded == value)
      break;
    value = folded;
  }
  APInt constant;
  if (!matchPattern(value, m_ConstantInt(&constant)))
    return std::nullopt;
  return constant.getSExtValue();
}

/// Declares size_t hexkl_matmul_f16f16_f32_scratch_bytes(uint64_t n_row,
/// uint64_t n_col, uint64_t n_inner). size_t is 32 bits on Hexagon.
static FailureOr<LLVM::LLVMFuncOp>
getHexKLMatmulScratchBytesFn(OpBuilder &builder, ModuleOp moduleOp) {
  Type i32Type = builder.getI32Type();
  Type i64Type = builder.getI64Type();
  return LLVM::lookupOrCreateFn(builder, moduleOp,
                                kHexKLMatmulScratchBytesFnName,
                                {i64Type, i64Type, i64Type}, i32Type);
}

/// Declares int hexkl_matmul_f16f16_f32_with_scratch(uint64_t n_row,
/// uint64_t n_col, uint64_t n_inner, float *outM, const _Float16 *inAct,
/// const _Float16 *inW, void *vtcm_scratch).
static FailureOr<LLVM::LLVMFuncOp>
getHexKLMatmulWithScratchFn(OpBuilder &builder, ModuleOp moduleOp) {
  Type i32Type = builder.getI32Type();
  Type i64Type = builder.getI64Type();
  Type ptrType = LLVM::LLVMPointerType::get(builder.getContext());
  return LLVM::lookupOrCreateFn(
      builder, moduleOp, kHexKLMatmulWithScratchFnName,
      {i64Type, i64Type, i64Type, ptrType, ptrType, ptrType, ptrType}, i32Type);
}

/// Declares void *hexagon_runtime_vtcm_alloc(const hexagon_rt_state_t *,
/// size_t nbytes, uint64_t alignment).
static FailureOr<LLVM::LLVMFuncOp>
getVtcmAllocFn(OpBuilder &builder, ModuleOp moduleOp, Type sizeType) {
  Type ptrType = LLVM::LLVMPointerType::get(builder.getContext());
  return LLVM::lookupOrCreateFn(builder, moduleOp, kVtcmAllocFnName,
                                {ptrType, sizeType, builder.getI64Type()},
                                ptrType);
}

/// Declares void *hexagon_runtime_coro_alloc(int64_t alignment, int64_t size).
static FailureOr<LLVM::LLVMFuncOp> getCoroAllocFn(OpBuilder &builder,
                                                  ModuleOp moduleOp) {
  Type i64Type = builder.getI64Type();
  Type ptrType = LLVM::LLVMPointerType::get(builder.getContext());
  return LLVM::lookupOrCreateFn(builder, moduleOp, kCoroAllocFnName,
                                {i64Type, i64Type}, ptrType);
}

/// Declares void hexagon_runtime_coro_free(void *frame).
static FailureOr<LLVM::LLVMFuncOp> getCoroFreeFn(OpBuilder &builder,
                                                 ModuleOp moduleOp) {
  Type ptrType = LLVM::LLVMPointerType::get(builder.getContext());
  return LLVM::lookupOrCreateFn(builder, moduleOp, kCoroFreeFnName, {ptrType},
                                LLVM::LLVMVoidType::get(builder.getContext()));
}

/// ConvertAsyncToLLVM allocates each coroutine frame with libc `aligned_alloc`
/// and frees it with `free`; the names are hard-coded there. Retargets exactly
/// these calls, the allocation that feeds `llvm.intr.coro.begin` and the free
/// of `llvm.intr.coro.free`, to the runtime's frame allocator. The libc
/// declarations take the 64 bit index of the async lowering, which does not
/// match the 32 bit size_t of Hexagon.
static LogicalResult useRuntimeCoroFrameAllocator(ModuleOp moduleOp) {
  SmallVector<LLVM::CallOp> allocCalls, freeCalls;
  moduleOp.walk([&](LLVM::CallOp callOp) {
    std::optional<StringRef> callee = callOp.getCallee();
    if (!callee)
      return;
    if (*callee == "aligned_alloc" && callOp.getNumResults() == 1 &&
        llvm::any_of(callOp->getUsers(), llvm::IsaPred<LLVM::CoroBeginOp>))
      allocCalls.push_back(callOp);
    else if (*callee == "free" && callOp.getArgOperands().size() == 1 &&
             callOp.getArgOperands()[0].getDefiningOp<LLVM::CoroFreeOp>())
      freeCalls.push_back(callOp);
  });

  OpBuilder builder(moduleOp.getContext());
  builder.setInsertionPointToEnd(moduleOp.getBody());
  if (!allocCalls.empty()) {
    FailureOr<LLVM::LLVMFuncOp> allocFn = getCoroAllocFn(builder, moduleOp);
    if (failed(allocFn))
      return moduleOp.emitError("failed to declare ") << kCoroAllocFnName;
    for (LLVM::CallOp callOp : allocCalls)
      callOp.setCallee(allocFn->getSymName());
  }
  if (!freeCalls.empty()) {
    FailureOr<LLVM::LLVMFuncOp> freeFn = getCoroFreeFn(builder, moduleOp);
    if (failed(freeFn))
      return moduleOp.emitError("failed to declare ") << kCoroFreeFnName;
    for (LLVM::CallOp callOp : freeCalls)
      callOp.setCallee(freeFn->getSymName());
  }
  return success();
}

/// Allocates |size| bytes aligned to |alignment| from the VTCM partition that
/// the runtime |state| holds. The runtime pads and aligns the allocation
/// itself
static Value createVtcmAlloc(OpBuilder &builder, Location loc,
                             LLVM::LLVMFuncOp vtcmAllocFn, Value state,
                             Value size, uint64_t alignment) {
  auto sizeType =
      cast<IntegerType>(vtcmAllocFn.getFunctionType().getParamType(1));
  Value alignmentValue = LLVM::ConstantOp::create(
      builder, loc, builder.getI64Type(), builder.getI64IntegerAttr(alignment));
  return LLVM::CallOp::create(
             builder, loc, vtcmAllocFn,
             ValueRange{state, castToIntType(builder, loc, size, sizeType),
                        alignmentValue})
      .getResult();
}

// The patterns below replace the upstream MemRefToLLVM and hexagon-mlir
// HexagonMemToLLVM lowerings of the same ops, which emit `malloc`, `free` and
// the hexagonmem runtime calls instead. Those stay registered at the default
// benefit of 1, so a higher benefit is all it takes to win over them.
constexpr unsigned kVtcmPartitionBenefit = 2;

/// Base of the patterns that route a dispatch's VTCM allocations to the
/// dispatch's VTCM partition, which the runtime passes via the runtime state.
/// The partition is released as a whole after the dispatch, so deallocations
/// are dropped.
///
/// A failed match falls back to the lower-benefit lowerings, which would
/// silently legalize the op. Errors are therefore emitted here and recorded in
/// |hadError| so that the pass fails.
template <typename SourceOp>
struct VtcmPartitionPattern : public ConvertOpToLLVMPattern<SourceOp> {
  VtcmPartitionPattern(HexagonDispatchABI &abi,
                       const LLVMTypeConverter &typeConverter, bool &hadError)
      : ConvertOpToLLVMPattern<SourceOp>(typeConverter, kVtcmPartitionBenefit),
        abi(abi), hadError(hadError) {}

  /// Emits |diag| as the reason the op can not be lowered.
  LogicalResult emitFailure(InFlightDiagnostic &&diag) const {
    hadError = true;
    return diag;
  }

  /// Only dispatch entry points receive the dispatch state, which holds the
  /// partition.
  LogicalResult verifyInDispatch(SourceOp op) const {
    auto funcOp = op->template getParentOfType<LLVM::LLVMFuncOp>();
    if (funcOp && hasDispatchEntryPointSignature(funcOp))
      return success();
    return emitFailure(op.emitOpError()
                       << "outside a dispatch function is not supported; VTCM "
                          "is allocated from the dispatch's partition, which "
                          "requires the dispatch state");
  }

  /// Declares the partition's allocation function. A conflicting declaration
  /// is an error, not a failed match, so it can't fall back to `malloc`.
  FailureOr<LLVM::LLVMFuncOp> declareVtcmAllocFn(SourceOp op,
                                                 OpBuilder &builder) const {
    FailureOr<LLVM::LLVMFuncOp> vtcmAllocFn =
        getVtcmAllocFn(builder, op->template getParentOfType<ModuleOp>(),
                       this->getIndexType());
    if (failed(vtcmAllocFn)) {
      (void)emitFailure(op.emitOpError()
                        << "can not declare '" << kVtcmAllocFnName
                        << "' to allocate from the dispatch's VTCM partition");
    }
    return vtcmAllocFn;
  }

  HexagonDispatchABI &abi;
  bool &hadError;
};

/// Lowers `memref.alloc` like MemRefToLLVM's `malloc` lowering, but takes the
/// buffer from the dispatch's VTCM partition. Heap allocations are VTCM
/// allocations on Hexagon.
struct ConvertAllocOpToVtcmPartition
    : public VtcmPartitionPattern<memref::AllocOp> {
  using VtcmPartitionPattern::VtcmPartitionPattern;

  LogicalResult
  matchAndRewrite(memref::AllocOp allocOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    MemRefType memRefType = allocOp.getType();
    if (!isConvertibleAndHasIdentityMaps(memRefType))
      return rewriter.notifyMatchFailure(allocOp, "incompatible memref type");
    if (failed(verifyInDispatch(allocOp)))
      return failure();
    FailureOr<LLVM::LLVMFuncOp> vtcmAllocFn =
        declareVtcmAllocFn(allocOp, rewriter);
    if (failed(vtcmAllocFn))
      return failure();

    Location loc = allocOp.getLoc();
    SmallVector<Value, 4> sizes;
    SmallVector<Value, 4> strides;
    Value sizeBytes;
    getMemRefDescriptorSizes(loc, memRefType, adaptor.getOperands(), rewriter,
                             sizes, strides, sizeBytes, /*sizeInBytes=*/true);

    // As in MemRefToLLVM: the requested alignment or, for non-scalar elements,
    // the element size.
    uint64_t alignment = 0;
    if (std::optional<uint64_t> requested = allocOp.getAlignment()) {
      alignment = *requested;
    } else if (!memRefType.getElementType().isSignlessIntOrIndexOrFloat()) {
      alignment = llvm::PowerOf2Ceil(DataLayout::closest(allocOp).getTypeSize(
          memRefType.getElementType()));
    }

    Value state = abi.getRuntimeStateAtEntry(allocOp, rewriter);
    Value ptr = createVtcmAlloc(rewriter, loc, *vtcmAllocFn, state, sizeBytes,
                                alignment);
    Type elementPtrType = getElementPtrType(memRefType);
    if (ptr.getType() != elementPtrType)
      ptr = LLVM::AddrSpaceCastOp::create(rewriter, loc, elementPtrType, ptr);

    Value descriptor = createMemRefDescriptor(loc, memRefType, ptr, ptr, sizes,
                                              strides, rewriter);
    rewriter.replaceOp(allocOp, descriptor);
    return success();
  }
};

/// Lowers VTCM `hexagonmem.alloc` like HexagonMemToLLVM, but takes the buffer
/// from the dispatch's VTCM partition instead of calling
/// `hexagon_runtime_alloc_1d_dsp`. DDR allocations are left to
/// HexagonMemToLLVM, which turns them into a `memref.alloc` which is handled by
/// the pattern above.
struct ConvertHexagonMemAllocOpToVtcmPartition
    : public VtcmPartitionPattern<hexagonmem::AllocOp> {
  using VtcmPartitionPattern::VtcmPartitionPattern;

  LogicalResult
  matchAndRewrite(hexagonmem::AllocOp allocOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto origMemRefType = dyn_cast<MemRefType>(allocOp.getBuffer().getType());
    if (!origMemRefType ||
        !::mlir::hexagon::isInVTCMAddressSpace(origMemRefType))
      return rewriter.notifyMatchFailure(allocOp, "not a VTCM allocation");
    if (failed(verifyInDispatch(allocOp)))
      return failure();
    FailureOr<LLVM::LLVMFuncOp> vtcmAllocFn =
        declareVtcmAllocFn(allocOp, rewriter);
    if (failed(vtcmAllocFn))
      return failure();

    Location loc = allocOp.getLoc();
    MemRefType memRefType = affine::normalizeMemRefType(origMemRefType);
    SmallVector<Value, 4> sizes;
    SmallVector<Value, 4> strides;
    Value sizeBytes;
    getMemRefDescriptorSizes(loc, memRefType, adaptor.getOperands(), rewriter,
                             sizes, strides, sizeBytes, /*sizeInBytes=*/true);
    Value state = abi.getRuntimeStateAtEntry(allocOp, rewriter);
    Value ptr = createVtcmAlloc(rewriter, loc, *vtcmAllocFn, state, sizeBytes,
                                allocOp.getAlignment());
    Value descriptor = createMemRefDescriptor(loc, memRefType, ptr, ptr, sizes,
                                              strides, rewriter);
    rewriter.replaceOp(allocOp, descriptor);
    return success();
  }
};

/// Drops `memref.dealloc`: its buffer comes from the dispatch's VTCM partition.
struct ConvertDeallocOpToVtcmPartition
    : public VtcmPartitionPattern<memref::DeallocOp> {
  using VtcmPartitionPattern::VtcmPartitionPattern;

  LogicalResult
  matchAndRewrite(memref::DeallocOp deallocOp, OpAdaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (failed(verifyInDispatch(deallocOp)))
      return failure();
    rewriter.eraseOp(deallocOp);
    return success();
  }
};

/// Drops VTCM `hexagonmem.dealloc`: its buffer comes from the dispatch's VTCM
/// partition. DDR deallocations are left to HexagonMemToLLVM, which turns them
/// into a `memref.dealloc`.
struct ConvertHexagonMemDeallocOpToVtcmPartition
    : public VtcmPartitionPattern<hexagonmem::DeallocOp> {
  using VtcmPartitionPattern::VtcmPartitionPattern;

  LogicalResult
  matchAndRewrite(hexagonmem::DeallocOp deallocOp, OpAdaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto memRefType = dyn_cast<MemRefType>(deallocOp.getBuffer().getType());
    if (!memRefType || !::mlir::hexagon::isInVTCMAddressSpace(memRefType))
      return rewriter.notifyMatchFailure(deallocOp, "not a VTCM deallocation");
    if (failed(verifyInDispatch(deallocOp)))
      return failure();
    rewriter.eraseOp(deallocOp);
    return success();
  }
};

static void populateVtcmPartitionToLLVMConversionPatterns(
    HexagonDispatchABI &abi, LLVMTypeConverter &typeConverter,
    RewritePatternSet &patterns, bool &hadError) {
  patterns.addWithLabel<ConvertAllocOpToVtcmPartition,
                        ConvertHexagonMemAllocOpToVtcmPartition,
                        ConvertDeallocOpToVtcmPartition,
                        ConvertHexagonMemDeallocOpToVtcmPartition>(
      {"vtcm-partition-to-llvm"}, abi, typeConverter, hadError);
}

/// Gives HexKL's matmul its VTCM scratch from the dispatch's partition: the
/// call to the matmul that allocates its own scratch is replaced by one to the
/// variant taking the scratch as an argument.
static LogicalResult
useHexKLMatmulWithScratch(ModuleOp moduleOp, HexagonDispatchABI &abi,
                          LLVMTypeConverter &typeConverter) {
  const std::string matmulFnName = hexkl::getMatmulMicroFnName();
  SmallVector<LLVM::CallOp> calls;
  moduleOp.walk([&](LLVM::CallOp callOp) {
    if (callOp.getCallee() == matmulFnName)
      calls.push_back(callOp);
  });
  if (calls.empty())
    return success();

  MLIRContext *ctx = moduleOp.getContext();
  auto i64Type = IntegerType::get(ctx, 64);
  OpBuilder builder(ctx);
  FailureOr<LLVM::LLVMFuncOp> vtcmAllocFn =
      getVtcmAllocFn(builder, moduleOp, typeConverter.getIndexType());
  FailureOr<LLVM::LLVMFuncOp> scratchBytesFn =
      getHexKLMatmulScratchBytesFn(builder, moduleOp);
  FailureOr<LLVM::LLVMFuncOp> matmulWithScratchFn =
      getHexKLMatmulWithScratchFn(builder, moduleOp);
  if (failed(vtcmAllocFn) || failed(scratchBytesFn) ||
      failed(matmulWithScratchFn))
    return failure();

  for (LLVM::CallOp callOp : calls) {
    auto funcOp = callOp->getParentOfType<LLVM::LLVMFuncOp>();
    if (!funcOp || !hasDispatchEntryPointSignature(funcOp)) {
      return callOp.emitOpError()
             << "calls '" << matmulFnName
             << "' outside a dispatch function; VTCM is allocated from the "
                "dispatch's partition, which requires the dispatch state";
    }
    // hexkl matmul(n_row, n_col, n_inner, out, lhs, rhs)
    ValueRange operands = callOp.getArgOperands();
    if (operands.size() != 6) {
      return callOp.emitOpError() << "expected 6 operands for '" << matmulFnName
                                  << "', got " << operands.size();
    }
    // The scratch is allocated at the dispatch entry, where only constant dims
    // are known.
    SmallVector<int64_t> dims;
    for (Value dim : operands.take_front(3)) {
      std::optional<int64_t> value = getConstantIntValue(dim);
      if (!value) {
        return callOp.emitOpError()
               << "with dynamic dims is not supported; its VTCM scratch is "
                  "allocated at the dispatch entry";
      }
      dims.push_back(*value);
    }
    Location loc = callOp.getLoc();
    Value scratch;
    {
      OpBuilder::InsertionGuard guard(builder);
      Value state = abi.getRuntimeStateAtEntry(callOp, builder);
      builder.setInsertionPointAfterValue(state);
      SmallVector<Value> dimValues;
      for (int64_t dim : dims) {
        dimValues.push_back(LLVM::ConstantOp::create(
            builder, loc, i64Type, builder.getI64IntegerAttr(dim)));
      }
      Value scratchBytes =
          LLVM::CallOp::create(builder, loc, *scratchBytesFn, dimValues)
              .getResult();
      scratch = createVtcmAlloc(builder, loc, *vtcmAllocFn, state, scratchBytes,
                                kHexKLScratchAlignment);
    }
    // The runtime takes the dims as uint64_t, independently of the index
    // bitwidth the HexKL lowering used.
    builder.setInsertionPoint(callOp);
    SmallVector<Value> args;
    for (Value dim : operands.take_front(3))
      args.push_back(castToIntType(builder, loc, dim, i64Type));
    llvm::append_range(args, operands.drop_front(3));
    args.push_back(scratch);
    auto newCall =
        LLVM::CallOp::create(builder, loc, *matmulWithScratchFn, args);
    if (callOp.getNumResults())
      callOp.getResult().replaceAllUsesWith(newCall.getResult());
    callOp.erase();
  }

  // Drop the now unused declaration.
  auto matmulFn = moduleOp.lookupSymbol<LLVM::LLVMFuncOp>(matmulFnName);
  if (matmulFn && matmulFn.isExternal() &&
      SymbolTable::symbolKnownUseEmpty(matmulFn, moduleOp)) {
    matmulFn.erase();
  }
  return success();
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
    // in addHexagonPreLLVMConversionPasses in LoweringPipelines.cpp. Related to
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
  // As in LLVMCPU, `index` has the pointer bitwidth (32 bits on Hexagon). The
  // module's MLIR data layout does not specify one, so the default would be 64.
  // Runtime arguments of a fixed width are typed explicitly instead of using
  // the index type.
  options.overrideIndexBitwidth(options.dataLayout.getPointerSizeInBits());
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
  bool vtcmPartitionError =
      false; // signal failure of the patterns that re-write the allocation
  // used, such that no other lower priority re-write is thred, but the pass
  // fails instead
  populateVtcmPartitionToLLVMConversionPatterns(abi, typeConverter, patterns,
                                                vtcmPartitionError);
  populateHexagonHmxToLLVMConversionPatterns(typeConverter, patterns);
  patterns.add<IndexToI64CastLowering, DropAssertLowering>(typeConverter,
                                                           &getContext());

  // Target-specific lowerings participate in the same conversion transaction
  // as the standard and HAL ABI lowerings. Pattern labels supplied by these
  // helpers preserve per-family observability with MLIR pattern debugging.
  hexagonmem::populateHexagonMemToLLVMConversionPatterns(typeConverter,
                                                         patterns);
  ::mlir::hexagon::populateDMAToLLVMConversionPatterns(typeConverter, patterns);
  hexkl::populateHexKLToLLVMConversionPatterns(typeConverter, patterns);

  target.addLegalOp<ModuleOp, IREE::Codegen::DispatchConfigOp>();
  target.markOpRecursivelyLegal<IREE::Codegen::DispatchConfigOp>();
  target.addIllegalDialect<func::FuncDialect, mlir::arith::ArithDialect,
                           IREE::Util::UtilDialect, IREE::HAL::HALDialect,
                           math::MathDialect, tosa::TosaDialect,
                           hexagonmem::HexagonMemDialect, hexkl::HexKLDialect,
                           IREE::Hexagon::IREEHexagonDialect>();
  target.addIllegalOp<memref::DmaStartOp, memref::DmaWaitOp, cf::AssertOp>();
  target.addDynamicallyLegalOp<UnrealizedConversionCastOp>(
      [](UnrealizedConversionCastOp castOp) {
        return !isIndexToI64Cast(castOp);
      });

  if (failed(applyPartialConversion(moduleOp, target, std::move(patterns))) ||
      vtcmPartitionError) {
    signalPassFailure();
    return;
  }

  // Private symbols must not end up public
  for (auto funcOp : moduleOp.getOps<LLVM::LLVMFuncOp>()) {
    if (!funcOp.isExternal() && funcOp.isPrivate())
      funcOp.setLinkage(LLVM::Linkage::Internal);
  }

  // Rename the generated async coroutines
  // third-party/llvm-project/mlir/lib/Dialect/Async/Transforms/AsyncToAsyncRuntime.cpp:338
  // contains a TODO: Derive outlined function name from the parent FuncOp so we
  // do that here for now and re-name the generated functions, to match te
  // dispatchs name
  for (auto funcOp :
       llvm::make_early_inc_range(moduleOp.getOps<LLVM::LLVMFuncOp>())) {
    if (funcOp.isExternal() ||
        !funcOp.getSymName().starts_with("async_execute_fn"))
      continue;
    auto uses = SymbolTable::getSymbolUses(funcOp, moduleOp);
    if (!uses || uses->empty())
      continue;
    auto caller = uses->begin()->getUser()->getParentOfType<LLVM::LLVMFuncOp>();
    SymbolTable symbolTable(moduleOp);
    std::string newName = (caller.getSymName() + "_async").str();
    for (int i = 1; symbolTable.lookup(newName); ++i)
      newName = (caller.getSymName() + "_async_" + Twine(i)).str();
    if (failed(SymbolTable::replaceAllSymbolUses(
            funcOp, StringAttr::get(&getContext(), newName), moduleOp))) {
      funcOp.emitOpError("failed to rename async helper");
      return signalPassFailure();
    }
    funcOp.setSymName(newName);
  }

  if (failed(useRuntimeCoroFrameAllocator(moduleOp)))
    return signalPassFailure();

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

  if (failed(useHexKLMatmulWithScratch(moduleOp, abi, typeConverter))) {
    return signalPassFailure();
  }

  // Helper externs such as memrefCopy are introduced during memref
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

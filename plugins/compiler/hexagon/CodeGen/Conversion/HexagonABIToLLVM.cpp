// Copyright 2025 RooflineAI GmbH
// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "hexagon/CodeGen/Conversion/HexagonABIToLLVM.h"

#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/schemas/instruments/dispatch.h"
#include "mlir/Analysis/DataLayoutAnalysis.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/SymbolTable.h"
#include "llvm/ADT/TypeSwitch.h"

// The local executable ABI patterns are derived from
// iree/compiler/Codegen/LLVMCPU/ConvertToLLVM.cpp at IREE revision
// a45adeaa6115e446c898e6eb21fb6edc0e65ddc4. Hexagon omits the LLVMCPU
// workgroup-local allocation policy and the dynamic/bitcode import rewrites.
// Native DSP runtime-link classification lives in HexagonRuntimeLinking.cpp,
// and the Hexagon dialect runtime conversions in the conversion driver.
// Last synchronized: 2026-09-25.

namespace mlir::iree_compiler::hexagon::codegen {
namespace IREE = mlir::iree_compiler::IREE;

namespace {

template <typename OpT>
struct ConvertOpToLLVMWithABIPattern : public ConvertOpToLLVMPattern<OpT> {
  ConvertOpToLLVMWithABIPattern(HexagonDispatchABI &abi,
                                LLVMTypeConverter &typeConverter,
                                PatternBenefit benefit = 1)
      : ConvertOpToLLVMPattern<OpT>(typeConverter, benefit), abi(abi) {}
  HexagonDispatchABI &abi;
};

/// Converts Standard MLIR FuncOps to LLVMFuncOps matching the IREE HAL ABI.
/// This is an IREE-specific conversion that assumes the input function is
/// `() -> ()` and that hal.interface.* ops are used to access all state.
///
/// Source function:
///
/// ```
/// func.func @foo() {
///   %0 = hal.interface.binding.subspan ...
/// }
/// ```
///
/// into:
///
/// ```
/// llvm.func foo(%state: !llvm.ptr<!...>,
///               %workgroup_id : !llvm.ptr<!llvm.array<i32, 3>>) {
///   %0 = <GEP/loads to access binding in %state>
/// }
/// ```
///
/// See `iree/hal/local/executable_library.h` for more information.
///
/// NOTE: we bump the benefit of the pattern to 100 to pick this pattern instead
/// of a competing pattern inserted by `populateFuncToLLVMConversionPatterns`.
struct ConvertHALEntryPointFuncOp
    : public ConvertOpToLLVMWithABIPattern<func::FuncOp> {
  ConvertHALEntryPointFuncOp(HexagonDispatchABI &abi,
                             LLVMTypeConverter &typeConverter)
      : ConvertOpToLLVMWithABIPattern(abi, typeConverter,
                                      /*benefit=*/100) {}
  LogicalResult
  matchAndRewrite(func::FuncOp stdFuncOp, func::FuncOpAdaptor operands,
                  ConversionPatternRewriter &rewriter) const override {
    if (!stdFuncOp.isPublic()) {
      return failure();
    }
    FunctionType fnType = stdFuncOp.getFunctionType();
    if (fnType.getNumInputs() != 0 || fnType.getNumResults() != 0) {
      stdFuncOp->emitWarning()
          << "public functions on executables must be () -> ()";
      return failure();
    }

    // Convert the function signature to take the HAL ABI LLVM pointers.
    TypeConverter::SignatureConversion signatureConverter(/*numOrigInputs=*/0);
    MLIRContext *context = rewriter.getContext();
    auto abiInputTypes = HexagonDispatchABI::getInputTypes(context);
    signatureConverter.addInputs(abiInputTypes);

    // Copy all attributes onto the LLVM function except the ones handled by
    // MLIR implicitly.
    SmallVector<NamedAttribute> funcAttrs;
    for (auto attr : stdFuncOp->getAttrs()) {
      if (attr.getName() == SymbolTable::getSymbolAttrName() ||
          attr.getName() == stdFuncOp.getFunctionTypeAttrName()) {
        continue;
      }
      funcAttrs.push_back(attr);
    }

    // Clone the function as an LLVMFuncOp and convert all interior types.
    auto int32Type = IntegerType::get(rewriter.getContext(), 32);
    auto llvmFuncType = LLVM::LLVMFunctionType::get(int32Type, abiInputTypes);
    auto llvmFuncOp = LLVM::LLVMFuncOp::create(
        rewriter, stdFuncOp.getLoc(), stdFuncOp.getName(), llvmFuncType,
        LLVM::Linkage::External, /*dsoLocal=*/false, /*cconv=*/LLVM::CConv::C,
        /*comdat=*/nullptr, funcAttrs);
    rewriter.inlineRegionBefore(stdFuncOp.getFunctionBody(),
                                llvmFuncOp.getFunctionBody(), llvmFuncOp.end());
    if (failed(rewriter.convertRegionTypes(&llvmFuncOp.getFunctionBody(),
                                           *typeConverter,
                                           &signatureConverter))) {
      return failure();
    }

    // Tag all arguments so LLVM can reason about our exports it otherwise
    // cannot analyze. We do this early on so that MLIR-based LLVM transforms
    // can use the attributes.
    // (%arg0: environment, %arg1: dispatch_state, %arg2: workgroup_state)
    for (unsigned i = 0; i <= 2; ++i) {
      Attribute unit = rewriter.getUnitAttr();
      llvmFuncOp.setArgAttr(i, LLVM::LLVMDialect::getNoAliasAttrName(), unit);
      llvmFuncOp.setArgAttr(i, LLVM::LLVMDialect::getNonNullAttrName(), unit);
      llvmFuncOp.setArgAttr(i, LLVM::LLVMDialect::getNoUndefAttrName(), unit);
      llvmFuncOp.setArgAttr(i, LLVM::LLVMDialect::getAlignAttrName(),
                            rewriter.getI64IntegerAttr(16));
    }

    // Add default zero return value.
    // TODO(ataei): do something meaningful with the return value; non-zero will
    // have the runtime bail out with an error.
    for (auto returnOp : llvm::make_early_inc_range(
             llvmFuncOp.getOps<mlir::func::ReturnOp>())) {
      rewriter.setInsertionPoint(returnOp);
      auto returnValue = rewriter.createOrFold<mlir::arith::ConstantIntOp>(
          returnOp.getLoc(), 0, 32);
      rewriter.replaceOpWithNewOp<mlir::func::ReturnOp>(returnOp, returnValue);
    }

    // Populate debug info for the subprogram signature. This is required in
    // order to get any debug information (including just line tables) from MLIR
    // into LLVM IR.
    auto scopeAttr = HexagonDispatchABI::buildScopeAttr(
        llvmFuncOp->getParentOfType<mlir::ModuleOp>(), llvmFuncOp,
        getTypeConverter());
    llvmFuncOp->setLoc(FusedLoc::get(llvmFuncOp.getContext(),
                                     {llvmFuncOp->getLoc()}, scopeAttr));

    rewriter.eraseOp(stdFuncOp);
    return success();
  }
};

/// Rewrites hal.interface.constant.load to ops loading from the ABI structs.
/// Because ordinals are not yet available we emit a placeholder global that
/// later gets updated with the value after linking.
///
/// The parent LLVMFuncOp must be compatible with HexagonDispatchABI.
struct ConvertHALExecutableConstantLoadOp
    : public ConvertOpToLLVMWithABIPattern<
          IREE::HAL::ExecutableConstantLoadOp> {
  using ConvertOpToLLVMWithABIPattern::ConvertOpToLLVMWithABIPattern;
  LogicalResult
  matchAndRewrite(IREE::HAL::ExecutableConstantLoadOp loadOp,
                  IREE::HAL::ExecutableConstantLoadOpAdaptor operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto resultType =
        typeConverter->convertType(loadOp->getResult(0).getType());
    rewriter.replaceOp(loadOp,
                       abi.loadExecutableConstant(loadOp, loadOp.getKey(),
                                                  resultType, rewriter));
    return success();
  }
};

/// Rewrites hal.interface.workgroup.id to ops loading from the ABI structs.
///
/// The parent LLVMFuncOp must be compatible with HexagonDispatchABI.
struct ConvertHALInterfaceWorkgroupIDOp
    : public ConvertOpToLLVMWithABIPattern<IREE::HAL::InterfaceWorkgroupIDOp> {
  using ConvertOpToLLVMWithABIPattern::ConvertOpToLLVMWithABIPattern;
  LogicalResult
  matchAndRewrite(IREE::HAL::InterfaceWorkgroupIDOp idOp,
                  IREE::HAL::InterfaceWorkgroupIDOpAdaptor operands,
                  ConversionPatternRewriter &rewriter) const override {
    int32_t dim = (int32_t)idOp.getDimension().getZExtValue();
    auto resultType = typeConverter->convertType(idOp->getResult(0).getType());
    rewriter.replaceOp(idOp,
                       abi.loadWorkgroupID(idOp, dim, resultType, rewriter));
    return success();
  }
};

/// Rewrites hal.interface.workgroup.size to ops loading from the ABI structs.
///
/// The parent LLVMFuncOp must be compatible with HexagonDispatchABI.
struct ConvertHALInterfaceWorkgroupSizeOp
    : public ConvertOpToLLVMWithABIPattern<
          IREE::HAL::InterfaceWorkgroupSizeOp> {
  using ConvertOpToLLVMWithABIPattern::ConvertOpToLLVMWithABIPattern;
  LogicalResult
  matchAndRewrite(IREE::HAL::InterfaceWorkgroupSizeOp sizeOp,
                  IREE::HAL::InterfaceWorkgroupSizeOpAdaptor operands,
                  ConversionPatternRewriter &rewriter) const override {
    int32_t dim = (int32_t)sizeOp.getDimension().getZExtValue();
    auto resultType =
        typeConverter->convertType(sizeOp->getResult(0).getType());
    rewriter.replaceOp(
        sizeOp, abi.loadWorkgroupSize(sizeOp, dim, resultType, rewriter));
    return success();
  }
};

/// Rewrites hal.interface.workgroup.count to ops loading from the ABI structs.
///
/// The parent LLVMFuncOp must be compatible with HexagonDispatchABI.
struct ConvertHALInterfaceWorkgroupCountOp
    : public ConvertOpToLLVMWithABIPattern<
          IREE::HAL::InterfaceWorkgroupCountOp> {
  using ConvertOpToLLVMWithABIPattern::ConvertOpToLLVMWithABIPattern;
  LogicalResult
  matchAndRewrite(IREE::HAL::InterfaceWorkgroupCountOp countOp,
                  IREE::HAL::InterfaceWorkgroupCountOpAdaptor operands,
                  ConversionPatternRewriter &rewriter) const override {
    int32_t dim = (int32_t)countOp.getDimension().getZExtValue();
    auto resultType =
        typeConverter->convertType(countOp->getResult(0).getType());
    rewriter.replaceOp(
        countOp, abi.loadWorkgroupCount(countOp, dim, resultType, rewriter));
    return success();
  }
};

/// Rewrites hal.interface.constant.load to ops loading from the ABI structs.
///
/// The parent LLVMFuncOp must be compatible with HexagonDispatchABI.
struct ConvertHALInterfaceConstantLoadOp
    : public ConvertOpToLLVMWithABIPattern<IREE::HAL::InterfaceConstantLoadOp> {
  using ConvertOpToLLVMWithABIPattern::ConvertOpToLLVMWithABIPattern;
  LogicalResult
  matchAndRewrite(IREE::HAL::InterfaceConstantLoadOp loadOp,
                  IREE::HAL::InterfaceConstantLoadOpAdaptor operands,
                  ConversionPatternRewriter &rewriter) const override {
    int64_t index = loadOp.getOrdinal().getZExtValue();
    auto resultType =
        typeConverter->convertType(loadOp->getResult(0).getType());
    rewriter.replaceOp(
        loadOp, abi.loadPushConstant(loadOp, index, resultType, rewriter));
    return success();
  }
};

/// Rewrites hal.interface.binding.subspan to ops loading from the ABI structs.
///
/// The parent LLVMFuncOp must be compatible with HexagonDispatchABI.
struct ConvertHALInterfaceBindingSubspanOp
    : public ConvertOpToLLVMWithABIPattern<
          IREE::HAL::InterfaceBindingSubspanOp> {
  using ConvertOpToLLVMWithABIPattern::ConvertOpToLLVMWithABIPattern;
  LogicalResult
  matchAndRewrite(IREE::HAL::InterfaceBindingSubspanOp subspanOp,
                  IREE::HAL::InterfaceBindingSubspanOpAdaptor operands,
                  ConversionPatternRewriter &rewriter) const override {
    MemRefType memRefType =
        dyn_cast<MemRefType>(subspanOp->getResult(0).getType());
    if (!memRefType) {
      return rewriter.notifyMatchFailure(
          subspanOp,
          "failed to convert interface.binding.subspan result to memref type");
    }
    auto memRefDesc =
        abi.loadBinding(subspanOp, subspanOp.getBinding().getSExtValue(),
                        operands.getByteOffset(), memRefType,
                        operands.getDynamicDims(), rewriter);
    rewriter.replaceOp(subspanOp, {memRefDesc});
    return success();
  }
};

struct InstrumentationEntry {
  // !llvm.ptr<i8> pointing at the base of the ringbuffer.
  Value basePtr;
  // !llvm.ptr<i8> pointing at the start of the entry (basePtr + offset).
  Value entryPtr;
  // i64 offset within the ringbuffer of the entry.
  Value offset;
};

// entrySize must be 16-byte aligned
static InstrumentationEntry
acquireInstrumentationEntry(Location loc, Value buffer, Value bufferPtr,
                            Value entrySize, OpBuilder &builder) {
  auto i64Type = builder.getI64Type();
  auto bufferType = cast<MemRefType>(buffer.getType());
  int64_t totalBufferSize =
      (bufferType.getNumElements() * bufferType.getElementTypeBitWidth()) / 8;
  int64_t headOffset = totalBufferSize - 8;
  int64_t ringSize = totalBufferSize - IREE_INSTRUMENT_DISPATCH_PADDING;
  assert(llvm::isPowerOf2_64(ringSize) &&
         "ringbuffer storage size must be a power-of-two");

  Value basePtr = MemRefDescriptor(bufferPtr).alignedPtr(builder, loc);

  Value offsetIndex =
      LLVM::ConstantOp::create(builder, loc, i64Type, headOffset);
  auto i8Type = builder.getI8Type();
  Value offsetPtr = LLVM::GEPOp::create(
      builder, loc, basePtr.getType(), i8Type, basePtr, offsetIndex,
      /*noWrapFlags =*/LLVM::GEPNoWrapFlags::inbounds);
  Value rawOffset =
      LLVM::AtomicRMWOp::create(builder, loc, LLVM::AtomicBinOp::add, offsetPtr,
                                entrySize, LLVM::AtomicOrdering::monotonic);
  Value offsetMask =
      LLVM::ConstantOp::create(builder, loc, i64Type, ringSize - 1);
  Value wrappedOffset =
      LLVM::AndOp::create(builder, loc, rawOffset, offsetMask);

  Value entryPtr = LLVM::GEPOp::create(builder, loc, basePtr.getType(), i8Type,
                                       basePtr, wrappedOffset);

  return {basePtr, entryPtr, wrappedOffset};
}

static InstrumentationEntry appendInstrumentationEntry(
    Location loc, Value buffer, Value bufferPtr, LLVM::LLVMStructType entryType,
    ArrayRef<Value> entryValues, DataLayout &dataLayout, OpBuilder &builder) {
  auto i64Type = builder.getI64Type();

  Value entrySize = LLVM::ConstantOp::create(builder, loc, i64Type,
                                             dataLayout.getTypeSize(entryType));
  auto entry =
      acquireInstrumentationEntry(loc, buffer, bufferPtr, entrySize, builder);

  Value entryStruct = LLVM::UndefOp::create(builder, loc, entryType);
  for (auto entryValue : llvm::enumerate(entryValues)) {
    entryStruct = LLVM::InsertValueOp::create(
        builder, loc, entryStruct, entryValue.value(), entryValue.index());
  }

  LLVM::StoreOp::create(
      builder, loc, entryStruct,
      LLVM::BitcastOp::create(builder, loc,
                              LLVM::LLVMPointerType::get(builder.getContext()),
                              entry.entryPtr),
      /*alignment=*/16);

  return entry;
}

// The workgroup key returned by `hal.instrument.workgroup` is the ring-buffer
// offset of the workgroup record, carried as an `index`. The offset always fits
// the index type, while the record header field below needs 64 bits. Returns
// the header bits: the offset in the upper 40 bits, as consumers expect.
static Value getWorkgroupKeyHeaderBits(Location loc, Value workgroupKey,
                                       OpBuilder &builder) {
  auto i64Type = builder.getI64Type();
  Value key = workgroupKey;
  if (key.getType() != i64Type) {
    key = LLVM::ZExtOp::create(builder, loc, i64Type, key);
  }
  return LLVM::ShlOp::create(
      builder, loc,
      LLVM::AndOp::create(
          builder, loc, key,
          LLVM::ConstantOp::create(builder, loc, i64Type, 0xFFFFFFFFFFll)),
      LLVM::ConstantOp::create(builder, loc, i64Type, 24));
}

static int64_t getMemoryAccessByteSize(Type type) {
  if (auto vectorType = dyn_cast<VectorType>(type)) {
    return (vectorType.getNumElements() * vectorType.getElementTypeBitWidth()) /
           8;
  } else {
    return type.getIntOrFloatBitWidth() / 8;
  }
}

struct ConvertHALInstrumentWorkgroupOp
    : public ConvertOpToLLVMWithABIPattern<IREE::HAL::InstrumentWorkgroupOp> {
  using ConvertOpToLLVMWithABIPattern::ConvertOpToLLVMWithABIPattern;
  LogicalResult
  matchAndRewrite(IREE::HAL::InstrumentWorkgroupOp instrumentOp,
                  IREE::HAL::InstrumentWorkgroupOpAdaptor operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = instrumentOp.getLoc();
    auto dataLayout =
        getTypeConverter()->getDataLayoutAnalysis()->getAbove(instrumentOp);
    auto i32Type = rewriter.getI32Type();

    auto entryType = LLVM::LLVMStructType::getLiteral(
        getContext(), {
                          i32Type, // header
                          i32Type, // workgroup_id_x
                          i32Type, // workgroup_id_y
                          i32Type, // workgroup_id_z
                          i32Type, // workgroup_count_x
                          i32Type, // workgroup_count_y
                          i32Type, // workgroup_count_z
                          i32Type, // processor_id
                      });

    // 8 bit tag = 00 | 24 bit dispatch id
    // NOTE: we could pre-shift this to avoid needing to do it in each group.
    // We just need to do the shift - the bottom two bits will be the 00 tag.
    Value rawDispatchId = instrumentOp.getDispatchId();
    Value header = LLVM::ShlOp::create(
        rewriter, loc, i32Type, rawDispatchId,
        LLVM::ConstantOp::create(rewriter, loc, i32Type, 8)); // | 8bit tag

    auto entry = appendInstrumentationEntry(
        loc, instrumentOp.getBuffer(), operands.getBuffer(), entryType,
        {
            header,
            abi.loadWorkgroupID(instrumentOp, 0, i32Type, rewriter),
            abi.loadWorkgroupID(instrumentOp, 1, i32Type, rewriter),
            abi.loadWorkgroupID(instrumentOp, 2, i32Type, rewriter),
            abi.loadWorkgroupCount(instrumentOp, 0, i32Type, rewriter),
            abi.loadWorkgroupCount(instrumentOp, 1, i32Type, rewriter),
            abi.loadWorkgroupCount(instrumentOp, 2, i32Type, rewriter),
            abi.loadProcessorID(instrumentOp, rewriter),
        },
        dataLayout, rewriter);

    // The key is the record offset in the index type; the consumers turn it
    // into header bits with `getWorkgroupKeyHeaderBits`.
    Type indexType = getTypeConverter()->getIndexType();
    Value workgroupKey = entry.offset;
    if (indexType != workgroupKey.getType()) {
      workgroupKey =
          LLVM::TruncOp::create(rewriter, loc, indexType, entry.offset);
    }
    rewriter.replaceOp(instrumentOp, workgroupKey);
    return success();
  }
};

static std::optional<uint64_t> mapValueType(Type type) {
  return TypeSwitch<Type, std::optional<uint64_t>>(type)
      .Case<IntegerType>([&](Type type) -> std::optional<uint64_t> {
        if (type.isUnsignedInteger()) {
          switch (type.getIntOrFloatBitWidth()) {
          case 8:
            return IREE_INSTRUMENT_DISPATCH_VALUE_TYPE_UINT_8;
          case 16:
            return IREE_INSTRUMENT_DISPATCH_VALUE_TYPE_UINT_16;
          case 32:
            return IREE_INSTRUMENT_DISPATCH_VALUE_TYPE_UINT_32;
          case 64:
            return IREE_INSTRUMENT_DISPATCH_VALUE_TYPE_UINT_64;
          default:
            return std::nullopt;
          }
        }
        switch (type.getIntOrFloatBitWidth()) {
        case 8:
          return IREE_INSTRUMENT_DISPATCH_VALUE_TYPE_SINT_8;
        case 16:
          return IREE_INSTRUMENT_DISPATCH_VALUE_TYPE_SINT_16;
        case 32:
          return IREE_INSTRUMENT_DISPATCH_VALUE_TYPE_SINT_32;
        case 64:
          return IREE_INSTRUMENT_DISPATCH_VALUE_TYPE_SINT_64;
        default:
          return std::nullopt;
        }
      })
      .Case<FloatType>([&](Type type) -> std::optional<uint64_t> {
        if (type.isBF16()) {
          return IREE_INSTRUMENT_DISPATCH_VALUE_TYPE_BFLOAT_16;
        }
        switch (type.getIntOrFloatBitWidth()) {
        case 16:
          return IREE_INSTRUMENT_DISPATCH_VALUE_TYPE_FLOAT_16;
        case 32:
          return IREE_INSTRUMENT_DISPATCH_VALUE_TYPE_FLOAT_32;
        case 64:
          return IREE_INSTRUMENT_DISPATCH_VALUE_TYPE_FLOAT_64;
        default:
          return std::nullopt;
        }
      })
      .Case<IndexType>([&](Type type) -> std::optional<uint64_t> {
        return IREE_INSTRUMENT_DISPATCH_VALUE_TYPE_SINT_64;
      })
      .Default([&](Type) -> std::optional<uint64_t> { return std::nullopt; });
}

struct ConvertHALInstrumentValueOp
    : public ConvertOpToLLVMWithABIPattern<IREE::HAL::InstrumentValueOp> {
  using ConvertOpToLLVMWithABIPattern::ConvertOpToLLVMWithABIPattern;
  LogicalResult
  matchAndRewrite(IREE::HAL::InstrumentValueOp instrumentOp,
                  IREE::HAL::InstrumentValueOpAdaptor operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = instrumentOp.getLoc();

    // Only convert ops we can handle, otherwise warn and discard.
    std::optional<uint64_t> valueType;
    if (isa<LLVM::LLVMPointerType>(operands.getOperand().getType())) {
      valueType = IREE_INSTRUMENT_DISPATCH_VALUE_TYPE_POINTER;
    } else {
      valueType = mapValueType(instrumentOp.getType());
    }
    if (!valueType) {
      mlir::emitWarning(loc,
                        "skipping hal.instrument.value on unsupported type: ")
          << instrumentOp.getType();
      rewriter.replaceOp(instrumentOp, {operands.getOperand()});
      return success();
    }

    auto dataLayout =
        getTypeConverter()->getDataLayoutAnalysis()->getAbove(instrumentOp);
    auto i64Type = rewriter.getI64Type();

    auto entryType =
        LLVM::LLVMStructType::getLiteral(getContext(), {
                                                           i64Type, // header
                                                           i64Type, // value
                                                       });

    // 8 bit tag
    // 8 bit type
    // 8 bit ordinal
    // 40 bit workgroup offset
    Value header = LLVM::OrOp::create(
        rewriter, loc,
        getWorkgroupKeyHeaderBits(loc, operands.getWorkgroupKey(), rewriter),
        LLVM::ConstantOp::create(
            rewriter, loc, i64Type,
            (instrumentOp.getOrdinal().getZExtValue() << 16) |
                (valueType.value() << 8) |
                IREE_INSTRUMENT_DISPATCH_TYPE_VALUE));

    // Bitcast to an integer and widen to 64 bits.
    Value bits = LLVM::ZExtOp::create(
        rewriter, loc, i64Type,
        LLVM::BitcastOp::create(
            rewriter, loc,
            rewriter.getIntegerType(
                instrumentOp.getType().getIntOrFloatBitWidth()),
            operands.getOperand()));

    appendInstrumentationEntry(loc, instrumentOp.getBuffer(),
                               operands.getBuffer(), entryType,
                               {
                                   header,
                                   bits,
                               },
                               dataLayout, rewriter);

    rewriter.replaceOp(instrumentOp, operands.getOperand());
    return success();
  }
};

struct ConvertHALInstrumentMemoryLoadOp
    : public ConvertOpToLLVMWithABIPattern<IREE::HAL::InstrumentMemoryLoadOp> {
  using ConvertOpToLLVMWithABIPattern::ConvertOpToLLVMWithABIPattern;
  LogicalResult
  matchAndRewrite(IREE::HAL::InstrumentMemoryLoadOp instrumentOp,
                  IREE::HAL::InstrumentMemoryLoadOpAdaptor operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = instrumentOp.getLoc();
    auto dataLayout =
        getTypeConverter()->getDataLayoutAnalysis()->getAbove(instrumentOp);
    auto i64Type = rewriter.getI64Type();

    auto entryType =
        LLVM::LLVMStructType::getLiteral(getContext(), {
                                                           i64Type, // header
                                                           i64Type, // address
                                                       });

    // 8 bit tag = 100 (read), 101 (write)
    // 16 bit length
    // 40 bit workgroup offset
    int64_t loadSize = getMemoryAccessByteSize(instrumentOp.getType());
    assert(loadSize <= UINT16_MAX && "16-bit length maximum");
    Value header = LLVM::OrOp::create(
        rewriter, loc,
        getWorkgroupKeyHeaderBits(loc, operands.getWorkgroupKey(), rewriter),
        LLVM::ConstantOp::create(
            rewriter, loc, i64Type,
            (loadSize << 8) | IREE_INSTRUMENT_DISPATCH_TYPE_MEMORY_LOAD));

    Value loadPtr = getStridedElementPtr(
        rewriter, loc, cast<MemRefType>(instrumentOp.getBase().getType()),
        operands.getBase(), operands.getIndices());
    Value addressI64 =
        LLVM::PtrToIntOp::create(rewriter, loc, i64Type, loadPtr);

    appendInstrumentationEntry(loc, instrumentOp.getBuffer(),
                               operands.getBuffer(), entryType,
                               {
                                   header,
                                   addressI64,
                               },
                               dataLayout, rewriter);

    rewriter.replaceOp(instrumentOp, operands.getLoadValue());
    return success();
  }
};

struct ConvertHALInstrumentMemoryStoreOp
    : public ConvertOpToLLVMWithABIPattern<IREE::HAL::InstrumentMemoryStoreOp> {
  using ConvertOpToLLVMWithABIPattern::ConvertOpToLLVMWithABIPattern;
  LogicalResult
  matchAndRewrite(IREE::HAL::InstrumentMemoryStoreOp instrumentOp,
                  IREE::HAL::InstrumentMemoryStoreOpAdaptor operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = instrumentOp.getLoc();
    auto dataLayout =
        getTypeConverter()->getDataLayoutAnalysis()->getAbove(instrumentOp);
    auto i64Type = rewriter.getI64Type();

    auto entryType =
        LLVM::LLVMStructType::getLiteral(getContext(), {
                                                           i64Type, // header
                                                           i64Type, // address
                                                       });

    // 8 bit tag = 10 (read), 11 (write)
    // 16 bit length
    // 40 bit workgroup offset
    int64_t storeSize = getMemoryAccessByteSize(instrumentOp.getType());
    assert(storeSize <= UINT16_MAX && "16-bit length maximum");
    Value header = LLVM::OrOp::create(
        rewriter, loc,
        getWorkgroupKeyHeaderBits(loc, operands.getWorkgroupKey(), rewriter),
        LLVM::ConstantOp::create(
            rewriter, loc, i64Type,
            (storeSize << 8) | IREE_INSTRUMENT_DISPATCH_TYPE_MEMORY_STORE));

    Value storePtr = getStridedElementPtr(
        rewriter, loc, cast<MemRefType>(instrumentOp.getBase().getType()),
        operands.getBase(), operands.getIndices());
    Value addressI64 =
        LLVM::PtrToIntOp::create(rewriter, loc, i64Type, storePtr);

    appendInstrumentationEntry(loc, instrumentOp.getBuffer(),
                               operands.getBuffer(), entryType,
                               {
                                   header,
                                   addressI64,
                               },
                               dataLayout, rewriter);

    rewriter.replaceOp(instrumentOp, operands.getStoreValue());
    return success();
  }
};

} // namespace

void populateHexagonABIToLLVMConversionPatterns(
    HexagonDispatchABI &abi, LLVMTypeConverter &typeConverter,
    RewritePatternSet &patterns) {
  patterns.addWithLabel<
      ConvertHALEntryPointFuncOp, ConvertHALExecutableConstantLoadOp,
      ConvertHALInterfaceWorkgroupIDOp, ConvertHALInterfaceWorkgroupSizeOp,
      ConvertHALInterfaceWorkgroupCountOp, ConvertHALInterfaceConstantLoadOp>(
      {"iree-hal-abi-to-llvm"}, abi, typeConverter);
  patterns.addWithLabel<ConvertHALInterfaceBindingSubspanOp>(
      {"iree-hal-abi-to-llvm"}, abi, typeConverter);
}

void populateHexagonInstrumentationToLLVMConversionPatterns(
    HexagonDispatchABI &abi, LLVMTypeConverter &typeConverter,
    RewritePatternSet &patterns) {
  patterns.addWithLabel<
      ConvertHALInstrumentWorkgroupOp, ConvertHALInstrumentValueOp,
      ConvertHALInstrumentMemoryLoadOp, ConvertHALInstrumentMemoryStoreOp>(
      {"iree-hal-abi-to-llvm"}, abi, typeConverter);
}

} // namespace mlir::iree_compiler::hexagon::codegen

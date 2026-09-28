// Copyright 2023 The IREE Authors
// Copyright 2026 RooflineAI GmbH
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "hexagon/CodeGen/Conversion/HexagonDispatchABI.h"

#include "mlir/IR/Matchers.h"
#include "llvm/BinaryFormat/Dwarf.h"
#include "llvm/Support/Path.h"

namespace mlir::iree_compiler::hexagon::codegen {

//------------------------------------------------------------------------------
// HexagonDispatchABI
//------------------------------------------------------------------------------

// static
llvm::sys::Mutex HexagonDispatchABI::sMutex;

// static
LLVM::LLVMStructType
HexagonDispatchABI::getProcessorType(MLIRContext *context) {
  llvm::sys::ScopedLock lock(sMutex);
  auto structType =
      LLVM::LLVMStructType::getIdentified(context, "iree_hal_processor_v0_t");
  if (structType.isInitialized()) {
    return structType;
  }

  auto uint64Type = IntegerType::get(context, 64);
  SmallVector<Type> fieldTypes;

  // uint64_t data[IREE_HAL_PROCESSOR_DATA_CAPACITY_V0];
  fieldTypes.push_back(
      LLVM::LLVMArrayType::get(uint64Type, ProcessorDataCapacity));

  LogicalResult bodySet = structType.setBody(fieldTypes, /*isPacked=*/false);
  assert(succeeded(bodySet) &&
         "could not set the body of an identified struct");
  (void)bodySet;

  return structType;
}

// static
LLVM::LLVMStructType
HexagonDispatchABI::getEnvironmentType(MLIRContext *context,
                                       LLVM::LLVMStructType processorType) {
  llvm::sys::ScopedLock lock(sMutex);
  auto structType = LLVM::LLVMStructType::getIdentified(
      context, "iree_hal_executable_environment_v0_t");
  if (structType.isInitialized()) {
    return structType;
  }

  auto opaquePtrType = LLVM::LLVMPointerType::get(context);
  SmallVector<Type> fieldTypes;

  // const uint32_t* constants;
  fieldTypes.push_back(opaquePtrType);

  // iree_hal_executable_import_thunk_v0_t import_thunk;
  // const iree_hal_executable_import_v0_t* import_funcs;
  // const void** import_contexts;
  fieldTypes.push_back(LLVM::LLVMPointerType::get(context));
  fieldTypes.push_back(LLVM::LLVMPointerType::get(context));
  fieldTypes.push_back(LLVM::LLVMPointerType::get(context));

  // iree_hal_processor_v0_t processor;
  fieldTypes.push_back(processorType);

  LogicalResult bodySet = structType.setBody(fieldTypes, /*isPacked=*/false);
  assert(succeeded(bodySet) &&
         "could not set the body of an identified struct");
  (void)bodySet;

  return structType;
}

// static
LLVM::LLVMStructType
HexagonDispatchABI::getDispatchStateType(MLIRContext *context) {
  llvm::sys::ScopedLock lock(sMutex);
  auto structType = LLVM::LLVMStructType::getIdentified(
      context, "iree_hal_executable_dispatch_state_v0_t");
  if (structType.isInitialized()) {
    return structType;
  }

  auto uint8Type = IntegerType::get(context, 8);
  auto uint16Type = IntegerType::get(context, 16);
  auto uint32Type = IntegerType::get(context, 32);
  auto opaquePtrType = LLVM::LLVMPointerType::get(context);
  SmallVector<Type> fieldTypes;

  // uint32_t workgroup_size_x;
  // uint32_t workgroup_size_y;
  // uint16_t workgroup_size_z;
  fieldTypes.push_back(uint32Type);
  fieldTypes.push_back(uint32Type);
  fieldTypes.push_back(uint16Type);

  // uint16_t constant_count;
  fieldTypes.push_back(uint16Type);

  // uint32_t workgroup_count_x;
  // uint32_t workgroup_count_y;
  // uint16_t workgroup_count_z;
  fieldTypes.push_back(uint32Type);
  fieldTypes.push_back(uint32Type);
  fieldTypes.push_back(uint16Type);

  // uint8_t max_concurrency;
  fieldTypes.push_back(uint8Type);

  // uint8_t binding_count;
  fieldTypes.push_back(uint8Type);

  // const uint32_t * constants;
  // void *const * binding_ptrs;
  // const size_t * binding_lengths;
  fieldTypes.push_back(opaquePtrType);
  fieldTypes.push_back(opaquePtrType);
  fieldTypes.push_back(opaquePtrType);

  LogicalResult bodySet = structType.setBody(fieldTypes, /*isPacked=*/false);
  assert(succeeded(bodySet) &&
         "could not set the body of an identified struct");
  (void)bodySet;

  return structType;
}

// static
LLVM::LLVMStructType
HexagonDispatchABI::getExtendedDispatchStateType(MLIRContext *context) {
  return LLVM::LLVMStructType::getLiteral(
      context,
      {getDispatchStateType(context), LLVM::LLVMPointerType::get(context)});
}

// static
LLVM::LLVMStructType
HexagonDispatchABI::getWorkgroupStateType(MLIRContext *context) {
  llvm::sys::ScopedLock lock(sMutex);
  auto structType = LLVM::LLVMStructType::getIdentified(
      context, "iree_hal_executable_workgroup_state_v0_t");
  if (structType.isInitialized()) {
    return structType;
  }

  auto uint16Type = IntegerType::get(context, 16);
  auto uint32Type = IntegerType::get(context, 32);
  auto opaquePtrType = LLVM::LLVMPointerType::get(context);
  SmallVector<Type> fieldTypes;

  // uint32_t workgroup_id_x;
  // uint32_t workgroup_id_y;
  // uint16_t workgroup_id_z;
  fieldTypes.push_back(uint32Type);
  fieldTypes.push_back(uint32Type);
  fieldTypes.push_back(uint16Type);

  // uint16_t reserved;
  fieldTypes.push_back(uint16Type);

  // uint32_t processor_id;
  fieldTypes.push_back(uint32Type);

  // void* local_memory;
  // uint32_t local_memory_size;
  fieldTypes.push_back(opaquePtrType);
  fieldTypes.push_back(uint32Type);

  LogicalResult bodySet = structType.setBody(fieldTypes, /*isPacked=*/false);
  assert(succeeded(bodySet) &&
         "could not set the body of an identified struct");
  (void)bodySet;

  return structType;
}

// static
SmallVector<Type, 5> HexagonDispatchABI::getInputTypes(MLIRContext *context) {
  return SmallVector<Type, 5>{
      // const iree_hal_executable_environment_v0_t* IREE_RESTRICT
      //   environment
      LLVM::LLVMPointerType::get(context),
      // const iree_hal_executable_dispatch_state_v0_t* IREE_RESTRICT
      //   dispatch_state
      LLVM::LLVMPointerType::get(context),
      // const iree_hal_executable_workgroup_state_v0_t* IREE_RESTRICT
      //   workgroup_state
      LLVM::LLVMPointerType::get(context),
  };
}

// static
LLVM::DISubprogramAttr
HexagonDispatchABI::buildScopeAttr(mlir::ModuleOp moduleOp,
                                   LLVM::LLVMFuncOp llvmFuncOp,
                                   const LLVMTypeConverter *typeConverter) {
  auto *context = &typeConverter->getContext();
  Builder builder(context);

  std::string inputFilePath("-");
  if (auto fileLoc = dyn_cast<mlir::FileLineColLoc>(moduleOp.getLoc())) {
    inputFilePath = fileLoc.getFilename().getValue();
  }

  auto fileAttr =
      LLVM::DIFileAttr::get(context, llvm::sys::path::filename(inputFilePath),
                            llvm::sys::path::parent_path(inputFilePath));
  auto compileUnitAttr = LLVM::DICompileUnitAttr::get(
      DistinctAttr::create(UnitAttr::get(context)), llvm::dwarf::DW_LANG_C17,
      fileAttr, builder.getStringAttr("IREE"),
      /*isOptimized=*/true, LLVM::DIEmissionKind::Full);

  auto int32TypeAttr =
      LLVM::DIBasicTypeAttr::get(context, llvm::dwarf::DW_TAG_base_type, "int",
                                 /*sizeInBits=*/32, llvm::dwarf::DW_ATE_signed);
  auto voidTypeAttr =
      LLVM::DIBasicTypeAttr::get(context, llvm::dwarf::DW_TAG_base_type, "void",
                                 /*sizeInBits=*/0, llvm::dwarf::DW_ATE_address);
  auto opaquePtrTypeAttr = LLVM::DIDerivedTypeAttr::get(
      context, llvm::dwarf::DW_TAG_pointer_type,
      /*name=*/nullptr, /*file=*/nullptr, /*line=*/0, /*scope=*/nullptr,
      voidTypeAttr, /*sizeInBits=*/typeConverter->getPointerBitwidth(),
      /*alignInBits=*/0, /*offsetInBits=*/0,
      /*dwarfAddressSpace=*/std::nullopt, LLVM::DIFlags::Zero,
      /*extraData=*/nullptr);
  auto subroutineTypeAttr = LLVM::DISubroutineTypeAttr::get(
      context, llvm::dwarf::DW_CC_normal,
      {int32TypeAttr, opaquePtrTypeAttr, opaquePtrTypeAttr, opaquePtrTypeAttr});

  auto funcNameAttr = builder.getStringAttr(llvmFuncOp.getName());
  DistinctAttr id;
  if (!llvmFuncOp.isExternal()) {
    id = DistinctAttr::create(UnitAttr::get(context));
  }
  return LLVM::DISubprogramAttr::get(
      context, id, compileUnitAttr, fileAttr, funcNameAttr, funcNameAttr,
      fileAttr,
      /*line=*/1,
      /*scopeline=*/1,
      LLVM::DISubprogramFlags::Definition | LLVM::DISubprogramFlags::Optimized,
      subroutineTypeAttr, /*retainedNodes =*/{}, /*annotations =*/{});
}

// Returns the argument at |argIndex| in the parent function of |forOp|.
static Value getLocalArgument(Operation *forOp, unsigned argIndex) {
  auto funcOp = forOp->getParentOfType<LLVM::LLVMFuncOp>();
  assert(funcOp && "usage requires an enclosing LLVMFuncOp");
  return funcOp.getArgument(argIndex);
}

Value HexagonDispatchABI::loadWorkgroupID(Operation *forOp, int32_t dim,
                                          Type resultType, OpBuilder &builder) {
  auto dimValue =
      loadFieldValue(forOp, WorkgroupStateField::workgroup_id_x + dim, builder);
  return castValueToType(forOp->getLoc(), dimValue, resultType, builder);
}

Value HexagonDispatchABI::loadWorkgroupCount(Operation *forOp, int32_t dim,
                                             Type resultType,
                                             OpBuilder &builder) {
  auto dimValue = loadFieldValue(
      forOp, DispatchStateField::workgroup_count_x + dim, builder);
  return castValueToType(forOp->getLoc(), dimValue, resultType, builder);
}

Value HexagonDispatchABI::loadWorkgroupSize(Operation *forOp, int32_t dim,
                                            Type resultType,
                                            OpBuilder &builder) {
  auto dimValue = loadFieldValue(
      forOp, DispatchStateField::workgroup_size_x + dim, builder);
  return castValueToType(forOp->getLoc(), dimValue, resultType, builder);
}

Value HexagonDispatchABI::loadPushConstant(Operation *forOp, int64_t offset,
                                           Type resultType,
                                           OpBuilder &builder) {
  auto loc = forOp->getLoc();
  auto constantsPtrValue =
      loadFieldValue(forOp, DispatchStateField::constants, builder);
  auto pushConstantType = IntegerType::get(context, 32);
  Value constantPtrValue = LLVM::GEPOp::create(
      builder, loc, constantsPtrValue.getType(), pushConstantType,
      constantsPtrValue, LLVM::GEPArg(int32_t(offset)));
  Value constantValue =
      LLVM::LoadOp::create(builder, loc, pushConstantType, constantPtrValue);
  return castValueToType(loc, constantValue, resultType, builder);
}

Value HexagonDispatchABI::loadBindingPtr(Operation *forOp, int64_t ordinal,
                                         OpBuilder &builder) {
  auto loc = forOp->getLoc();
  auto ptrsPtrValue =
      loadFieldValue(forOp, DispatchStateField::binding_ptrs, builder);
  auto elementPtrValue = LLVM::GEPOp::create(
      builder, loc, ptrsPtrValue.getType(),
      mlir::LLVM::LLVMPointerType::get(builder.getContext()), ptrsPtrValue,
      LLVM::GEPArg(int32_t(ordinal)));
  return LLVM::LoadOp::create(
      builder, loc, mlir::LLVM::LLVMPointerType::get(builder.getContext()),
      elementPtrValue);
}

MemRefDescriptor
HexagonDispatchABI::loadBinding(Operation *forOp, int64_t ordinal,
                                Value baseOffsetValue, MemRefType memRefType,
                                ValueRange dynamicDims, OpBuilder &builder) {
  auto loc = forOp->getLoc();

  // Load the base buffer pointer in the appropriate type (f32*, etc).
  Value basePtrValue = loadBindingPtr(forOp, ordinal, builder);

  // Apply the subspan byte offset to the base pointer. The memref descriptor
  // offset below is the memref layout offset, not the HAL binding byte offset.
  if (baseOffsetValue && !matchPattern(baseOffsetValue, m_Zero())) {
    basePtrValue =
        LLVM::GEPOp::create(builder, loc, basePtrValue.getType(),
                            builder.getI8Type(), basePtrValue, baseOffsetValue);
  }

  // NOTE: if we wanted to check the range was in bounds here would be the
  // place to do it.

  // Construct the MemRefDescriptor type based on the information we have.
  // NOTE: we could use the binding length to clamp this/check that the
  // requested range is valid.
  auto [strides, offset] = memRefType.getStridesAndOffset();
  if (memRefType.hasStaticShape() &&
      llvm::none_of(strides, ShapedType::isDynamic) &&
      ShapedType::isStatic(offset)) {
    return MemRefDescriptor::fromStaticShape(builder, loc, *typeConverter,
                                             memRefType, basePtrValue);
  } else {
    assert(memRefType.getNumDynamicDims() == dynamicDims.size());
    int64_t rank = memRefType.getRank();

    // Build MemRef descriptor for this interface binding.
    auto desc = MemRefDescriptor::poison(
        builder, loc, typeConverter->convertType(memRefType));
    desc.setAllocatedPtr(builder, loc, basePtrValue);
    desc.setAlignedPtr(builder, loc, basePtrValue);
    auto llvmIndexType = typeConverter->convertType(builder.getIndexType());
    if (ShapedType::isDynamic(offset)) {
      desc.setOffset(builder, loc,
                     LLVM::ConstantOp::create(builder, loc, llvmIndexType, 0));
    } else {
      desc.setConstantOffset(builder, loc, offset);
    }

    // Update memref descriptor shape. Dynamic dimensions can be mixed with
    // static dimensions, like [128, ?, 128].
    int dynamicDimIndex = 0;
    for (int i = 0; i < rank; ++i) {
      if (memRefType.isDynamicDim(i)) {
        desc.setSize(builder, loc, i, dynamicDims[dynamicDimIndex++]);
      } else {
        desc.setConstantSize(builder, loc, i, memRefType.getDimSize(i));
      }
    }

    // Compute and update strides. Assume that MemRefs are row-major, that is,
    // following index linearization:
    //   x[i, j, k] = i * x.dim[1] * x.dim[2] + j * x.dim[2] + k
    if (!strides.empty()) {
      assert(strides.back() == 1 &&
             "unexpected non-unit stride for innermost dimension");
      desc.setConstantStride(builder, loc, rank - 1, 1);
      OpFoldResult currentStride = builder.getIndexAttr(1);
      for (int i = rank - 1; i > 0; --i) {
        if (ShapedType::isDynamic(strides[i - 1])) {
          auto dim = desc.size(builder, loc, i);
          Value currentStrideVal;
          if (std::optional<int64_t> currentStrideInt =
                  getConstantIntValue(currentStride)) {
            currentStrideVal = LLVM::ConstantOp::create(
                builder, loc, llvmIndexType, currentStrideInt.value());
          } else {
            currentStrideVal = cast<Value>(currentStride);
          }
          currentStride =
              LLVM::MulOp::create(builder, loc, currentStrideVal, dim)
                  .getResult();
          desc.setStride(builder, loc, i - 1, cast<Value>(currentStride));
        } else {
          currentStride = builder.getIndexAttr(strides[i - 1]);
          desc.setConstantStride(builder, loc, i - 1, strides[i - 1]);
        }
      }
    }

    return desc;
  }
}

Value HexagonDispatchABI::loadProcessorID(Operation *forOp,
                                          OpBuilder &builder) {
  return loadFieldValue(forOp, WorkgroupStateField::processor_id, builder);
}

Value HexagonDispatchABI::loadExecutableConstant(Operation *forOp,
                                                 StringRef key, Type resultType,
                                                 OpBuilder &builder) {
  auto loc = forOp->getLoc();

  // Create top-level global placeholder.
  // The magic attribute is used by future assignment passes.
  std::string globalName = ("__constant_ordinal_" + key).str();
  auto moduleOp =
      builder.getInsertionPoint()->getParentOfType<mlir::ModuleOp>();
  LLVM::GlobalOp globalOp;
  if (!(globalOp = moduleOp.lookupSymbol<LLVM::GlobalOp>(globalName))) {
    auto moduleBuilder = OpBuilder::atBlockBegin(moduleOp.getBody());
    globalOp = LLVM::GlobalOp::create(
        moduleBuilder, loc, builder.getI32Type(),
        /*isConstant=*/false, LLVM::Linkage::Internal, globalName, Attribute{});
    globalOp->setAttr(IREE::HAL::ExecutableConstantBlockOp::getKeyAttrName(),
                      builder.getStringAttr(key));
  }

  // Load the placeholder global ordinal.
  Value globalPtr = LLVM::AddressOfOp::create(builder, loc, globalOp);
  Value ordinalValue =
      LLVM::LoadOp::create(builder, loc, globalOp.getType(), globalPtr);

  // Load constant from the executable constants struct.
  auto constantsPtrValue =
      loadFieldValue(forOp, EnvironmentField::constants, builder);
  Value constantPtrValue =
      LLVM::GEPOp::create(builder, loc, constantsPtrValue.getType(), resultType,
                          constantsPtrValue, ordinalValue);
  Value constantValue =
      LLVM::LoadOp::create(builder, loc, resultType, constantPtrValue);
  return castValueToType(loc, constantValue, resultType, builder);
}

Value HexagonDispatchABI::castValueToType(Location loc, Value value,
                                          Type resultType, OpBuilder &builder) {
  // NOTE: we should handle more cases here (and proper sign extension).
  if (value.getType() == resultType) {
    return value;
  }
  return builder.createOrFold<LLVM::ZExtOp>(loc, resultType, value);
}

Value HexagonDispatchABI::loadRuntimeState(Operation *forOp,
                                           OpBuilder &builder) {
  Location loc = forOp->getLoc();
  Value dispatchState = getLocalArgument(forOp, /*argNum=*/1);
  auto ptrType = LLVM::LLVMPointerType::get(context);
  Value runtimeStateAddr = LLVM::GEPOp::create(
      builder, loc, ptrType, getExtendedDispatchStateType(context),
      dispatchState, ArrayRef<LLVM::GEPArg>{0, 1},
      LLVM::GEPNoWrapFlags::inbounds);
  return LLVM::LoadOp::create(builder, loc, ptrType, runtimeStateAddr);
}

Value HexagonDispatchABI::loadFieldValue(Operation *forOp,
                                         EnvironmentField field,
                                         OpBuilder &builder) {
  auto loc = forOp->getLoc();
  auto environmentPtrValue = getLocalArgument(forOp, 0);
  Value environmentValue =
      LLVM::LoadOp::create(builder, loc, environmentType, environmentPtrValue);
  SmallVector<int64_t, 1> position = {int64_t(field)};
  return LLVM::ExtractValueOp::create(builder, loc, environmentValue, position);
}

Value HexagonDispatchABI::loadFieldValue(Operation *forOp,
                                         DispatchStateField field,
                                         OpBuilder &builder) {
  auto loc = forOp->getLoc();
  auto statePtrValue = getLocalArgument(forOp, 1);
  Value stateValue =
      LLVM::LoadOp::create(builder, loc, dispatchStateType, statePtrValue);
  SmallVector<int64_t, 1> position = {int64_t(field)};
  return LLVM::ExtractValueOp::create(builder, loc, stateValue, position);
}

Value HexagonDispatchABI::loadFieldValue(Operation *forOp,
                                         WorkgroupStateField field,
                                         OpBuilder &builder) {
  auto loc = forOp->getLoc();
  auto statePtrValue = getLocalArgument(forOp, 2);
  Value stateValue =
      LLVM::LoadOp::create(builder, loc, workgroupStateType, statePtrValue);
  SmallVector<int64_t, 1> position = {int64_t(field)};
  return LLVM::ExtractValueOp::create(builder, loc, stateValue, position);
}

} // namespace mlir::iree_compiler::hexagon::codegen

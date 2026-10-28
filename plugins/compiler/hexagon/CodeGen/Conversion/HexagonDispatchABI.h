// Copyright 2023 The IREE Authors
// Copyright 2026 RooflineAI GmbH
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HEXAGON_PLUGINS_CODEGEN_CONVERSION_HEXAGONDISPATCHABI_H_
#define IREE_HEXAGON_PLUGINS_CODEGEN_CONVERSION_HEXAGONDISPATCHABI_H_

#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "mlir/Conversion/LLVMCommon/MemRefBuilder.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/Support/Mutex.h"

namespace mlir::iree_compiler::hexagon::codegen {

// Derived from:
//   iree/compiler/Codegen/LLVMCPU/DispatchABI.{h,cpp}
// IREE revision: a45adeaa6115e446c898e6eb21fb6edc0e65ddc4
//
// Hexagon owns this ABI; the environment, dispatch-state and workgroup-state
// layouts match hexagon/dsp/rt/executable_library.h
// The dispatch state also carries the Hexagon runtime state.

//------------------------------------------------------------------------------
// HexagonDispatchABI
//------------------------------------------------------------------------------

// Utility for accessing the IREE HAL dispatch function ABI values.
// All accesses should route through this vs directly manipulating LLVM function
// arguments so that we can adjust the ABI over time and support multiple
// versions in the same compiled output.
class HexagonDispatchABI {
public:
  // Matches the field order in hexagon_executable_environment_t.
  enum class EnvironmentField {
    constants,
  };

  // Returns a Type representing hexagon_executable_environment_t.
  static LLVM::LLVMStructType getEnvironmentType(MLIRContext *context);

  // Matches the field order in hexagon_executable_dispatch_state_t.
  enum class DispatchStateField {
    /*uint32_t*/ workgroup_size_x,
    /*uint32_t*/ workgroup_size_y,
    /*uint16_t*/ workgroup_size_z,
    /*uint32_t*/ workgroup_count_x,
    /*uint32_t*/ workgroup_count_y,
    /*uint16_t*/ workgroup_count_z,
    /*uint8_t*/ binding_count,
    /*intptr_t*/ constants,
    /*intptr_t*/ binding_ptrs,
    /*intptr_t*/ binding_lengths,
    /*intptr_t*/ prof_context,
    /*intptr_t*/ vtcm_partition,
  };
  friend DispatchStateField operator+(DispatchStateField lhs, int32_t rhs) {
    return static_cast<DispatchStateField>(static_cast<int32_t>(lhs) + rhs);
  }

  // Returns a Type representing hexagon_executable_dispatch_state_t.
  static LLVM::LLVMStructType getDispatchStateType(MLIRContext *context);

  enum class WorkgroupStateField {
    /*uint32_t*/ workgroup_id_x = 0,
    /*uint32_t*/ workgroup_id_y,
    /*uint16_t*/ workgroup_id_z,
    /*uint32_t*/ processor_id,
  };
  friend WorkgroupStateField operator+(WorkgroupStateField lhs, int32_t rhs) {
    return static_cast<WorkgroupStateField>(static_cast<int32_t>(lhs) + rhs);
  }

  // Returns a Type representing hexagon_executable_workgroup_state_t.
  static LLVM::LLVMStructType getWorkgroupStateType(MLIRContext *context);

  // Returns the types of the LLVM function inputs for the ABI.
  // This matches the signature of `hexagon_executable_dispatch_fn_t` in
  // `hexagon/dsp/rt/executable_library.h`.
  static SmallVector<Type, 5> getInputTypes(MLIRContext *context);

  // Builds a DISubprogram for |llvmFuncOp| function in |moduleOp|.
  // This is required in order to get any debug information (including line
  // tables) from MLIR into LLVM IR. It does not need to match the exact
  // definition but the closer we can make it to the real thing the more useful
  // downstream tools will be.
  static LLVM::DISubprogramAttr
  buildScopeAttr(mlir::ModuleOp moduleOp, LLVM::LLVMFuncOp llvmFuncOp,
                 const LLVMTypeConverter *typeConverter);

  explicit HexagonDispatchABI(LLVMTypeConverter *typeConverter)
      : context(&typeConverter->getContext()), typeConverter(typeConverter),
        environmentType(getEnvironmentType(context)),
        dispatchStateType(getDispatchStateType(context)),
        workgroupStateType(getWorkgroupStateType(context)) {}

  // Returns the runtime-state handle (argument)
  Value getRuntimeStateAtEntry(Operation *forOp);

  // Loads the workgroup_id[dim] value (XYZ) and casts it to |resultType|.
  Value loadWorkgroupID(Operation *forOp, int32_t dim, Type resultType,
                        OpBuilder &builder);

  // Loads the workgroup_count[dim] value (XYZ) and casts it to |resultType|.
  Value loadWorkgroupCount(Operation *forOp, int32_t dim, Type resultType,
                           OpBuilder &builder);

  // Loads the workgroup_size[dim] value (XYZ) and casts it to |resultType|.
  Value loadWorkgroupSize(Operation *forOp, int32_t dim, Type resultType,
                          OpBuilder &builder);

  // Loads a push constant at |offset| and casts it to |resultType|.
  Value loadPushConstant(Operation *forOp, int64_t offset, Type resultType,
                         OpBuilder &builder);

  // Loads the base pointer of the binding |ordinal| as an `i8**`.
  // Equivalent to:
  //   int8_t** base_ptr = &state->binding_ptrs[ordinal];
  Value loadBindingPtr(Operation *forOp, int64_t ordinal, OpBuilder &builder);

  // Loads a binding as a constructed MemRefDescriptor.
  // |baseOffset| can optionally adjust the base byte offset of the buffer.
  MemRefDescriptor loadBinding(Operation *forOp, int64_t ordinal,
                               Value baseOffsetValue, MemRefType memRefType,
                               ValueRange dynamicDims, OpBuilder &builder);

  // Loads the processor ID the code is (most likely) being run on.
  // Equivalent to:
  //   uint32_t processor_id = state->processor_id;
  Value loadProcessorID(Operation *forOp, OpBuilder &builder);

  // Loads an executable constant with |key| and casts it to |resultType|.
  // A placeholder global will be added for the ordinal.
  Value loadExecutableConstant(Operation *forOp, StringRef key, Type resultType,
                               OpBuilder &builder);

private:
  Value castValueToType(Location loc, Value value, Type resultType,
                        OpBuilder &builder);

  Value loadFieldValue(Operation *forOp, EnvironmentField field,
                       OpBuilder &builder);
  Value loadFieldValue(Operation *forOp, DispatchStateField field,
                       OpBuilder &builder);
  Value loadFieldValue(Operation *forOp, WorkgroupStateField field,
                       OpBuilder &builder);

  mlir::MLIRContext *context;
  const LLVMTypeConverter *typeConverter;
  LLVM::LLVMStructType environmentType;
  LLVM::LLVMStructType dispatchStateType;
  LLVM::LLVMStructType workgroupStateType;

  // Used to lock around mutations of shared LLVM type information, e.g.
  // mlir::LLVM::LLVMStructType::getIdentified.
  static llvm::sys::Mutex sMutex;
};

} // namespace mlir::iree_compiler::hexagon::codegen

#endif // IREE_HEXAGON_PLUGINS_CODEGEN_CONVERSION_HEXAGONDISPATCHABI_H_

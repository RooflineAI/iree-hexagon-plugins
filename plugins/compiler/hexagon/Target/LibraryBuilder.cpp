// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Derived from IREE's compiler/plugins/target/LLVMCPU/LibraryBuilder.cpp.
// See LibraryBuilder.h.

#include "hexagon/Target/LibraryBuilder.h"

#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/IRBuilder.h"

// =============================================================================
//
// NOTE: these structures model 1:1 those in the Hexagon executable library
// ABI, hexagon/dsp/rt/executable_library.h
//
// Only the latest version is emitted; the runtime rejects any other.
//
// =============================================================================

namespace mlir::iree_compiler::hexagon::target {

//===----------------------------------------------------------------------===//
// hexagon/dsp/rt/executable_library.h structure types
//===----------------------------------------------------------------------===//
// %struct.hexagon_executable_import_table_t = type {
//   i32,
//   i8**
// }
static llvm::StructType *makeImportTableType(llvm::LLVMContext &context) {
  if (auto *existingType = llvm::StructType::getTypeByName(
          context, "hexagon_executable_import_table_t")) {
    return existingType;
  }
  auto *i32Type = llvm::IntegerType::getInt32Ty(context);
  auto *type = llvm::StructType::create(context,
                                        {
                                            i32Type,
                                            llvm::PointerType::get(context, 0),
                                        },
                                        "hexagon_executable_import_table_t",
                                        /*isPacked=*/false);
  return type;
}

// %struct.hexagon_executable_dispatch_attrs_t = type {
//   i64,     // hexagon_flags
//   i64,     // vtcm_bytes
//   i64,     // threads
// }
// Must match hexagon/dsp/rt/executable_library.h;
// HexagonExecutableSerialization checks the offsets at build time.
static llvm::StructType *makeDispatchAttrsType(llvm::LLVMContext &context) {
  if (auto *existingType = llvm::StructType::getTypeByName(
          context, "hexagon_executable_dispatch_attrs_t")) {
    return existingType;
  }
  auto *i64Type = llvm::IntegerType::getInt64Ty(context);
  auto *type = llvm::StructType::create(context,
                                        {
                                            i64Type, // hexagon_flags
                                            i64Type, // vtcm_bytes
                                            i64Type, // threads
                                        },
                                        "hexagon_executable_dispatch_attrs_t",
                                        /*isPacked=*/false);
  return type;
}

// %struct.hexagon_executable_export_table_t = type {
//   i32,
//   ptr,  // ptrs
//   ptr,  // attrs
//   ptr,  // names
// }
static llvm::StructType *makeExportTableType(llvm::LLVMContext &context) {
  if (auto *existingType = llvm::StructType::getTypeByName(
          context, "hexagon_executable_export_table_t")) {
    return existingType;
  }
  auto *i32Type = llvm::IntegerType::getInt32Ty(context);
  auto *ptrType = llvm::PointerType::get(context, 0);
  auto *type = llvm::StructType::create(context,
                                        {
                                            i32Type, // count
                                            ptrType, // ptrs
                                            ptrType, // attrs
                                            ptrType, // names
                                        },
                                        "hexagon_executable_export_table_t",
                                        /*isPacked=*/false);
  return type;
}

// %struct.hexagon_executable_library_header_t = type {
//   i32
// }
static llvm::StructType *makeLibraryHeaderType(llvm::LLVMContext &context) {
  if (auto *existingType = llvm::StructType::getTypeByName(
          context, "hexagon_executable_library_header_t")) {
    return existingType;
  }
  auto *i32Type = llvm::IntegerType::getInt32Ty(context);
  auto *type = llvm::StructType::create(context,
                                        {
                                            i32Type, // version
                                        },
                                        "hexagon_executable_library_header_t",
                                        /*isPacked=*/false);
  return type;
}

// %struct.hexagon_executable_library_t = type {
//   %struct.hexagon_executable_library_header_t*,
//   %struct.hexagon_executable_import_table_t,
//   %struct.hexagon_executable_export_table_t,
// }
static llvm::StructType *makeLibraryType(llvm::StructType *libraryHeaderType) {
  auto &context = libraryHeaderType->getContext();
  if (auto *existingType = llvm::StructType::getTypeByName(
          context, "hexagon_executable_library_t")) {
    return existingType;
  }
  auto *importTableType = makeImportTableType(context);
  auto *exportTableType = makeExportTableType(context);
  auto *type = llvm::StructType::create(context,
                                        {
                                            llvm::PointerType::get(context, 0),
                                            importTableType,
                                            exportTableType,
                                        },
                                        "hexagon_executable_library_t",
                                        /*isPacked=*/false);
  return type;
}

//===----------------------------------------------------------------------===//
// IR construction utilities
//===----------------------------------------------------------------------===//

// Creates a global NUL-terminated string constant.
//
// Example:
//   @.str.2 = private unnamed_addr constant [6 x i8] c"lib_a\00", align 1
static llvm::Constant *createStringConstant(StringRef value,
                                            llvm::Module *module) {
  auto i8Type = llvm::IntegerType::getInt8Ty(module->getContext());
  auto i32Type = llvm::IntegerType::getInt32Ty(module->getContext());
  auto *stringType = llvm::ArrayType::get(i8Type, value.size() + /*NUL*/ 1);
  auto *literal =
      llvm::ConstantDataArray::getString(module->getContext(), value);
  auto *global = new llvm::GlobalVariable(*module, stringType,
                                          /*isConstant=*/true,
                                          llvm::GlobalVariable::PrivateLinkage,
                                          literal, /*Name=*/"");
  global->setAlignment(llvm::MaybeAlign(1));
  llvm::Constant *zero = llvm::ConstantInt::get(i32Type, 0);
  return llvm::ConstantExpr::getInBoundsGetElementPtr(
      stringType, global, ArrayRef<llvm::Constant *>{zero, zero});
}

// Creates a global constant with the given elements.
static llvm::Constant *createArrayConstant(StringRef name,
                                           llvm::Type *elementType,
                                           ArrayRef<llvm::Constant *> elements,
                                           llvm::Module *module) {
  auto *i32Type = llvm::IntegerType::getInt32Ty(module->getContext());
  llvm::Constant *zero = llvm::ConstantInt::get(i32Type, 0);
  auto *arrayType = llvm::ArrayType::get(elementType, elements.size());
  auto *global = new llvm::GlobalVariable(
      *module, arrayType, /*isConstant=*/true,
      llvm::GlobalVariable::PrivateLinkage,
      llvm::ConstantArray::get(arrayType, elements), name);
  return llvm::ConstantExpr::getInBoundsGetElementPtr(
      arrayType, global, ArrayRef<llvm::Constant *>{zero, zero});
}

//===----------------------------------------------------------------------===//
// Builder interface
//===----------------------------------------------------------------------===//

llvm::Function *LibraryBuilder::build(StringRef queryFuncName) {
  auto &context = module->getContext();
  auto *i32Type = llvm::IntegerType::getInt32Ty(context);
  auto *ptrType = llvm::PointerType::get(context, 0);

  // %struct.hexagon_executable_library_header_t**
  // @iree_hal_library_query(i32, %struct.hexagon_executable_environment_t*)
  auto *queryFuncType = llvm::FunctionType::get(ptrType,
                                                {
                                                    i32Type,
                                                    ptrType,
                                                },
                                                /*isVarArg=*/false);
  auto *func =
      llvm::Function::Create(queryFuncType, llvm::GlobalValue::InternalLinkage,
                             queryFuncName, *module);

  auto *entryBlock = llvm::BasicBlock::Create(context, "entry", func);
  llvm::IRBuilder<> builder(entryBlock);

  // Only one version is emitted:
  //   return max_version == version ? &library : NULL;
  auto *v0 = buildLibraryV0((queryFuncName + "_v0").str());
  builder.CreateRet(builder.CreateSelect(
      builder.CreateICmpEQ(
          func->getArg(0),
          llvm::ConstantInt::get(i32Type, static_cast<int64_t>(version))),
      builder.CreatePointerCast(v0, ptrType),
      llvm::ConstantPointerNull::get(ptrType)));

  return func;
}

llvm::Constant *
LibraryBuilder::buildLibraryV0ImportTable(std::string libraryName) {
  auto &context = module->getContext();
  auto *importTableType = makeImportTableType(context);
  auto *i32Type = llvm::IntegerType::getInt32Ty(context);
  llvm::Constant *symbolNames =
      llvm::Constant::getNullValue(llvm::PointerType::get(context, 0));
  if (!imports.empty()) {
    SmallVector<llvm::Constant *> symbolNameValues;
    for (auto &import : imports) {
      auto symbolName = import.symbol_name;
      if (import.weak) {
        symbolName = "?" + symbolName;
      }
      symbolNameValues.push_back(createStringConstant(symbolName, module));
    }
    symbolNames = createArrayConstant(libraryName + "_import_names",
                                      llvm::PointerType::get(context, 0),
                                      symbolNameValues, module);
  }
  return llvm::ConstantStruct::get(
      importTableType, {
                           // count=
                           llvm::ConstantInt::get(i32Type, imports.size()),
                           // symbols=
                           symbolNames,
                       });
}

llvm::Constant *
LibraryBuilder::buildLibraryV0ExportTable(std::string libraryName) {
  auto &context = module->getContext();
  auto *exportTableType = makeExportTableType(context);
  auto *dispatchAttrsType = makeDispatchAttrsType(context);
  auto *i32Type = llvm::IntegerType::getInt32Ty(context);
  auto *i64Type = llvm::IntegerType::getInt64Ty(context);
  auto *ptrType = llvm::PointerType::get(context, 0);

  // hexagon_executable_export_table_t::ptrs
  SmallVector<llvm::Constant *> exportPtrValues;
  for (const Dispatch &dispatch : exports) {
    exportPtrValues.push_back(dispatch.func);
  }
  llvm::Constant *exportPtrs = createArrayConstant(
      libraryName + "_funcs", ptrType, exportPtrValues, module);

  // hexagon_executable_export_table_t::attrs
  // Always populate the attrs table as it contains required dispatch metadata.
  SmallVector<llvm::Constant *> exportAttrValues;
  for (const Dispatch &dispatch : exports) {
    exportAttrValues.push_back(llvm::ConstantStruct::get(
        dispatchAttrsType,
        {
            // hexagon_flags=
            llvm::ConstantInt::get(i64Type, dispatch.attrs.hexagonFlags),
            // vtcm_bytes=
            llvm::ConstantInt::get(i64Type, dispatch.attrs.vtcmBytes),
            // threads=
            llvm::ConstantInt::get(i64Type, dispatch.attrs.threads),
        }));
  }
  llvm::Constant *exportAttrs = createArrayConstant(
      libraryName + "_attrs", dispatchAttrsType, exportAttrValues, module);

  // hexagon_executable_export_table_t::names
  llvm::Constant *exportNames = llvm::Constant::getNullValue(ptrType);
  if (!exports.empty()) {
    SmallVector<llvm::Constant *> exportNameValues;
    for (const Dispatch &dispatch : exports) {
      exportNameValues.push_back(createStringConstant(dispatch.name, module));
    }
    exportNames = createArrayConstant(libraryName + "_names", ptrType,
                                      exportNameValues, module);
  }

  return llvm::ConstantStruct::get(
      exportTableType, {
                           // count=
                           llvm::ConstantInt::get(i32Type, exports.size()),
                           // ptrs=
                           exportPtrs,
                           // attrs=
                           exportAttrs,
                           // names=
                           exportNames,
                       });
}

llvm::Constant *LibraryBuilder::buildLibraryV0(std::string libraryName) {
  auto &context = module->getContext();
  auto *libraryHeaderType = makeLibraryHeaderType(context);
  auto *libraryType = makeLibraryType(libraryHeaderType);
  auto *i32Type = llvm::IntegerType::getInt32Ty(context);

  // ----- Header -----

  auto *libraryHeader = new llvm::GlobalVariable(
      *module, libraryHeaderType, /*isConstant=*/true,
      llvm::GlobalVariable::PrivateLinkage,
      llvm::ConstantStruct::get(
          libraryHeaderType,
          {
              // version=
              llvm::ConstantInt::get(i32Type, static_cast<int64_t>(version)),
          }),
      /*Name=*/libraryName + "_header");
  // TODO(benvanik): force alignment (8? natural pointer width?)

  // ----- Library -----

  auto *library = new llvm::GlobalVariable(
      *module, libraryType, /*isConstant=*/true,
      llvm::GlobalVariable::PrivateLinkage,
      llvm::ConstantStruct::get(libraryType,
                                {
                                    // header=
                                    libraryHeader,
                                    // imports=
                                    buildLibraryV0ImportTable(libraryName),
                                    // exports=
                                    buildLibraryV0ExportTable(libraryName),
                                }),
      /*Name=*/libraryName);
  // TODO(benvanik): force alignment (8? natural pointer width?)

  return library;
}

} // namespace mlir::iree_compiler::hexagon::target

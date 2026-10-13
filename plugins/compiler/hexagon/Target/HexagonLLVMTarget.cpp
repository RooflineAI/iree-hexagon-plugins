// Copyright 2025 RooflineAI GmbH
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "hexagon/Target/HexagonLLVMTarget.h"

#include "hexagon/CodeGen/TargetConfig.h"
#include "iree/compiler/Codegen/Utils/Utils.h"
#include "mlir/IR/Builders.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/TargetParser/Triple.h"

#include <mutex>

namespace mlir::iree_compiler::hexagon::target {

using codegen::kMaxStackAllocationSizeAttrName;
using codegen::kNativeVectorSizeAttrName;

static constexpr char kCpuAttrName[] = "cpu";

// Registers all LLVM components required for Hexagon code generation.
void initializeHexagonTarget() {
  static std::once_flag init;
  std::call_once(init, []() {
    LLVMInitializeHexagonTargetInfo();
    LLVMInitializeHexagonTarget();
    LLVMInitializeHexagonTargetMC();
    LLVMInitializeHexagonAsmPrinter();
    // These two symbols are not currently needed. If they became necessary
    // though, they are not provided by the compiler object this plugin is being
    // dynamically loaded into, so they would need to be added to the plugin
    // itself.
    // LLVMInitializeHexagonAsmParser();
    // LLVMInitializeHexagonDisassembler();
  });
}

void HexagonTarget::storeToConfigAttrs(
    mlir::MLIRContext *context,
    llvm::SmallVectorImpl<mlir::NamedAttribute> &config) const {
  mlir::Builder b(context);
  addConfigTargetTriple(context, triple, config);
  config.emplace_back(b.getStringAttr(kCpuAttrName), b.getStringAttr(dsp));
  addConfigCpuFeatures(context, dspFeatures, config);
  addConfigDataLayout(context, dataLayout, config);
  config.emplace_back(b.getStringAttr(kNativeVectorSizeAttrName),
                      b.getI64IntegerAttr(vectorWidthInBytes));
  config.emplace_back(b.getStringAttr(kMaxStackAllocationSizeAttrName),
                      b.getI64IntegerAttr(maxStackAllocSizeInBytes));
}

HexagonTarget createHexagonTarget(const HexagonOptions &options) {
  HexagonTarget target;
  target.triple = "hexagon-unknown-unknown-elf";
  target.dsp = "hexagonv" + options.version;
  target.dspFeatures = options.features;
  target.dataLayout =
      "e-m:e-p:32:32:32-a:0-n16:32-i64:64:64-i32:32:32-i16:16:16-i1:8:8-f32:"
      "32:32-f64:64:64-v32:32:32-v64:64:64-v512:512:512-v1024:1024:1024-"
      "v2048:2048:2048";
  // TODO: Setting the actual vector bitwidth and using IREE's kernel
  // dispatching will result in compilation errors.
  // target.vectorWidthInBytes = 128;
  target.vectorWidthInBytes = 32;
  target.maxStackAllocSizeInBytes = 16 * 1024;
  return target;
}

std::unique_ptr<llvm::TargetMachine>
createHexagonTargetMachine(mlir::DictionaryAttr config) {
  std::optional<llvm::StringRef> tripleStr = getConfigTargetTriple(config);
  if (!tripleStr) {
    return nullptr;
  }
  auto dspAttr = config.getAs<mlir::StringAttr>(kCpuAttrName);
  llvm::StringRef dsp = dspAttr ? dspAttr.getValue() : "";
  llvm::StringRef dspFeatures = getConfigCpuFeatures(config).value_or("");

  llvm::Triple triple(*tripleStr);
  std::string errorMessage;
  const llvm::Target *llvmTarget =
      llvm::TargetRegistry::lookupTarget(triple, errorMessage);
  if (!llvmTarget) {
    return nullptr;
  }

  llvm::TargetOptions targetOptions;
  targetOptions.FloatABIType = llvm::FloatABI::Hard;
  // Place every function and global in its own section.
  targetOptions.FunctionSections = true;
  targetOptions.DataSections = true;
  targetOptions.UniqueSectionNames = true;

  return std::unique_ptr<llvm::TargetMachine>(llvmTarget->createTargetMachine(
      triple, dsp, dspFeatures, targetOptions, llvm::Reloc::Model::PIC_,
      /*CM=*/std::nullopt, llvm::CodeGenOptLevel::Aggressive, /*JIT=*/false));
}

} // namespace mlir::iree_compiler::hexagon::target

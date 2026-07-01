// Copyright 2025 RooflineAI GmbH
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "cellar-hexagon/Target/HexagonLLVMTarget.h"

#include "llvm/Support/TargetSelect.h"

#include <mutex>
#include <string>

namespace mlir::iree_compiler::cellar_hexagon::target {
namespace HAL = mlir::iree_compiler::IREE::HAL;

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

// This function creates a LLVM target for Hexagon. This is different
// from LLVMCPU since this logic is usually managed by multiple
// classes that do necessary adjustments depending on host machine or
// cross-compilation options. The cleanest way of implementing this would be to
// extend the LLVMCPUTarget functions managing this to support Hexagon, but this
// wraps around some of those calls instead to avoid modifying code outside this
// plugin
HAL::LLVMTarget createLLVMTargetForHexagon(const HexagonOptions &options) {
  constexpr llvm::StringRef triple = "hexagon-unknown-unknown-elf";
  // I found this value in the adsprpc logs. I am not actually totally sure
  // about it.
  constexpr int64_t kHexagonMaxStackAllocSizeInBytes = 16 * 1024;
  std::string cpuName = std::string("hexagonv") + options.version;
  HAL::ResolveCPUAndCPUFeaturesStatus status;

  // FIXME: This calls resolveCPUAndCPUFeatures that will fail because hexagon
  // is not registered as an LLVMTarget in IREE. Since I am just prototyping, I
  // will ignore the failed status and manually input the necessary info
  // (hardcoded) in the target (dataLayout and vectorWidth). The correct way
  // of doing this would be to update IREE I guess.
  auto targetOption =
      HAL::LLVMTarget::create(triple, cpuName, options.features, false, status);
  if (!targetOption)
    llvm::errs() << "Failed to define default LLVMTarget for Hexagon";
  auto target = targetOption.value();

  target.dataLayout =
      "e-m:e-p:32:32:32-a:0-n16:32-i64:64:64-i32:32:32-i16:16:16-i1:8:8-f32:"
      "32:32-f64:64:64-v32:32:32-v64:64:64-v512:512:512-v1024:1024:1024-"
      "v2048:2048:2048";

  // TODO: Setting the actual vector bitwidth and using IREE's kernel
  // dispatching will result in compilation errors This is also currently reused
  // by the new hexagon kernelDispatch, but that code is still quite unreliable
  // and needs fixing and testing
  // target.vectorWidthInBytes = 128;
  target.vectorWidthInBytes = 32;
  // Match the current DSP worker thread stack budget reported by adsprpc.
  target.maxStackAllocSizeInBytes = kHexagonMaxStackAllocSizeInBytes;

  return target;
}

} // namespace mlir::iree_compiler::cellar_hexagon::target

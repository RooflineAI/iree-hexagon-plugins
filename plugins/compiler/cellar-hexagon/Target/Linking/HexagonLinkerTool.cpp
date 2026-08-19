// Copyright 2025 RooflineAI GmbH
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "cellar-hexagon/Target/Linking/HexagonLinkerTool.h"

#include "iree/compiler/Utils/ToolUtils.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/IR/Function.h"

#include <string>
#include <variant>

#define DEBUG_TYPE "llvm-linker"

namespace mlir::iree_compiler::cellar_hexagon::target::linking {
namespace HAL = mlir::iree_compiler::IREE::HAL;

class HexagonLinkerTool : public HAL::LinkerTool {
public:
  HexagonLinkerTool(const llvm::Triple &targetTriple,
                    HAL::LLVMTargetOptions &targetOptions,
                    bool allowNativeUndefinedSymbols)
      : HAL::LinkerTool(targetTriple, targetOptions),
        allowNativeUndefinedSymbols(allowNativeUndefinedSymbols) {}

  std::string getLinkerToolPath() const {
    // Try to use the tool specified for this configuration from the serializer
    // first. This structure is meant for the LLVMCPUTarget, so I will be
    // reusing the embeddedLinkerPath from it, even though it is not meant for
    // this
    if (!targetOptions.embeddedLinkerPath.empty()) {
      return targetOptions.embeddedLinkerPath;
    }

    // Fall back to check for setting the linker explicitly via environment
    // variables
    char *envVarPath = std::getenv("IREE_HEXAGON_LINKER_PATH");
    if (envVarPath && envVarPath[0] != '\0')
      return std::string(envVarPath);

    // No explicit linker specified, search the install/build dir or env.
    // Note that hexagon clang is not really a linker and depends on the hexagon
    // sdk. Calling it adds default flags (e.g. -call_shared, --hash-style) to
    // the hexagon proprietary linker that has its own defaults. I tried
    // mimicking those defaults for the lld configuration.
    // These are checked in this order, so hexagon-clang has priority over lld
    const SmallVector<std::string> toolNames{"hexagon-clang", "lld", "ld.lld"};
    std::string toolPath = findTool(toolNames);
    if (!toolPath.empty())
      return toolPath;

    llvm::errs() << "error: required hexagon linker tool (supported: "
                    "`hexagon-clang`, available in hexagon SDK or "
                    "'lld'/'ld.lld) not found "
                    "after searching:\n"
                    "  * --iree-hexagon-linker-path= flag\n"
                    "  * IREE_HEXAGON_LINKER_PATH environment variable\n"
                    "  * system PATH\n"
                    "Run with --debug-only=llvm-linker for search details\n";
    return "";
  }

  std::optional<HAL::Artifacts>
  linkDynamicLibrary(llvm::StringRef libraryName,
                     llvm::ArrayRef<HAL::Artifact> objectFiles) override {
    HAL::Artifacts artifacts;

    artifacts.libraryFile = HAL::Artifact::createTemporary(libraryName, "so");
    artifacts.libraryFile.close();

    std::string toolPath = getLinkerToolPath();
    if (toolPath.empty())
      return std::nullopt;

    SmallVector<std::string, 32> flags;
    // When using hexagon-clang instead of a linker, most flags are inserted by
    // the sdk
    if (toolPath.find("hexagon-clang") != std::variant_npos) {
      flags.push_back(toolPath);
      flags.push_back("-shared");
      if (allowNativeUndefinedSymbols) {
        flags.push_back("-Wl,--unresolved-symbols=ignore-all");
      }
    } else {
      // Other linkers attempt not to use any dependencies on the sdk instead
      flags.push_back(toolPath);

      // MATCH -> Loading the elf still works without this flag, but this
      // matches the hexagon linker more closely.
      //
      // REQUIRED -> Loading the elf without this flag ends up in failure
      //
      // HEX-CLANG -> Flag normally added by hexagon clang itself when linking

      // MATCH Use lld's GNU driver; the SDK linker used the same syntax.
      flags.push_back("-flavor");
      flags.push_back("gnu");

      // MATCH Hexagon-link emitted a SysV .hash; matching hexagon-clang call to
      // the linker.
      flags.push_back("--hash-style=sysv");

      // REQUIRED Produce a shared object
      flags.push_back("-shared");

      // MATCH Link against dynamic library. The SDK binary set the shared-call
      // ABI bits; without the ELFs differed.
      flags.push_back("-call_shared");

      // MATCH HEX-CLANG Optimization level (none)
      flags.push_back("-G0");

      // MATCH Force 4 KiB p_align; LLD defaulted to 64 KiB which added extra
      // LOAD headers.
      flags.push_back("-z");
      flags.push_back("max-page-size=0x1000");

      // MATCH Keep common page size at 4 KiB so the RX/RW segments start at
      // 0x0/0x1000 like hexagon-link.
      flags.push_back("-z");
      flags.push_back("common-page-size=0x1000");

      // MATCH The hexagon-link ELF lacked PT_GNU_RELRO; disabling RELRO removes
      // it
      flags.push_back("-z");
      flags.push_back("norelro");

      // REQUIRED Without this lld split text into RO/RX loads; removing it
      // keeps it at two segments.
      flags.push_back("--no-rosegment");

      // MATCH Hexagon-link’s PT_GNU_STACK was RWX; mirroring it
      flags.push_back("-z");
      flags.push_back("execstack");

      // REQUIRED Forcing page-separated LOADs fixed the adsprpc “Cannot protect
      // segment 1” error.
      flags.push_back("-z");
      flags.push_back("separate-loadable-segments");

      if (allowNativeUndefinedSymbols) {
        // Native DSP runtime linking expects selected runtime symbols to remain
        // unresolved in the kernel .so and be bound by the DSP loader against
        // libhexagon_dsp_skel.so at load time.
        flags.push_back("--unresolved-symbols=ignore-all");
      }
    }

    flags.push_back("-o");
    flags.push_back(artifacts.libraryFile.path);

    // Link all input objects. Note that we are not linking whole-archive as
    // we want to allow dropping of unused codegen outputs (copied from cpu
    // linking).
    for (auto &objectFile : objectFiles)
      flags.push_back(objectFile.path);

    if (failed(runLinkCommand(llvm::join(flags, " ")))) {
      // Ensure we save inputs if we fail so that the user can replicate the
      // command themselves.
      if (targetOptions.keepLinkerArtifacts) {
        for (auto &objectFile : objectFiles) {
          if (objectFile.outputFile) {
            llvm::errs() << "linker input preserved: "
                         << objectFile.outputFile->getFilename();
            objectFile.keep();
          }
        }
      }
      return std::nullopt;
    }
    return artifacts;
  }

private:
  bool allowNativeUndefinedSymbols;
};

std::unique_ptr<HAL::LinkerTool>
createHexagonLinkerTool(const llvm::Triple &targetTriple,
                        HAL::LLVMTargetOptions &targetOptions,
                        bool allowNativeUndefinedSymbols) {
  return std::make_unique<HexagonLinkerTool>(targetTriple, targetOptions,
                                             allowNativeUndefinedSymbols);
}

} // namespace mlir::iree_compiler::cellar_hexagon::target::linking

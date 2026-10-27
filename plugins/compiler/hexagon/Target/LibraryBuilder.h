// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Derived from:
//   compiler/plugins/target/LLVMCPU/LibraryBuilder.{h,cpp}
// IREE revision: a45adeaa6115e446c898e6eb21fb6edc0e65ddc4
//
// Emits the Hexagon executable library ABI defined by
// hexagon/dsp/rt/executable_library.h. static_asserts in
// HexagonExecutableSerialization.cpp fail the build when the version or the
// dispatch attrs layout diverge from the header. See Target/README.md.

#ifndef IREE_HEXAGON_TARGET_LIBRARYBUILDER_H_
#define IREE_HEXAGON_TARGET_LIBRARYBUILDER_H_

#include <string>
#include <vector>

#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/IR/Module.h"
#include "llvm/TargetParser/Triple.h"

namespace mlir::iree_compiler::hexagon::target {

// Defines an `hexagon_executable_library_t` and builds the runtime metadata
// structures and query functions.
//
// See hexagon/dsp/rt/executable_library.h for more information.
//
// Usage:
//  LibraryBuilder builder(&module);
//  builder.addExport("hello", DispatchAttrs{}, &helloFunc);
//  ...
//  auto *queryFunc = builder.build("_query_library_foo");
//  // call queryFunc, export it, etc
class LibraryBuilder {
public:
  // hexagon_executable_library_version_t
  // Tracks HEXAGON_EXECUTABLE_LIBRARY_VERSION_LATEST.
  enum class Version : uint32_t {
    // 'HX' in the upper half keeps it apart from upstream IREE versions.
    HEXAGON_1 = 0x4858'0001u,

    // Pinned to the latest version.
    // Requires that the runtime be compiled with the same version.
    LATEST = HEXAGON_1,
  };

  // hexagon_dispatch_flags_t
  enum class HexagonDispatchFlags : uint64_t {
    // HEXAGON_DISPATCH_FLAG_NONE
    NONE = 0ull,
    // HEXAGON_DISPATCH_FLAG_RESOURCES_VALID
    RESOURCES_VALID = 1ull << 0,
    // HEXAGON_DISPATCH_FLAG_USES_HVX
    USES_HVX = 1ull << 1,
    // HEXAGON_DISPATCH_FLAG_USES_HMX
    USES_HMX = 1ull << 2,
  };

  // hexagon_executable_dispatch_attrs_t
  struct DispatchAttrs {
    // Resources the planner recorded for the dispatch.
    // Bitwise OR of HexagonDispatchFlags values.
    uint64_t hexagonFlags = 0;
    // Planned peak VTCM bytes. Only meaningful if hexagonFlags has
    // RESOURCES_VALID.
    uint64_t vtcmBytes = 0;
    // Maximum number of tasks in flight on worker threads, 1 if none. Only
    // meaningful if hexagonFlags has RESOURCES_VALID.
    uint64_t threads = 1;
  };

  LibraryBuilder(llvm::Module *module, Version version = Version::LATEST)
      : module(module), version(version) {}

  // Defines a new runtime import function.
  // The declared ordinal of the import matches the order they are declared.
  // FIXME: the DSP runtime rejects libraries that declare imports; see
  // hexagon_executable_import_table_t.
  void addImport(StringRef name, bool weak) {
    imports.push_back({name.str(), weak});
  }

  // Defines a new entry point on the library implemented by |func|.
  // |name| will be used as the library export.
  void addExport(StringRef name, DispatchAttrs attrs, llvm::Function *func) {
    exports.push_back({name.str(), attrs, func});
  }

  // Builds a `hexagon_executable_library_query_fn_t` with the given
  // |queryFuncName| that will return the current library metadata.
  //
  // The returned function will be inserted into the module with internal
  // linkage. Callers may change the linkage based on their needs (such as
  // exporting if producing a dynamic library or making dso_local for static
  // libraries that reference the function via extern in another compilation
  // unit, etc).
  llvm::Function *build(StringRef queryFuncName);

private:
  // Builds and returns an hexagon_executable_library_t global constant.
  llvm::Constant *buildLibraryV0(std::string libraryName);
  llvm::Constant *buildLibraryV0ImportTable(std::string libraryName);
  llvm::Constant *buildLibraryV0ExportTable(std::string libraryName);

  llvm::Module *module = nullptr;
  Version version = Version::LATEST;

  struct Import {
    std::string symbol_name;
    bool weak = false;
  };
  SmallVector<Import> imports;

  struct Dispatch {
    std::string name;
    DispatchAttrs attrs;
    llvm::Function *func;
  };
  std::vector<Dispatch> exports;
};

} // namespace mlir::iree_compiler::hexagon::target

#endif // IREE_HEXAGON_TARGET_LIBRARYBUILDER_H_

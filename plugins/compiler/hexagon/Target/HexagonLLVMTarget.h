// Copyright 2025 RooflineAI GmbH
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef ROOF_HEXAGON_HAL_HEXAGONLLVMTARGET_H_
#define ROOF_HEXAGON_HAL_HEXAGONLLVMTARGET_H_

#include "hexagon/Target/HexagonOptions.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "llvm/Target/TargetMachine.h"

#include <memory>
#include <string>

namespace mlir::iree_compiler::hexagon::target {

void initializeHexagonTarget();

// Describes the LLVM Hexagon target. It is stored in the executable target
// configuration, which is the only thing serialization reads back.
struct HexagonTarget {
  std::string triple;
  std::string dsp;
  std::string dspFeatures;
  std::string dataLayout;
  int64_t vectorWidthInBytes;
  int64_t maxStackAllocSizeInBytes;

  // Writes the configuration keys read by the codegen passes (several of them
  // reused from LLVMCPU) and by serialization.
  void
  storeToConfigAttrs(mlir::MLIRContext *context,
                     llvm::SmallVectorImpl<mlir::NamedAttribute> &config) const;
};

HexagonTarget createHexagonTarget(const HexagonOptions &options);

// Creates the LLVM target machine for the target described in `config`.
// Returns nullptr on failure.
std::unique_ptr<llvm::TargetMachine>
createHexagonTargetMachine(mlir::DictionaryAttr config);

} // namespace mlir::iree_compiler::hexagon::target

#endif // ROOF_HEXAGON_HAL_HEXAGONLLVMTARGET_H_

// Copyright 2025 RooflineAI GmbH
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef ROOF_HEXAGON_HAL_LINKING_HEXAGONLINKER_H_
#define ROOF_HEXAGON_HAL_LINKING_HEXAGONLINKER_H_

#include "mlir/Support/LLVM.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"

#include <string>

namespace mlir::iree_compiler::hexagon::target::linking {

// Links `objectPaths` into a Hexagon shared object written to `outputPath`.
//
// `linkerPath` is the --iree-hexagon-linker-path value. When empty, the
// IREE_HEXAGON_LINKER_PATH environment variable and then the install
// directory and PATH are searched for hexagon-clang, lld and ld.lld.
//
// `allowNativeUndefinedSymbols` leaves undefined symbols unresolved so the DSP
// loader can bind them against the runtime at load time.
LogicalResult linkHexagonSharedObject(StringRef linkerPath,
                                      ArrayRef<std::string> objectPaths,
                                      StringRef outputPath,
                                      bool allowNativeUndefinedSymbols);

} // namespace mlir::iree_compiler::hexagon::target::linking

#endif // ROOF_HEXAGON_HAL_LINKING_HEXAGONLINKER_H_

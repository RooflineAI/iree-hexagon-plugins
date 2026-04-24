// Copyright 2025 RooflineAI GmbH
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef ROOF_HEXAGON_HAL_HEXAGONOPTIONS_H_
#define ROOF_HEXAGON_HAL_HEXAGONOPTIONS_H_

#include <string>

#include "iree/compiler/Utils/OptionUtils.h"

namespace mlir::iree_compiler::cellar_hexagon::target {

// These options describe the available fields that are reused in the Hexagon
// TargetDevice and TargetBackend. They are available as CLI arguments obviously
struct HexagonOptions {
  std::string version = "79";
  std::string features = "";
  std::string linker = "";

  void bindOptions(mlir::iree_compiler::OptionsBinder &binder);
};

} // namespace mlir::iree_compiler::cellar_hexagon::target

#endif // ROOF_HEXAGON_HAL_HEXAGONOPTIONS_H_

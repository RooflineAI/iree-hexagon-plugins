// Copyright 2025 RooflineAI GmbH
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "cellar-hexagon/Target/HexagonSession.h"

extern "C" bool iree_register_compiler_plugin_cellar_hexagon(
    mlir::iree_compiler::PluginRegistrar *registrar) {
  return mlir::iree_compiler::cellar_hexagon::target::
      registerCellarHexagonPlugin(registrar);
}

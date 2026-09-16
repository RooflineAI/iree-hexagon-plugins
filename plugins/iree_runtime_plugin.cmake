# Copyright 2026 RooflineAI GmbH
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Discovered by IREE's CMake build via -DIREE_CMAKE_PLUGIN_PATHS=<this repo>/plugins.
# Registers the Hexagon HAL driver as an external driver, statically linked into
# iree-run-module/iree-benchmark-module when
# -DIREE_EXTERNAL_HAL_DRIVERS=hexagon is passed at configure time.
include(${CMAKE_CURRENT_LIST_DIR}/_iree_hexagon_plugins_root.cmake)

iree_register_external_hal_driver(
  NAME
    hexagon
  DRIVER_TARGET
    iree_hexagon_plugins::plugins::runtime::hexagon::registration::DriverModuleLib
  REGISTER_FN
    iree_hal_hexagon_driver_module_register
)

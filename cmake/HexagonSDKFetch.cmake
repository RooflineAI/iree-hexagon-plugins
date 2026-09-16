# Copyright 2026 RooflineAI GmbH
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Sets HEXAGON_SDK_ROOT from the environment, with no CMake target
# definitions -- split out from cmake/HexagonSDK.cmake (which defines
# IMPORTED targets on top of this) specifically so
# cmake/HexagonToolchain.cmake can include just this half.
#
if(DEFINED HEXAGON_SDK_ROOT)
  return()
endif()

if(NOT DEFINED ENV{HEXAGON_SDK_ROOT})
  message(FATAL_ERROR
    "HEXAGON_SDK_ROOT is not set. Run via "
    "build_tools/cmake/build_and_package.sh, which fetches the Hexagon SDK "
    "and exports HEXAGON_SDK_ROOT before invoking cmake.")
endif()
set(HEXAGON_SDK_ROOT "$ENV{HEXAGON_SDK_ROOT}")

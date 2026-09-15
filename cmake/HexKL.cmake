# Copyright 2026 RooflineAI GmbH
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Exposes the Hexagon Kernel Library (HexKL) archive this repo actually uses
# (v79) as an IMPORTED target. Only needed in the Hexagon-DSP cross-build
# tree (cmake/HexagonToolchain.cmake).
if(TARGET hexkl::hexkl_micro_v79)
  return()
endif()

if(NOT DEFINED ENV{HEXKL_ROOT})
  message(FATAL_ERROR
    "HEXKL_ROOT is not set. Run via build_tools/cmake/build_and_package.sh, "
    "which fetches HexKL and exports HEXKL_ROOT before invoking cmake.")
endif()
set(HEXKL_ROOT "$ENV{HEXKL_ROOT}")

add_library(hexkl::hexkl_micro_v79_archive STATIC IMPORTED)
set_target_properties(hexkl::hexkl_micro_v79_archive PROPERTIES
  IMPORTED_LOCATION "${HEXKL_ROOT}/hexkl_addon/lib/hexagon_toolv19_v79/libhexkl_micro.a"
)

add_library(hexkl::hexkl_micro_v79 INTERFACE IMPORTED)
set_target_properties(hexkl::hexkl_micro_v79 PROPERTIES
  INTERFACE_INCLUDE_DIRECTORIES "${HEXKL_ROOT}/hexkl_addon/include"
  INTERFACE_LINK_LIBRARIES hexkl::hexkl_micro_v79_archive
)

# Copyright 2026 RooflineAI GmbH
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Not a standalone CMake project: this repo is built as an add_subdirectory()
# of IREE's own CMake build (see plugins/_iree_hexagon_plugins_root.cmake,
# pulled in via IREE's -DIREE_CMAKE_PLUGIN_PATHS=<this repo>/plugins plugin
# discovery, itself pointed at from third-party/iree as the actual `cmake -S`
# source root.
#
# Establishes this repo's own CMake package-namespace root (every target's
# CMake name becomes "iree_hexagon_plugins::<path-from-here>" before descending
# into it
iree_setup_c_src_root(PACKAGE_ROOT_PREFIX "iree_hexagon_plugins")

# Pass -DIREE_HEXAGON_ANDROID_BUILD=ON for the dedicated Android cross-build
# tree
option(IREE_HEXAGON_ANDROID_BUILD
  "Whether this is the Android/ARM cross-build tree"
  OFF
)

# CMAKE_SYSTEM_NAME Generic (set in cmake/HexagonToolchain.cmake) makes
# project() load CMake's Platform/Generic.cmake, which assumes bare-metal
# targets can't do dynamic linking and sets this property to FALSE. QuRT
# (Hexagon's RTOS) does support shared libraries -- it's how hexagon_dsp_skel
# itself gets loaded -- so restore it here, after project() has already run.
if(IREE_HEXAGON_MLIR_DSP_BUILD)
  set_property(GLOBAL PROPERTY TARGET_SUPPORTS_SHARED_LIBS TRUE)
endif()

include("${CMAKE_CURRENT_LIST_DIR}/cmake/HexagonSDK.cmake")
# Only the DSP cross-build tree (cmake/HexagonToolchain.cmake, which sets
# this cache var) needs HexKL (hexagon-mlir overlay's bin/runtime).
if(IREE_HEXAGON_MLIR_DSP_BUILD)
  include("${CMAKE_CURRENT_LIST_DIR}/cmake/HexKL.cmake")
endif()

# The hexagon_mlir overlay's dialect/codegen targets need mlir-tblgen.
# It is only built for IREE_BUILD_COMPILER=ON.
if(IREE_BUILD_COMPILER OR IREE_HEXAGON_MLIR_DSP_BUILD)
  add_subdirectory(build_tools/bazel/overlays/hexagon_mlir)
endif()
add_subdirectory(plugins)

if(IREE_HEXAGON_ANDROID_BUILD)
  add_subdirectory(integration_tests/device/tools)
endif()

# Copyright 2026 RooflineAI GmbH
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Hand-written: this directory is a pure namespace directory with no
# BUILD.bazel of its own (only plugins/compiler/hexagon/ has one), so there
# is nothing for bazel_to_cmake.py to autogenerate here -- but
# iree_add_all_subdirs() (used by plugins/CMakeLists.txt, one level up) only
# descends into subdirectories that already have a CMakeLists.txt, so this
# bridge is needed to reach plugins/compiler/hexagon/CMakeLists.txt at all.
#
# Add include dirs to support absolute "hexagon/..." includes.
include_directories(
  "${CMAKE_CURRENT_SOURCE_DIR}"
  "${CMAKE_CURRENT_BINARY_DIR}"
)
list(APPEND IREE_COMPILER_TABLEGEN_INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}")

add_subdirectory(hexagon)

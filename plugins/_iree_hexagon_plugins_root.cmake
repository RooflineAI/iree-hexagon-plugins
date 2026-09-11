# Copyright 2026 RooflineAI GmbH
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Shared by iree_compiler_plugin.cmake and iree_runtime_plugin.cmake.
# IREE's compiler-plugin and runtime-HAL-driver-plugin discovery mechanisms
# each include() one of those marker files independently.
# The repo root must be added as subdirectory exactly once, even if compiler
# and runtime plugins are both enabled.
get_property(_iree_hexagon_plugins_root_added GLOBAL PROPERTY IREE_HEXAGON_PLUGINS_ROOT_ADDED)
if(NOT _iree_hexagon_plugins_root_added)
  set_property(GLOBAL PROPERTY IREE_HEXAGON_PLUGINS_ROOT_ADDED TRUE)
  add_subdirectory("${CMAKE_CURRENT_LIST_DIR}/.." iree_hexagon_plugins)
endif()

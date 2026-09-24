# Copyright 2026 RooflineAI GmbH
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Fetches the free-to-download Hexagon SDK and exposes it as a set of
# IMPORTED CMake targets.
# Included from:
#   - cmake/HexagonToolchain.cmake (the Hexagon-DSP cross-build tree), where
#     HEXAGON_SDK_ROOT also locates the compiler itself.
#   - the host and Android configure trees, where only hexagon_sdk::qaic and
#     the ARM-side hexagon_sdk::cdsprpc_android_aarch64 shared-lib import are
#     needed.
#
# Included multiple times (once per configure tree) with guard,
# re-inclusion in the same configure is a no-op.
if(TARGET hexagon_sdk::qaic)
  return()
endif()

include("${CMAKE_CURRENT_LIST_DIR}/HexagonSDKFetch.cmake")

add_executable(hexagon_sdk::qaic IMPORTED)
set_target_properties(hexagon_sdk::qaic PROPERTIES
  IMPORTED_LOCATION "${HEXAGON_SDK_ROOT}/ipc/fastrpc/qaic/bin/qaic"
)

# Host simulator and v79 QuRT payloads used by CMake simulator tests. The
# payloads are represented as imported executable/library targets only to make
# their paths available through $<TARGET_FILE:...>; CMake never executes them
# directly on the build host.
add_executable(hexagon_sdk::hexagon_sim IMPORTED)
set_target_properties(hexagon_sdk::hexagon_sim PROPERTIES
  IMPORTED_LOCATION
    "${HEXAGON_SDK_ROOT}/tools/HEXAGON_Tools/19.0.04/Tools/bin/hexagon-sim"
)

add_executable(hexagon_sdk::run_main_on_hexagon_sim_v79 IMPORTED)
set_target_properties(hexagon_sdk::run_main_on_hexagon_sim_v79 PROPERTIES
  IMPORTED_LOCATION
    "${HEXAGON_SDK_ROOT}/libs/run_main_on_hexagon/ship/hexagon_toolv19_v79/run_main_on_hexagon_sim"
)

add_executable(hexagon_sdk::qurt_runelf_v79 IMPORTED)
set_target_properties(hexagon_sdk::qurt_runelf_v79 PROPERTIES
  IMPORTED_LOCATION
    "${HEXAGON_SDK_ROOT}/rtos/qurt/computev79/sdksim_bin/runelf.pbn"
)

add_library(hexagon_sdk::qurt_model_v79 SHARED IMPORTED)
set_target_properties(hexagon_sdk::qurt_model_v79 PROPERTIES
  IMPORTED_LOCATION
    "${HEXAGON_SDK_ROOT}/rtos/qurt/computev79/debugger/lnx64/qurt_model.so"
)

add_library(hexagon_sdk::cdsprpc_android_aarch64 SHARED IMPORTED)
set_target_properties(hexagon_sdk::cdsprpc_android_aarch64 PROPERTIES
  IMPORTED_LOCATION "${HEXAGON_SDK_ROOT}/ipc/fastrpc/remote/ship/android_aarch64/libcdsprpc.so"
  INTERFACE_INCLUDE_DIRECTORIES
    "${HEXAGON_SDK_ROOT}/incs;${HEXAGON_SDK_ROOT}/incs/stddef;${HEXAGON_SDK_ROOT}/ipc/fastrpc/rpcmem/inc"
)

add_library(hexagon_sdk::qurt_headers INTERFACE IMPORTED)
set_target_properties(hexagon_sdk::qurt_headers PROPERTIES
  INTERFACE_INCLUDE_DIRECTORIES
    "${HEXAGON_SDK_ROOT}/rtos/qurt/computev79/include/qurt;${HEXAGON_SDK_ROOT}/rtos/qurt/computev79/include/posix"
)

add_library(hexagon_sdk::qhl_hvx_headers INTERFACE IMPORTED)
set_target_properties(hexagon_sdk::qhl_hvx_headers PROPERTIES
  INTERFACE_INCLUDE_DIRECTORIES
    "${HEXAGON_SDK_ROOT}/libs/qfe/inc;${HEXAGON_SDK_ROOT}/libs/qhl_hvx/inc"
)

add_library(hexagon_sdk::hexagon_toolchain_bit_headers INTERFACE IMPORTED)
set_target_properties(hexagon_sdk::hexagon_toolchain_bit_headers PROPERTIES
  INTERFACE_INCLUDE_DIRECTORIES
    "${HEXAGON_SDK_ROOT}/tools/HEXAGON_Tools/19.0.04/Tools/target/hexagon/include/c++/v1/__bit"
)

add_library(hexagon_sdk::qurt_lib STATIC IMPORTED)
set_target_properties(hexagon_sdk::qurt_lib PROPERTIES
  IMPORTED_LOCATION "${HEXAGON_SDK_ROOT}/rtos/qurt/computev68/lib/libqurt.a"
)

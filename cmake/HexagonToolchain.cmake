# Copyright 2026 RooflineAI GmbH
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# CMAKE_TOOLCHAIN_FILE for cross-compiling to Hexagon-DSP.
#
# Usage:
#   cmake -G Ninja -S third-party/iree -B build/dsp \
#     -DCMAKE_TOOLCHAIN_FILE=<repo>/cmake/HexagonToolchain.cmake \
#     -DIREE_HEXAGON_MLIR_DSP_BUILD=ON \
#     -DIREE_HEXAGON_PLUGINS_ROOT=<repo> \
#     ...
#
# CMAKE_TOOLCHAIN_FILE is evaluated very early (before project()).
# No -D flags are evaluated, so we need to relay on an environment variable:
# IREE_HEXAGON_PLUGINS_ROOT.
if(NOT DEFINED ENV{IREE_HEXAGON_PLUGINS_ROOT})
  message(FATAL_ERROR
    "HexagonToolchain.cmake requires the IREE_HEXAGON_PLUGINS_ROOT "
    "environment variable to be set to this repo's root.")
endif()
set(IREE_HEXAGON_PLUGINS_ROOT "$ENV{IREE_HEXAGON_PLUGINS_ROOT}")

# Only the fetch (no target definitions -- add_library()/add_executable()
# are not reliably usable this early, before project()); the full
# cmake/HexagonSDK.cmake (defining hexagon_sdk::* targets) is included later,
# unconditionally, from this repo's root CMakeLists.txt.
include("${IREE_HEXAGON_PLUGINS_ROOT}/cmake/HexagonSDKFetch.cmake")

set(CMAKE_SYSTEM_NAME Generic)
set(CMAKE_SYSTEM_PROCESSOR hexagon)
# No runnable target on the build host; only ever build static/shared
# libraries or objects for this toolchain (see try_compile() docs).
set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)

set(_HEXAGON_TOOLS_BIN "${HEXAGON_SDK_ROOT}/tools/HEXAGON_Tools/19.0.04/Tools/bin")
set(CMAKE_C_COMPILER "${_HEXAGON_TOOLS_BIN}/hexagon-clang")
set(CMAKE_CXX_COMPILER "${_HEXAGON_TOOLS_BIN}/hexagon-clang++")
set(CMAKE_AR "${_HEXAGON_TOOLS_BIN}/hexagon-ar" CACHE FILEPATH "" FORCE)
set(CMAKE_RANLIB "${_HEXAGON_TOOLS_BIN}/hexagon-ranlib" CACHE FILEPATH "" FORCE)

set(_HEXAGON_MV "79")

set(_HEXAGON_SDK_RELATIVE_INCLUDE_DIRS
  incs
  incs/stddef
  ipc/fastrpc/incs
  ipc/fastrpc/rpcmem/inc
  incs/qnx
  libs/common/qnx/ship/hexagon_Debug_toolv88_v${_HEXAGON_MV}
  utils/examples
  ipc/fastrpc/rtld/ship/hexagon_toolv88_v${_HEXAGON_MV}
  ipc/fastrpc/remote/ship/hexagon_Debug_toolv88_v${_HEXAGON_MV}
  ipc/fastrpc/rtld/ship/inc
  libs/atomic/inc
  utils/sim_utils/inc
  libs/atomic/prebuilt/hexagon_toolv88_v${_HEXAGON_MV}
  utils/sim_utils/prebuilt/hexagon_toolv88_v${_HEXAGON_MV}
  rtos/qurt/computev${_HEXAGON_MV}/include/qurt
  rtos/qurt/computev${_HEXAGON_MV}/include/posix
)
set(_HEXAGON_SDK_ISYSTEM_FLAGS "")
foreach(_dir ${_HEXAGON_SDK_RELATIVE_INCLUDE_DIRS})
  string(APPEND _HEXAGON_SDK_ISYSTEM_FLAGS " -isystem ${HEXAGON_SDK_ROOT}/${_dir}")
endforeach()

# The Hexagon SDK toolchain has no <alloca.h> anywhere in its include tree
# (unlike glibc/bionic/musl); IREE's iree/base/allocator.h unconditionally
# includes it on non-Windows platforms. Shim it in ahead of everything else.
string(PREPEND _HEXAGON_SDK_ISYSTEM_FLAGS
  " -isystem ${IREE_HEXAGON_PLUGINS_ROOT}/build_tools/cmake/hexagon_libc_shims")

# Builtin include dirs (relative to the compiler's own directory, from
# `clang -x c(++) -E -v /dev/null`'s output against the SDK's bundled clang).
set(_HEXAGON_BUILTIN_C_ISYSTEM_FLAGS
  " -isystem ${_HEXAGON_TOOLS_BIN}/../target/hexagon/include"
  " -isystem ${_HEXAGON_TOOLS_BIN}/../lib/clang/19/include"
)
string(REPLACE ";" "" _HEXAGON_BUILTIN_C_ISYSTEM_FLAGS "${_HEXAGON_BUILTIN_C_ISYSTEM_FLAGS}")
set(_HEXAGON_BUILTIN_CXX_ISYSTEM_FLAGS
  " -isystem ${_HEXAGON_TOOLS_BIN}/../target/hexagon/include/c++/v1${_HEXAGON_BUILTIN_C_ISYSTEM_FLAGS}"
)

set(_HEXAGON_COMMON_FLAGS
  "-mv${_HEXAGON_MV} -fdata-sections -fstack-protector -fpic -D__V_DYNAMIC__ -mhvx -mhvx-length=128B -DIREE_TIME_NOW_FN=\"\{ return 0; \}\" -DIREE_CPUINFO_TARGET=\\\"\\\" -DIREE_TASK_CPUINFO_DISABLED=1 -DIREE_PLATFORM_GENERIC=1${_HEXAGON_SDK_ISYSTEM_FLAGS}"
)

set(CMAKE_C_FLAGS_INIT "${_HEXAGON_COMMON_FLAGS}${_HEXAGON_BUILTIN_C_ISYSTEM_FLAGS}")
set(CMAKE_CXX_FLAGS_INIT "${_HEXAGON_COMMON_FLAGS}${_HEXAGON_BUILTIN_CXX_ISYSTEM_FLAGS}")

# Shared-link flags common to the DSP skeleton and simulator test modules.
# Do not put a C++ runtime here: the C++ linker driver selects it for the DSP
# skeleton, while the C-only test modules must remain loadable by the SDK's
# self-contained run_main_on_hexagon_sim executable.
set(CMAKE_SHARED_LINKER_FLAGS_INIT
  "-mv${_HEXAGON_MV} -Wl,--defsym=ISDB_TRUSTED_FLAG=2 -Wl,--defsym=ISDB_SECURE_FLAG=2 -Wl,--no-threads -fpic -Wl,-Bsymbolic -Wl,--wrap=malloc -Wl,--wrap=calloc -Wl,--wrap=free -Wl,--wrap=realloc -Wl,--wrap=memalign"
)

# This is the DSP-only configure tree: guards plugins/runtime/hexagon's
# target_compatible_with on //constraints:{cpu_hexagon,os_qurt} (see
# .bazel_to_cmake.cfg.py) and the hexagon-mlir overlay's bin/runtime.
set(IREE_HEXAGON_MLIR_DSP_BUILD ON CACHE BOOL "" FORCE)

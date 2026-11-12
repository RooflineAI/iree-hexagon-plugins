# Copyright 2026 RooflineAI GmbH
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# tracy-capture and tracy-csvexport from IREE's vendored Tracy, for the Tracy
# runtime package, where they capture and export a profile on the device itself
# (see integration_tests/stages/running.py).
#
# The CMake mirror of build_tools/bazel/tracy/BUILD.tracy_tools.bazel, with the
# same sources and the same pinned capstone and ppqsort as MODULE.bazel. Tracy's
# own capture/ and csvexport/ CMake projects are not used: their vendor.cmake
# also fetches the GUI dependencies (glfw, freetype, ImGui) and queries
# pkg-config, which does not work in a cross build.

cmake_minimum_required(VERSION 3.24)

project(tracy_tools LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(TRACY_DIR "${CMAKE_CURRENT_LIST_DIR}/../../../third-party/iree/third_party/tracy")
cmake_path(NORMAL_PATH TRACY_DIR)

include(FetchContent)

# capstone 6.0.0-Alpha1
FetchContent_Declare(
  tracy_capstone
  URL https://github.com/capstone-engine/capstone/archive/97db712c91e964718f9cc300e81b9cf76b31a22e.tar.gz
  URL_HASH SHA256=4d62cf5f8a1578d2c70db1fa93533bc5ce58d207abe83ca654c36246dd4f266a
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)
# PPQSort v1.0.3
FetchContent_Declare(
  tracy_ppqsort
  URL https://github.com/GabTux/PPQSort/archive/4b964020d67b435dae7ebac7b8f5ecea1f421c58.tar.gz
  URL_HASH SHA256=c9d8c346f90e7dca3d38e2c8d053d5073231153cea92d964744f3f65a4e163c1
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)
# zstd v1.5.7, as fetched by Tracy's cmake/vendor.cmake
FetchContent_Declare(
  tracy_zstd
  URL https://github.com/facebook/zstd/releases/download/v1.5.7/zstd-1.5.7.tar.gz
  URL_HASH SHA256=eb33e51f49a15e023950cd7825ca74a4a2b43db8354825ac24fc1b7ee09e6fa3
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)
# Only the sources are needed: they are built below, as in the Bazel overlays.
FetchContent_GetProperties(tracy_capstone)
if(NOT tracy_capstone_POPULATED)
  FetchContent_Populate(tracy_capstone)
endif()
FetchContent_GetProperties(tracy_ppqsort)
if(NOT tracy_ppqsort_POPULATED)
  FetchContent_Populate(tracy_ppqsort)
endif()
FetchContent_GetProperties(tracy_zstd)
if(NOT tracy_zstd_POPULATED)
  FetchContent_Populate(tracy_zstd)
endif()

# Vendored code, warnings are not ours to fix.
add_compile_options(-w)

if(ANDROID)
  # ppqsort uses std::jthread, which the NDK's libc++ only provides as an
  # experimental feature.
  add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-fexperimental-library>)
  # The NDK toolchain always adds -g, which triples the size of the tools that
  # are pushed with every deployment. Like the Bazel opt build, go without.
  add_compile_options(-g0)
endif()

# capstone, reduced to the architectures the Tracy server disassembles
# (x86, ARM, AArch64), as in build_tools/bazel/tracy/BUILD.capstone.bazel.
set(_CS "${tracy_capstone_SOURCE_DIR}")
add_library(tracy_capstone STATIC
  ${_CS}/MCInst.c
  ${_CS}/MCInstPrinter.c
  ${_CS}/MCInstrDesc.c
  ${_CS}/MCRegisterInfo.c
  ${_CS}/Mapping.c
  ${_CS}/SStream.c
  ${_CS}/arch/AArch64/AArch64BaseInfo.c
  ${_CS}/arch/AArch64/AArch64Disassembler.c
  ${_CS}/arch/AArch64/AArch64DisassemblerExtension.c
  ${_CS}/arch/AArch64/AArch64InstPrinter.c
  ${_CS}/arch/AArch64/AArch64Mapping.c
  ${_CS}/arch/AArch64/AArch64Module.c
  ${_CS}/arch/ARM/ARMBaseInfo.c
  ${_CS}/arch/ARM/ARMDisassembler.c
  ${_CS}/arch/ARM/ARMDisassemblerExtension.c
  ${_CS}/arch/ARM/ARMInstPrinter.c
  ${_CS}/arch/ARM/ARMMapping.c
  ${_CS}/arch/ARM/ARMModule.c
  ${_CS}/arch/X86/X86Disassembler.c
  ${_CS}/arch/X86/X86DisassemblerDecoder.c
  ${_CS}/arch/X86/X86InstPrinterCommon.c
  ${_CS}/arch/X86/X86IntelInstPrinter.c
  ${_CS}/arch/X86/X86Mapping.c
  ${_CS}/arch/X86/X86Module.c
  ${_CS}/cs.c
  ${_CS}/utils.c
)
target_compile_definitions(tracy_capstone PRIVATE
  CAPSTONE_HAS_AARCH64
  CAPSTONE_HAS_ARM
  CAPSTONE_HAS_X86
  CAPSTONE_USE_SYS_DYN_MEM
  CAPSTONE_X86_ATT_DISABLE
)
target_include_directories(tracy_capstone PRIVATE "${_CS}")
# Tracy includes <capstone.h> directly.
target_include_directories(tracy_capstone PUBLIC
  "${_CS}/include"
  "${_CS}/include/capstone"
)

add_library(tracy_ppqsort INTERFACE)
target_include_directories(tracy_ppqsort INTERFACE "${tracy_ppqsort_SOURCE_DIR}/include")

set(_ZSTD "${tracy_zstd_SOURCE_DIR}/lib")
file(GLOB _ZSTD_SOURCES
  "${_ZSTD}/common/*.c"
  "${_ZSTD}/compress/*.c"
  "${_ZSTD}/decompress/*.c"
  "${_ZSTD}/dictBuilder/*.c"
)
# Only the C sources are built, so the x86-64 assembly decoder is disabled.
add_library(tracy_zstd STATIC ${_ZSTD_SOURCES})
target_compile_definitions(tracy_zstd PRIVATE ZSTD_DISABLE_ASM)
target_include_directories(tracy_zstd PUBLIC "${_ZSTD}")

# Tracy's cmake/GitRef.cmake fallback when git is unavailable. The tools still
# print the Tracy version next to it.
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/gitref/GitRef.hpp"
  "#pragma once\nnamespace tracy { static inline const char* GitRef = \"unknown\"; }\n")

add_library(tracy_getopt STATIC "${TRACY_DIR}/getopt/getopt.c")
target_include_directories(tracy_getopt PUBLIC "${TRACY_DIR}/getopt")

# The statistics are left enabled (no TRACY_NO_STATISTICS): csvexport needs
# them, and capture works either way.
find_package(Threads REQUIRED)
add_library(tracy_server STATIC
  "${TRACY_DIR}/public/common/TracySocket.cpp"
  "${TRACY_DIR}/public/common/TracyStackFrames.cpp"
  "${TRACY_DIR}/public/common/TracySystem.cpp"
  "${TRACY_DIR}/public/common/tracy_lz4.cpp"
  "${TRACY_DIR}/public/common/tracy_lz4hc.cpp"
  "${TRACY_DIR}/server/TracyBroadcast.cpp"
  "${TRACY_DIR}/server/TracyLocks.cpp"
  "${TRACY_DIR}/server/TracyMemory.cpp"
  "${TRACY_DIR}/server/TracyMmap.cpp"
  "${TRACY_DIR}/server/TracyPrint.cpp"
  "${TRACY_DIR}/server/TracySysUtil.cpp"
  "${TRACY_DIR}/server/TracyTaskDispatch.cpp"
  "${TRACY_DIR}/server/TracyTextureCompression.cpp"
  "${TRACY_DIR}/server/TracyThreadCompress.cpp"
  "${TRACY_DIR}/server/TracyWorker.cpp"
)
# As TracyServer's target_include_directories in cmake/server.cmake.
target_include_directories(tracy_server PUBLIC
  "${TRACY_DIR}/public/common"
  "${TRACY_DIR}/server"
)
target_link_libraries(tracy_server PUBLIC
  tracy_capstone
  tracy_ppqsort
  tracy_zstd
  Threads::Threads
)

# Capture helpers shared by tracy-capture and tracy-capture-daemon
# (cmake/util.cmake).
add_library(tracy_util STATIC
  "${TRACY_DIR}/util/CaptureConsole.cpp"
  "${TRACY_DIR}/util/CaptureFileBackup.cpp"
  "${TRACY_DIR}/util/CaptureOutput.cpp"
)
target_include_directories(tracy_util PUBLIC "${TRACY_DIR}/util")
target_link_libraries(tracy_util PUBLIC tracy_server)

add_executable(tracy-capture "${TRACY_DIR}/capture/src/capture.cpp")
target_include_directories(tracy-capture PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/gitref")
target_link_libraries(tracy-capture PRIVATE tracy_getopt tracy_server tracy_util)

add_executable(tracy-csvexport "${TRACY_DIR}/csvexport/src/csvexport.cpp")
target_include_directories(tracy-csvexport PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/gitref")
target_link_libraries(tracy-csvexport PRIVATE tracy_getopt tracy_server)

if(ANDROID)
  foreach(_TOOL tracy-capture tracy-csvexport)
    # See the source. A source of the executable itself, so the constructor
    # is not dropped as an unreferenced archive member.
    target_sources(${_TOOL} PRIVATE
      "${CMAKE_CURRENT_LIST_DIR}/../../bazel/tracy/no_heap_pointer_tagging.c"
    )
  endforeach()
endif()

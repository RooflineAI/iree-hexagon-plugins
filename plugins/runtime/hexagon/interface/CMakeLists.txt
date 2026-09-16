# Copyright 2026 RooflineAI GmbH
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Hand-written, because not supported by the bazel_to_cmake generator.

iree_package_ns(_PACKAGE_NS)
iree_package_name(_PACKAGE_NAME)

set(_HEXAGON_DSP_INTERFACE_DIR "${CMAKE_CURRENT_BINARY_DIR}/hexagon_dsp")
set(_HEXAGON_DSP_INTERFACE_HEADER "${_HEXAGON_DSP_INTERFACE_DIR}/hexagon_dsp.h")
set(_HEXAGON_DSP_INTERFACE_STUB_C "${_HEXAGON_DSP_INTERFACE_DIR}/hexagon_dsp_stub.c")
set(_HEXAGON_DSP_INTERFACE_SKEL_C "${_HEXAGON_DSP_INTERFACE_DIR}/hexagon_dsp_skel.c")

file(MAKE_DIRECTORY "${_HEXAGON_DSP_INTERFACE_DIR}")
add_custom_command(
  OUTPUT
    "${_HEXAGON_DSP_INTERFACE_HEADER}"
    "${_HEXAGON_DSP_INTERFACE_STUB_C}"
    "${_HEXAGON_DSP_INTERFACE_SKEL_C}"
  COMMAND
    hexagon_sdk::qaic
    -mdll
    -o "${_HEXAGON_DSP_INTERFACE_DIR}"
    "-I${HEXAGON_SDK_ROOT}/incs"
    "-I${HEXAGON_SDK_ROOT}/incs/stddef"
    "-I${HEXAGON_SDK_ROOT}/ipc/fastrpc/rpcmem/inc"
    "${CMAKE_CURRENT_SOURCE_DIR}/hexagon_dsp.idl"
  DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/hexagon_dsp.idl"
    hexagon_sdk::qaic
  VERBATIM
)
add_custom_target(hexagon_dsp_interface_gen DEPENDS
  "${_HEXAGON_DSP_INTERFACE_HEADER}"
  "${_HEXAGON_DSP_INTERFACE_STUB_C}"
  "${_HEXAGON_DSP_INTERFACE_SKEL_C}"
)

if(IREE_HEXAGON_ANDROID_BUILD)
iree_cc_library(
  NAME
    hexagon_dsp_stub_lib
  HDRS
    "${_HEXAGON_DSP_INTERFACE_HEADER}"
  SRCS
    "${_HEXAGON_DSP_INTERFACE_STUB_C}"
  DEPS
    hexagon_sdk::cdsprpc_android_aarch64
  INCLUDES
    "$<BUILD_INTERFACE:${_HEXAGON_DSP_INTERFACE_DIR}>"
  PUBLIC
)
add_dependencies(${_PACKAGE_NAME}_hexagon_dsp_stub_lib hexagon_dsp_interface_gen)
endif()

iree_cc_library(
  NAME
    hexagon_dsp_skel_hdr_lib
  HDRS
    "${_HEXAGON_DSP_INTERFACE_HEADER}"
  INCLUDES
    "$<BUILD_INTERFACE:${_HEXAGON_DSP_INTERFACE_DIR}>"
  PUBLIC
)
add_dependencies(${_PACKAGE_NAME}_hexagon_dsp_skel_hdr_lib hexagon_dsp_interface_gen)

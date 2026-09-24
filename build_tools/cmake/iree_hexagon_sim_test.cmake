# Copyright 2026 RooflineAI GmbH
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Registers a host-side CTest command that runs a cross-compiled Hexagon shared
# object in the SDK's v79 QuRT simulator. This function is called from the DSP
# cross-build tree: MODULE is a Hexagon target, while Python and hexagon-sim run
# on the build host.
function(iree_hexagon_sim_test)
  if(NOT IREE_BUILD_TESTS OR NOT IREE_HEXAGON_MLIR_DSP_BUILD)
    return()
  endif()

  cmake_parse_arguments(
    _RULE
    ""
    "NAME;MODULE;TIMEOUT"
    "LABELS"
    ${ARGN}
  )

  if(NOT _RULE_NAME OR NOT _RULE_MODULE)
    message(FATAL_ERROR "iree_hexagon_sim_test requires NAME and MODULE")
  endif()

  iree_package_ns(_PACKAGE_NS)
  iree_package_path(_PACKAGE_PATH)
  string(REGEX REPLACE "^::" "${_PACKAGE_NS}::" _MODULE_TARGET
    "${_RULE_MODULE}")
  if(NOT TARGET "${_MODULE_TARGET}")
    message(FATAL_ERROR
      "iree_hexagon_sim_test MODULE target does not exist: ${_MODULE_TARGET}")
  endif()

  foreach(_SDK_TARGET
      hexagon_sdk::hexagon_sim
      hexagon_sdk::run_main_on_hexagon_sim_v79
      hexagon_sdk::qurt_runelf_v79
      hexagon_sdk::qurt_model_v79)
    if(NOT TARGET "${_SDK_TARGET}")
      message(FATAL_ERROR
        "iree_hexagon_sim_test SDK target does not exist: ${_SDK_TARGET}")
    endif()
  endforeach()

  if(NOT TARGET iree_hexagon_sim_test_modules)
    add_custom_target(iree_hexagon_sim_test_modules)
  endif()
  get_target_property(_MODULE_REAL_TARGET "${_MODULE_TARGET}" ALIASED_TARGET)
  if(NOT _MODULE_REAL_TARGET)
    set(_MODULE_REAL_TARGET "${_MODULE_TARGET}")
  endif()
  add_dependencies(iree_hexagon_sim_test_modules "${_MODULE_REAL_TARGET}")

  set(_TEST_NAME "${_PACKAGE_PATH}/${_RULE_NAME}")
  add_test(
    NAME
      "${_TEST_NAME}"
    COMMAND
      "${Python3_EXECUTABLE}"
      "${IREE_HEXAGON_PLUGINS_ROOT}/build_tools/bazel/hexagon_sim_test_runner.py"
      --module "$<TARGET_FILE:${_MODULE_TARGET}>"
      --simulator "$<TARGET_FILE:hexagon_sdk::hexagon_sim>"
      --run-main "$<TARGET_FILE:hexagon_sdk::run_main_on_hexagon_sim_v79>"
      --runelf "$<TARGET_FILE:hexagon_sdk::qurt_runelf_v79>"
      --qurt-model "$<TARGET_FILE:hexagon_sdk::qurt_model_v79>"
  )

  if(NOT _RULE_TIMEOUT)
    set(_RULE_TIMEOUT 300)
  endif()
  list(APPEND _RULE_LABELS
    "${_PACKAGE_PATH}"
    "no-remote"
    "test-type=hexagon-sim"
  )
  set_property(TEST "${_TEST_NAME}" PROPERTY LABELS "${_RULE_LABELS}")
  set_property(TEST "${_TEST_NAME}" PROPERTY TIMEOUT "${_RULE_TIMEOUT}")
  iree_configure_test("${_TEST_NAME}")
endfunction()

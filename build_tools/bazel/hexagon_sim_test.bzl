# Copyright 2026 RooflineAI GmbH
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Build definitions for tests run in the Hexagon simulator."""

load("@aspect_rules_py//py:defs.bzl", "py_test")
load("//build_tools/bazel:platform_aliases.bzl", "hexagon_platform_alias")

_HOST_COMPAT = [
    "@platforms//cpu:x86_64",
    "@platforms//os:linux",
]

def hexagon_sim_test(name, module, tags = None):
    """Runs a cross-compiled shared-object module in the v79 QuRT simulator."""
    module_alias = name + "_module"
    hexagon_platform_alias(
        name = module_alias,
        actual = module,
        testonly = True,
    )

    py_test(
        name = name,
        srcs = [
            "//build_tools/bazel:hexagon_sim_test_runner.py",
        ],
        main = "hexagon_sim_test_runner.py",
        args = [
            "--module",
            "$(location :%s)" % module_alias,
            "--simulator",
            "$(location @hexagon_sdk//:hexagon_sim)",
            "--run-main",
            "$(location @hexagon_sdk//:run_main_on_hexagon_sim_v79)",
            "--runelf",
            "$(location @hexagon_sdk//:qurt_runelf_v79)",
            "--qurt-model",
            "$(location @hexagon_sdk//:qurt_model_v79)",
        ],
        data = [
            ":%s" % module_alias,
            "@hexagon_sdk//:hexagon_sim",
            "@hexagon_sdk//:hexagon_sim_host_libs",
            "@hexagon_sdk//:qurt_model_v79",
            "@hexagon_sdk//:qurt_runelf_v79",
            "@hexagon_sdk//:run_main_on_hexagon_sim_v79",
        ],
        size = "small",
        tags = ["no-remote"] + (tags or []),
        target_compatible_with = _HOST_COMPAT,
        timeout = "moderate",
    )

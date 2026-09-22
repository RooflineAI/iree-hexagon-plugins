# Copyright 2026 RooflineAI GmbH
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Build definitions for HMX simulator unit tests."""

load("@rules_cc//cc:cc_binary.bzl", "cc_binary")

_HEXAGON_COMPAT = [
    "//constraints:cpu_hexagon",
    "//constraints:os_qurt",
]

def hmx_test_module(name, src):
    """Defines a C-only Hexagon shared-object test module."""
    cc_binary(
        name = name,
        testonly = True,
        srcs = [src],
        # run_main_on_hexagon_sim resolves main() with dlsym after loading the
        # module, so keep the entry point visible under strict build defaults.
        copts = [
            "-std=gnu99",
            "-fvisibility=default",
        ],
        # The modules are C-only. Avoid the toolchain's default C++ runtime
        # dependency so run_main can use its built-in C and compiler runtimes.
        features = ["-default_cpp_link"],
        linkshared = True,
        target_compatible_with = _HEXAGON_COMPAT,
        deps = [
            ":test_support",
            "//plugins/runtime/hexagon/dsp/ukernel/hmx:hmx_ukernels",
        ],
    )

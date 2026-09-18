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
        copts = ["-std=gnu99"],
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

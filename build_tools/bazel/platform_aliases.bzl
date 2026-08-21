# Copyright 2026 RooflineAI GmbH
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Platform aliases for the hexagon runtime plugin: aarch64_android and hexagon.
"""

load("@rules_cc//cc/common:cc_info.bzl", "CcInfo")
load("@with_cfg.bzl", "with_cfg")

# Using with_cfg(native.alias) in the *_platform_alias definitions below results
# in platform_alias rules that do not accept the tags attribute. However, some
# platform aliases need the manual tag. Thus, implement a custom _alias rule,
# which will accept the tags attribute by default. This comes with a drawback
# that the resulting platform aliases do not work in bazel run commands, because
# the produced targets are not executable. However, no platform alias is
# currently needed in bazel run, so that's fine.
def _alias_impl(ctx):
    actual = ctx.attr.actual
    providers = []
    if DefaultInfo in actual:
        di = actual[DefaultInfo]
        providers.append(
            DefaultInfo(
                files = di.files,
                default_runfiles = di.default_runfiles,
                data_runfiles = di.data_runfiles,
            ),
        )
    if OutputGroupInfo in actual:
        providers.append(actual[OutputGroupInfo])
    if InstrumentedFilesInfo in actual:
        providers.append(actual[InstrumentedFilesInfo])
    if CcInfo in actual:
        providers.append(actual[CcInfo])
    return providers

_alias = rule(
    implementation = _alias_impl,
    attrs = {
        "actual": attr.label(mandatory = True),
    },
)

# The _platform_alias_internal variables are not actually "unused".
# It is apparently needed to store the value and keep it alive.
# Otherwise bazel fails with "unexported rule".

# buildifier: disable=unused-variable
_aarch64_android_platform_alias, _aarch64_android_platform_alias_internal = with_cfg(_alias).set("platforms", [Label("//platform:aarch64_android")]).build()

def aarch64_android_platform_alias(*, name, actual, tags = None, target_compatible_with = None):
    """
    Define an alias to be built with aarch64_android platform.

    The alias will build the same libs, binaries, ... as ``actual``, but for
    the aarch64_android platform (i.e. using the aarch64_android toolchain) - independent of
    the selected platform for the entire build.

    Use like regular alias(name = ..., actual = ..., [tags = ...], [target_compatible_with = ...])
    """
    _aarch64_android_platform_alias(name = name, actual = actual, tags = tags, target_compatible_with = target_compatible_with)

# Builds ``actual`` for aarch64_android with IREE runtime tracing set to "tracy"
# (profiling enabled), scoped to this subtree via a config transition. Because
# tracing_provider feeds iree/base (and thus the host IREE compiler), scoping it
# here keeps the host compiler in the default config and reuses its build cache.
# TRACY_TIMER_FALLBACK is applied per-file to tracy.cc (required under bazel
# sandboxing) to avoid re-keying the whole compiler tree.
# buildifier: disable=unused-variable
_aarch64_android_tracy_platform_alias, _aarch64_android_tracy_platform_alias_internal = with_cfg(_alias).set(
    "platforms",
    [Label("//platform:aarch64_android")],
).set(
    Label("@iree//runtime/src/iree/base/tracing:tracing_provider"),
    "tracy",
).extend(
    "per_file_copt",
    [".*base/tracing/tracy.cc@-DTRACY_TIMER_FALLBACK"],
).build()

def aarch64_android_tracy_platform_alias(*, name, actual, tags = None, target_compatible_with = None):
    """
    Define an aarch64_android alias built with IREE runtime tracing set to tracy.

    Like ``aarch64_android_platform_alias`` but additionally enables profiling
    (tracy) for the subtree of ``actual``. The tracy config is scoped via a
    transition, so the host IREE compiler stays in the default config and reuses
    its build cache.

    Use like regular alias(name = ..., actual = ..., [tags = ...], [target_compatible_with = ...])
    """
    _aarch64_android_tracy_platform_alias(name = name, actual = actual, tags = tags, target_compatible_with = target_compatible_with)

# buildifier: disable=unused-variable
_hexagon_platform_alias, _hexagon_platform_alias_internal = with_cfg(_alias).set("platforms", [Label("//platform:hexagon")]).build()

def hexagon_platform_alias(*, name, actual, tags = None):
    """
    Define an alias to be built with hexagon platform.

    The alias will build the same libs, binaries, ... as ``actual``, but for
    the hexagon platform (i.e. using the hexagon toolchain) - independent of
    the selected platform for the entire build.

    Use like regular alias(name = ..., actual = ..., [tags = ...])
    """
    _hexagon_platform_alias(name = name, actual = actual, tags = tags)

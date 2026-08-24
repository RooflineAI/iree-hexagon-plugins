# Copyright 2026 RooflineAI GmbH
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Bazel macros for building hexagon compiler and runtime plugin libraries.
"""

load(
    "@iree//build_tools/bazel:build_defs.oss.bzl",
    "iree_cc_library",
)
load("@llvm-project//mlir:build_defs.bzl", "cc_headers_only")

def _sanitize_label_fragment(label):
    return str(label) \
        .replace("@", "") \
        .replace("//", "_") \
        .replace("/", "_") \
        .replace(":", "_") \
        .replace("-", "_") \
        .replace(".", "_") \
        .replace("+", "_")

def _unique_wrapped_name(prefix, src, used_names):
    # This is meant as a safeguard for collisions after sanitizing the label,
    # but I actually doubt we will ever have one.
    base_name = "{}_{}".format(prefix, _sanitize_label_fragment(Label(src)))
    count = used_names.get(base_name, 0)
    used_names[base_name] = count + 1
    return "{}_{}".format(base_name, count + 1)

def _cc_headers_only_group(prefix, hdr_deps, visibility = None):
    # We need to generate wrapper rule names
    # `hdr_deps` therefore accepts a plain label list,
    # and this helper derives local wrapper names automatically.
    wrapped_targets = []
    used_names = {}
    for src in hdr_deps:
        wrapped_name = _unique_wrapped_name(prefix, src, used_names)
        cc_headers_only(
            name = wrapped_name,
            src = src,
            visibility = visibility,
        )
        wrapped_targets.append(":" + wrapped_name)
    return wrapped_targets

def _hexagon_library(name, srcs, hdrs, deps, hdr_deps, **kwargs):
    wrapped_hdr_deps = _cc_headers_only_group(prefix = name, hdr_deps = hdr_deps)
    iree_cc_library(
        name = name,
        srcs = srcs,
        hdrs = hdrs,
        deps = deps + wrapped_hdr_deps,
        linkstatic = True,
        alwayslink = True,
        **kwargs
    )

def hexagon_plugin_library(name, srcs, hdrs, deps, hdr_deps, **kwargs):
    _hexagon_library(
        name = name,
        srcs = srcs,
        hdrs = hdrs,
        deps = deps,
        hdr_deps = hdr_deps,
        **kwargs
    )

def hexagon_mlir_overlay_library(name, srcs, hdrs = None, deps = None, hdr_deps = None, **kwargs):
    _hexagon_library(
        name = name,
        srcs = srcs,
        hdrs = hdrs or [],
        deps = deps,
        hdr_deps = hdr_deps,
        **kwargs
    )

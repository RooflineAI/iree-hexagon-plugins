"""
Bazel macros for building cellar-hexagon compiler and runtime plugin libraries.
This file is made in the image of the nxp plugin equivalent file.
"""

load(
    "@iree//build_tools/bazel:build_defs.oss.bzl",
    "iree_cc_library",
)
load("@llvm-project//mlir:build_defs.bzl", "cc_headers_only")

def _cc_headers_only_group(prefix, name_to_src, visibility = None):
    wrapped_targets = []
    for name, src in name_to_src.items():
        wrapped_name = "{}_{}".format(prefix, name)
        cc_headers_only(
            name = wrapped_name,
            src = src,
            visibility = visibility,
        )
        wrapped_targets.append(":" + wrapped_name)
    return wrapped_targets

def cellar_hexagon_plugin_library(name, srcs, hdrs, deps, hdr_deps, **kwargs):
    wrapped_hdr_deps = _cc_headers_only_group(prefix = name, name_to_src = hdr_deps)
    iree_cc_library(
        name = name,
        srcs = srcs,
        hdrs = hdrs,
        deps = deps + wrapped_hdr_deps,
        linkstatic = True,
        alwayslink = True,
        **kwargs
    )

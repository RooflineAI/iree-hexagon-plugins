# Copyright 2026 RooflineAI GmbH
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
This file provides a rule to download the free-to-download version of the
Hexagon SDK.
"""

def _hexagon_sdk_repository_impl(repository_ctx):
    repository_ctx.download_and_extract(
        url = "https://github.com/snapdragon-toolchain/hexagon-sdk/releases/download/v6.4.0.2/hexagon-sdk-v6.4.0.2-amd64-lnx.tar.xz",
        sha256 = "b4a57a774795cf12da19a777a5d306e970905bf9758a4c4765e5e4593428ae0b",
        stripPrefix = "6.4.0.2",
    )
    repo_name = repository_ctx.name
    repository_ctx.file(
        "BUILD.bazel",
        content = """
package(default_visibility = ["//visibility:public"])

# This defines a target for one specific file - the Hexagon clang binary - in
# the Hexagon SDK. This can be used as an anchor to construct paths to other
# files and directories in the SDK. For example, the Hexagon toolchain uses this
# to define paths to include directories, libs, ...
filegroup(
    name = "hexagon_clang",
    srcs = ["tools/HEXAGON_Tools/19.0.04/Tools/bin/hexagon-clang"],
)

# Single file target to the QAIC interface generator tool.
filegroup(
    name = "qaic",
    srcs = ["ipc/fastrpc/qaic/bin/qaic"],
)

# All files in the inc subdir, used by QAIC
filegroup(
    name = "incs_tree",
    srcs = glob(["incs/**", "ipc/fastrpc/rpcmem/inc/**"]),
)

# This file group refers to all files in the SDK. It can be used as a dependency
# for rules that depend on the entire Hexagon SDK. For example, the Hexagon
# toochain uses this to amke sure all Hexagon SDK files are in the sandbox
# when compiling for Hexagon.
filegroup(
    name = "sdk_tree",
    srcs = glob(["**"]),
)

cc_library(
    name = "qurt_headers",
    hdrs = glob([
        "rtos/qurt/computev79/include/qurt/**",
        "rtos/qurt/computev79/include/posix/**",
    ]),
    includes = [
        "rtos/qurt/computev79/include/qurt",
        "rtos/qurt/computev79/include/posix",
    ],
)

cc_library(
    name = "qhl_hvx_headers",
    hdrs = glob([
        "libs/qfe/inc/**",
        "libs/qhl_hvx/inc/**",
    ]),
    includes = [
        "libs/qfe/inc",
        "libs/qhl_hvx/inc",
    ],
)

cc_library(
    name = "hexagon_toolchain_bit_headers",
    hdrs = glob([
        "tools/HEXAGON_Tools/19.0.04/Tools/target/hexagon/include/c++/v1/__bit/**",
    ]),
    includes = [
        "tools/HEXAGON_Tools/19.0.04/Tools/target/hexagon/include/c++/v1/__bit",
    ],
)

cc_import(
    name = "cdsprpc_android_aarch64",
    hdrs = glob(["incs/**", "ipc/fastrpc/rpcmem/inc/**"]),
    shared_library = "ipc/fastrpc/remote/ship/android_aarch64/libcdsprpc.so",
    includes = [
        "external/{repo_name}/incs",
        "external/{repo_name}/incs/stddef",
        "external/{repo_name}/ipc/fastrpc/rpcmem/inc",
    ],
)
""".format(repo_name = repo_name),
    )

hexagon_sdk_repository = repository_rule(
    implementation = _hexagon_sdk_repository_impl,
)

# Copyright 2026 RooflineAI GmbH
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
This file defines a Bazel rule for configuring a aarch64 cross-compilation
toolchain for Android.
"""

load("@android_ndk_detect//:defs.bzl", "ANDROID_NDK_LIB_CLANG_DIR", "ANDROID_NDK_TOOLCHAIN_DIR", "ANDROID_NDK_VERSION")
load("@bazel_tools//tools/build_defs/cc:action_names.bzl", "ACTION_NAMES")
load(
    "@bazel_tools//tools/cpp:cc_toolchain_config_lib.bzl",
    "feature",
    "flag_group",
    "flag_set",
    "tool_path",
)
load("@rules_cc//cc/common:cc_common.bzl", "cc_common")

# NDK dir is defined once in MODULE.bazel (the ndk_dir attribute of the
# android_ndk_detect() call) and loaded here via @android_ndk_detect//:defs.bzl.

def _aarch64_toolchain_config_impl(ctx):
    # Define the locations of the necessary tools in the aarch64 cross-compiler
    clang = ANDROID_NDK_TOOLCHAIN_DIR + "/bin/aarch64-linux-android" + ANDROID_NDK_VERSION + "-clang"
    tool_paths = [
        tool_path(name = "cpp", path = clang),
        tool_path(name = "gcc", path = clang),
        tool_path(name = "ld", path = ANDROID_NDK_TOOLCHAIN_DIR + "/bin/ld.lld"),
    ] + [
        tool_path(name = tool, path = ANDROID_NDK_TOOLCHAIN_DIR + "/bin/llvm-" + tool)
        for tool in ["ar", "nm", "objdump", "strip"]
    ]

    # -- Link action lists ------------------------------------------------
    # Bazel selects the link action based on the target type:
    #   cpp_link_executable            — cc_binary / cc_test (linkshared = False)
    #   cpp_link_dynamic_library       — cc_binary(linkshared = True) / cc_shared_library
    #   cpp_link_nodeps_dynamic_library — internal .so Bazel creates for a cc_library
    # Each fired action is matched against every flag_set whose `actions` list
    # contains it, so flag_sets using all_link_actions apply to every link.
    all_link_actions = [
        ACTION_NAMES.cpp_link_executable,
        ACTION_NAMES.cpp_link_dynamic_library,
        ACTION_NAMES.cpp_link_nodeps_dynamic_library,
    ]
    executable_link_actions = [
        ACTION_NAMES.cpp_link_executable,
    ]
    dynamic_library_link_actions = [
        ACTION_NAMES.cpp_link_dynamic_library,
        ACTION_NAMES.cpp_link_nodeps_dynamic_library,
    ]

    # -- Compile features --------------------------------------------------

    aarch64_compile_flags = feature(
        name = "aarch64_arch_flags",
        enabled = True,
        flag_sets = [
            flag_set(
                actions = [
                    ACTION_NAMES.c_compile,
                    ACTION_NAMES.cpp_compile,
                ],
                flag_groups = [flag_group(flags = [
                    "-march=" + ctx.attr.march,
                ])],
            ),
            flag_set(
                actions = [
                    ACTION_NAMES.assemble,
                    ACTION_NAMES.preprocess_assemble,
                    ACTION_NAMES.cpp_link_dynamic_library,
                    ACTION_NAMES.cpp_link_executable,
                ],
                flag_groups = [flag_group(flags = [
                    "--target=aarch64-linux-android" + ANDROID_NDK_VERSION,
                    "-march=" + ctx.attr.march,
                ])],
            ),
        ],
    )

    android_compile_flags = feature(
        name = "android_compile_flags",
        enabled = True,
        flag_sets = [
            flag_set(
                actions = [
                    ACTION_NAMES.c_compile,
                    ACTION_NAMES.cpp_compile,
                ],
                flag_groups = [flag_group(flags = [
                    "-fPIC",
                    "-isystem",
                    ANDROID_NDK_TOOLCHAIN_DIR + "/sysroot/usr/include/c++/v1",
                    "-D_ANDROID_",
                    "-DANDROID",
                ])],
            ),
        ],
    )

    iree_specific_flags = feature(
        name = "iree_specific_flags",
        enabled = True,
        flag_sets = [
            flag_set(
                actions = [
                    ACTION_NAMES.c_compile,
                    ACTION_NAMES.cpp_compile,
                ],
                flag_groups = [flag_group(flags = [
                    "-DIREE_TIME_NOW_FN={ return 0; }",
                    "-DIREE_CPUINFO_TARGET=\"\"",
                    "-DIREE_TASK_CPUINFO_DISABLED=1",  # FIXME CPU info disabled
                ])],
            ),
        ],
    )

    # This will be automatically enabled if the compilation mode is set to
    # optimized when running the build, via "bazel build -c opt ..."
    optimization = feature(
        name = "opt",
        flag_sets = [
            flag_set(
                actions = [
                    ACTION_NAMES.c_compile,
                    ACTION_NAMES.cpp_compile,
                ],
                flag_groups = [flag_group(flags = [
                    "-O2",
                    "-DNDEBUG",
                ])],
            ),
        ],
    )

    # Ensure the GNU C++ standard library is linked for C++ binaries:
    default_cpp_link = feature(
        name = "default_cpp_link",
        enabled = True,
        flag_sets = [
            flag_set(
                actions = all_link_actions,
                flag_groups = [flag_group(flags = ["-lc++", "-lm"])],
            ),
        ],
    )

    # -- Link features -----------------------------------------------------
    # android_linker_flags is split into three flag_sets:
    #   (1) common flags for all link outputs,
    #   (2) executable-specific flags (PIE, dynamic linker, CRT for binaries),
    #   (3) shared-library-specific flags (PIC, CRT for .so files).

    _sysroot_android_dir = ANDROID_NDK_TOOLCHAIN_DIR + "/sysroot/usr/lib/aarch64-linux-android/" + ANDROID_NDK_VERSION
    android_linker_flags = feature(
        name = "android_linker_flags",
        enabled = True,
        flag_sets = [
            flag_set(
                actions = all_link_actions,
                flag_groups = [flag_group(flags = [
                    "-target",
                    "aarch64-linux-android" + ANDROID_NDK_VERSION,
                    "-Wl,-z,nocopyreloc",
                    "-Wl,-rpath-link=" + _sysroot_android_dir,
                    "-nostdlib",
                    "-Bdynamic",
                    "-Wl,-llog",
                    "-Wl,-ldl",
                    "-L" + _sysroot_android_dir,
                    "-lm",
                    "-lc",
                    "-ldl",
                    "-lunwind",
                    "-L" + ANDROID_NDK_LIB_CLANG_DIR + "/lib/linux",
                    "-lclang_rt.builtins-aarch64-android",
                ])],
            ),
            flag_set(
                actions = executable_link_actions,
                flag_groups = [flag_group(flags = [
                    "-fPIE",
                    "-Wl,-dynamic-linker,/system/bin/linker64",
                    "-Wl,-rpath,$ORIGIN/../lib",
                    "-pie",
                    _sysroot_android_dir + "/crtbegin_dynamic.o",
                    _sysroot_android_dir + "/crtend_android.o",
                ])],
            ),
            flag_set(
                actions = dynamic_library_link_actions,
                flag_groups = [flag_group(flags = [
                    "-fPIC",
                    "-Wl,-rpath,$ORIGIN/../lib",
                    "-Wl,-unresolved-symbols=ignore-in-shared-libs",
                    _sysroot_android_dir + "/crtbegin_so.o",
                    _sysroot_android_dir + "/crtend_so.o",
                ])],
            ),
        ],
    )

    # Opt-in escalation over the default -Wl,-unresolved-symbols=ignore-in-shared-libs
    # above.  When enabled, the linker tolerates *all* unresolved symbols in the
    # output .so — not just those coming from other shared libraries.  This is
    # required for plugin .so files whose missing symbols are provided by the host
    # process at dlopen() time (like our dynamic plugin). Enable per-target with:
    #   features = ["allow_all_undefined_symbols"]
    allow_all_undefined_symbols = feature(
        name = "allow_all_undefined_symbols",
        enabled = False,
        flag_sets = [
            flag_set(
                actions = dynamic_library_link_actions,
                flag_groups = [flag_group(flags = [
                    "-Wl,--unresolved-symbols=ignore-all",
                ])],
            ),
        ],
    )

    # List of all features to enable in this toolchain configuration
    features = [
        aarch64_compile_flags,
        android_compile_flags,
        iree_specific_flags,
        default_cpp_link,
        optimization,
        android_linker_flags,
        allow_all_undefined_symbols,
    ]

    # Specify include directories for the cross-compiler's default includes (C
    # and C++ standard library headers)
    cxx_builtin_include_directories = [
        ANDROID_NDK_LIB_CLANG_DIR + "/include",
        ANDROID_NDK_TOOLCHAIN_DIR + "/sysroot/usr/include/aarch64-linux-android",
        ANDROID_NDK_TOOLCHAIN_DIR + "/sysroot/usr/include",
    ]

    return cc_common.create_cc_toolchain_config_info(
        ctx = ctx,
        toolchain_identifier = "aarch64-android",
        host_system_name = "local",
        target_system_name = "local",
        target_cpu = "aarch64",
        target_libc = "glibc",
        compiler = "clang",
        abi_libc_version = "unknown",
        tool_paths = tool_paths,
        cxx_builtin_include_directories = cxx_builtin_include_directories,
        features = features,
    )

# You can extract this information by running:
#    gcc -Q --help=target
# on the target platform (assuming gcc is installed).
AARCH64_ARCH = "armv8-a"

# Define the rule that provides CcToolchainConfigInfo
aarch64_android_toolchain_config = rule(
    implementation = _aarch64_toolchain_config_impl,
    attrs = {
        "march": attr.string(
            default = AARCH64_ARCH,
            doc = "aarch64 architecture to target (e.g., armv8-a).",
        ),
    },
    provides = [CcToolchainConfigInfo],
)

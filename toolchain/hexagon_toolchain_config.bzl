"""
This file defines a Bazel rule for configuring a hexagon cross-compilation
toolchain. This toolchain depends on the download of the Hexagon SDK.
"""

load("@bazel_skylib//lib:paths.bzl", "paths")
load("@bazel_tools//tools/build_defs/cc:action_names.bzl", "ACTION_NAMES")
load(
    "@bazel_tools//tools/cpp:cc_toolchain_config_lib.bzl",
    "feature",
    "flag_group",
    "flag_set",
    "tool_path",
)
load("@rules_cc//cc/common:cc_common.bzl", "cc_common")

HEXAGON_SDK_RELATIVE_INCLUDE_DIRS = [
    "incs",
    "incs/stddef",
    "ipc/fastrpc/incs",
    "ipc/fastrpc/rpcmem/inc",
    "incs/qnx",
    "libs/common/qnx/ship/hexagon_Debug_toolv88_vXX",
    "utils/examples",
    "ipc/fastrpc/rtld/ship/hexagon_toolv88_vXX",
    "ipc/fastrpc/remote/ship/hexagon_Debug_toolv88_vXX",
    "ipc/fastrpc/rtld/ship/inc",
    "libs/atomic/inc",
    "utils/sim_utils/inc",
    "libs/atomic/prebuilt/hexagon_toolv88_vXX",
    "utils/sim_utils/prebuilt/hexagon_toolv88_vXX",
    "rtos/qurt/computevXX/include/qurt",
    "rtos/qurt/computevXX/include/posix",
]

def _hexagon_toolchain_config_impl(ctx):
    hexagon_clang = ctx.file.hexagon_clang.path
    hexagon_tools_dir = paths.dirname(paths.dirname(hexagon_clang))

    # Derive the SDK root from the hexagon_clang path to avoid hardcoding the
    # canonical repo name (which changes across Bazel major versions).
    # hexagon_clang path: external/<repo>/tools/HEXAGON_Tools/.../bin/hexagon-clang
    hexagon_sdk_root = hexagon_clang.split("/tools/HEXAGON_Tools/")[0]
    hexagon_sdk_include_directories = [
        paths.join(hexagon_sdk_root, rel_dir.replace("vXX", "v" + ctx.attr.mv))
        for rel_dir in HEXAGON_SDK_RELATIVE_INCLUDE_DIRS
    ]

    # Define the locations of the necessary tools in the hexagon cross-compiler
    # We need to use trampoline scripts to call tools in external/hexagon_sdk/
    # because tool_path accepts only relative paths that don't go up (i.e.
    # refer to files inside the toolchain dir).
    tool_paths = [
        tool_path(name = "cpp", path = "hexagon_trampolines/clang"),
        tool_path(name = "gcc", path = "hexagon_trampolines/clang"),
        tool_path(name = "ld", path = "hexagon_trampolines/link"),
    ] + [
        tool_path(name = tool, path = "hexagon_trampolines/hexagon-" + tool)
        for tool in ["ar", "nm", "objdump", "strip"]
    ]

    hexagon_arch_flags = feature(
        name = "hexagon_arch_flags",
        enabled = True,
        flag_sets = [
            flag_set(
                actions = [
                    ACTION_NAMES.c_compile,
                    ACTION_NAMES.cpp_compile,
                    ACTION_NAMES.assemble,
                    ACTION_NAMES.preprocess_assemble,
                    ACTION_NAMES.cpp_link_dynamic_library,
                    ACTION_NAMES.cpp_link_executable,
                ],
                #flag_groups = [flag_group(flags = [ ... ])],
            ),
        ],
    )

    def add_isystem(inc_dirs):
        return [
            flag
            for inc_dir in inc_dirs
            for flag in ["-isystem", inc_dir]
        ]

    # Need to override the builtin include dirs, which are absolute paths,
    # with relative versionf of the same. Otherwise bazel complains that
    # "non-builtin files with absolute paths" are included. It's not possible
    # to list the absolute include paths as builtin include paths because
    # bazel does not allow to get the absolute path of the downloaded Hexagon
    # SDK.
    # The dirs come from the output of "clang -x c -E -v /dev/null" and
    # "clang -x c++ -E -v /dev/null" and
    hexagon_builtin_c_include_dirs = [
        paths.join(hexagon_tools_dir, "target/hexagon/include"),
        paths.join(hexagon_tools_dir, "lib/clang/19/include"),
    ]
    hexagon_builtin_cpp_include_dirs = [
        paths.join(hexagon_tools_dir, "target/hexagon/include/c++/v1"),
    ] + hexagon_builtin_c_include_dirs
    hexagon_builtin_includes = feature(
        name = "hexagon_builtin_includes",
        enabled = True,
        flag_sets = [
            flag_set(
                actions = [
                    ACTION_NAMES.c_compile,
                ],
                flag_groups = [flag_group(
                    flags = add_isystem(hexagon_builtin_c_include_dirs),
                )],
            ),
            flag_set(
                actions = [
                    ACTION_NAMES.cpp_compile,
                ],
                flag_groups = [flag_group(
                    flags = add_isystem(hexagon_builtin_cpp_include_dirs),
                )],
            ),
        ],
    )

    hexagon_sdk_include_flags = feature(
        name = "hexagon_sdk_include_flags",
        enabled = True,
        flag_sets = [
            flag_set(
                actions = [
                    ACTION_NAMES.c_compile,
                    ACTION_NAMES.cpp_compile,
                ],
                flag_groups = [flag_group(
                    flags = add_isystem(hexagon_sdk_include_directories),
                )],
            ),
        ],
    )

    hexagon_sdk_compile_flags = feature(
        name = "hexagon_sdk_compile_flags",
        enabled = True,
        flag_sets = [
            flag_set(
                actions = [
                    ACTION_NAMES.c_compile,
                    ACTION_NAMES.cpp_compile,
                ],
                flag_groups = [flag_group(flags = [
                    "-mv" + ctx.attr.mv,
                    "-fdata-sections",
                    "-fstack-protector",
                    "-fpic",
                    "-D__V_DYNAMIC__",
                    "-mhvx",
                    "-mhvx-length=128B",
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
    link_actions = [
        ACTION_NAMES.cpp_link_executable,
        ACTION_NAMES.cpp_link_dynamic_library,
        ACTION_NAMES.cpp_link_nodeps_dynamic_library,
    ]
    default_cpp_link = feature(
        name = "default_cpp_link",
        enabled = True,
        flag_sets = [
            flag_set(
                actions = link_actions,
                flag_groups = [flag_group(flags = ["-lstdc++", "-lm"])],
            ),
        ],
    )

    hexagon_sdk_dynamic_lib_flags = feature(
        name = "hexagon_sdk_dynamic_lib_flags",
        enabled = True,
        flag_sets = [flag_set(actions = [ACTION_NAMES.cpp_link_dynamic_library], flag_groups = [flag_group([
            "-mv" + ctx.attr.mv,
            "-Wl,--defsym=ISDB_TRUSTED_FLAG=2",
            "-Wl,--defsym=ISDB_SECURE_FLAG=2",
            "-Wl,--no-threads",
            "-fpic",
            "-shared",
            "-Wl,-Bsymbolic",
            "-Wl,--wrap=malloc",
            "-Wl,--wrap=calloc",
            "-Wl,--wrap=free",
            "-Wl,--wrap=realloc",
            "-Wl,--wrap=memalign",
        ])])],
    )

    # List of all features to enable in this toolchain configuration
    features = [hexagon_arch_flags, hexagon_builtin_includes, hexagon_sdk_include_flags, hexagon_sdk_compile_flags, iree_specific_flags, optimization, default_cpp_link, hexagon_sdk_dynamic_lib_flags]

    return cc_common.create_cc_toolchain_config_info(
        ctx = ctx,
        toolchain_identifier = "hexagon",
        host_system_name = "local",
        target_system_name = "local",
        target_cpu = "hexagon",
        target_libc = "hexagon-libc",
        compiler = "clang",
        abi_libc_version = "unknown",
        tool_paths = tool_paths,
        cxx_builtin_include_directories = hexagon_builtin_cpp_include_dirs + hexagon_sdk_include_directories,
        features = features,
    )

# Define the rule that provides CcToolchainConfigInfo
hexagon_toolchain_config = rule(
    implementation = _hexagon_toolchain_config_impl,
    attrs = {
        "mv": attr.string(
            default = "79",
            doc = "hexagon core version to target.",
        ),
        "hexagon_clang": attr.label(
            allow_single_file = True,
            default = Label("@hexagon_sdk//:hexagon_clang"),
            doc = "Executable used to locate the Hexagon SDK installation.",
        ),
    },
    provides = [CcToolchainConfigInfo],
)

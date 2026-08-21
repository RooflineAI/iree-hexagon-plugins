"""
Repository rule to detect Android NDK installation.

Creates a repository with:

- A ``config_setting`` named ``android_ndk_available`` that matches any
  platform configuration when the NDK directory is found on disk, and never
  matches when it is absent.

- A ``filegroup`` named ``libcxx_shared`` exposing ``libc++_shared.so`` from
  the NDK sysroot (empty when the NDK is absent).

Typical usage in BUILD files:

    target_compatible_with = select({
        "@android_ndk_detect//:android_ndk_available": [],
        "//conditions:default": ["@platforms//:incompatible"],
    })

    # To consume libc++_shared.so:
    srcs = ["@android_ndk_detect//:libcxx_shared"]
"""

_LIBCXX_REL_PATH = (
    "toolchains/llvm/prebuilt/linux-x86_64/sysroot" +
    "/usr/lib/aarch64-linux-android/libc++_shared.so"
)

def _android_ndk_detect_impl(repo_ctx):
    ndk_dir = repo_ctx.attr.ndk_dir
    ndk_found = repo_ctx.path(ndk_dir).exists

    toolchain_dir = ndk_dir + "/toolchains/llvm/prebuilt/linux-x86_64"

    # _android_ndk_flag defaults to True when the NDK is present and False when
    # absent.  android_ndk_available matches iff the flag is True, so select()
    # expressions fall through to //conditions:default when the NDK is missing.
    flag_default = "True" if ndk_found else "False"

    if ndk_found:
        # Symlink libc++_shared.so from the NDK sysroot into this repo so it
        # can be exposed as a filegroup target.
        repo_ctx.symlink(ndk_dir + "/" + _LIBCXX_REL_PATH, "libc++_shared.so")
        libcxx_srcs = '["libc++_shared.so"]'

        ndk_version = ndk_dir.split("/")[-1].split(".")[0]

        # lib/clang/ contains exactly one subdirectory named after the bundled
        # clang version (e.g. "19").  Use readdir() to resolve it rather than
        # hardcoding the version.
        clang_entries = repo_ctx.path(toolchain_dir + "/lib/clang").readdir()
        lib_clang_dir = str(clang_entries[0])
    else:
        libcxx_srcs = "[]"
        ndk_version = "0"
        lib_clang_dir = toolchain_dir + "/lib/clang/unknown"

    # Expose the NDK directory path as a Starlark constant so other .bzl files
    # can load it instead of repeating the literal.  This makes MODULE.bazel
    # the single source of truth for the NDK path.
    repo_ctx.file("defs.bzl", """
ANDROID_NDK_DIR = "{ndk_dir}"
ANDROID_NDK_VERSION = "{ndk_version}"
ANDROID_NDK_TOOLCHAIN_DIR = "{toolchain_dir}"
ANDROID_NDK_LIB_CLANG_DIR = "{lib_clang_dir}"
""".format(ndk_dir = ndk_dir, ndk_version = ndk_version, toolchain_dir = toolchain_dir, lib_clang_dir = lib_clang_dir))

    repo_ctx.file("BUILD.bazel", """
load("@bazel_skylib//rules:common_settings.bzl", "bool_flag")

# Internal flag whose default reflects whether the NDK was detected on disk.
# Can be overridden on the command line if needed.
bool_flag(
    name = "_android_ndk_flag",
    build_setting_default = {flag_default},
    visibility = ["//visibility:public"],
)

config_setting(
    name = "android_ndk_available",
    flag_values = {{":_android_ndk_flag": "True"}},
    visibility = ["//visibility:public"],
)

filegroup(
    name = "libcxx_shared",
    srcs = {libcxx_srcs},
    visibility = ["//visibility:public"],
)
""".format(flag_default = flag_default, libcxx_srcs = libcxx_srcs))

android_ndk_detect = repository_rule(
    implementation = _android_ndk_detect_impl,
    attrs = {
        "ndk_dir": attr.string(
            doc = "Absolute path to the Android NDK root directory to detect.",
        ),
    },
    # local = True tells Bazel to re-run this rule whenever local state may
    # have changed (e.g. after installing or removing the NDK).
    local = True,
    doc = """Detects whether the Android NDK is installed at the given path.

Exposes ``android_ndk_available`` (config_setting) and ``libcxx_shared``
(filegroup) targets for use by targets that require the Android NDK toolchain.
""",
)

"""
This file provides a rule to download the Hexagon Kernel Library (HexKL).
"""

def _hexkl_repository_impl(repository_ctx):
    outer_zip_name = "Hexagon_KL.Core.1.0.0.Linux-Any.zip"
    inner_zip_name = "hexkl-1.0.0-beta1-6.4.0.0.zip"

    repository_ctx.download(
        url = "https://softwarecenter.qualcomm.com/api/download/software/tools/Hexagon_KL/Linux/1.0.0/" + outer_zip_name,
        output = outer_zip_name,
    )

    # Extract outer archive, then the embedded inner archive that contains
    # the actual hexkl_addon directory.
    repository_ctx.extract(outer_zip_name)

    inner_zip_path = inner_zip_name
    if not repository_ctx.path(inner_zip_path).exists:
        nested_inner_zip_path = "Hexagon_KL.Core.1.0.0.Linux-Any/" + inner_zip_name
        if repository_ctx.path(nested_inner_zip_path).exists:
            inner_zip_path = nested_inner_zip_path
        else:
            fail("Could not find inner HexKL archive: " + inner_zip_name)

    repository_ctx.extract(inner_zip_path)

    if not repository_ctx.path("hexkl_addon").exists:
        fallback_addon_path = "hexkl-1.0.0-beta1-6.4.0.0/hexkl_addon"
        if repository_ctx.path(fallback_addon_path).exists:
            repository_ctx.symlink(fallback_addon_path, "hexkl_addon")
        else:
            fail("Could not locate extracted hexkl_addon directory")

    repository_ctx.file(
        "BUILD.bazel",
        content = """
package(default_visibility = ["//visibility:public"])

# All files in HexKL.
filegroup(
    name = "hexkl_tree",
    srcs = glob(["**"]),
)

# Header tree used when compiling consumers of the HexKL APIs.
filegroup(
    name = "hexkl_includes_tree",
    srcs = glob(["hexkl_addon/include/**"]),
)

# All libhexkl_micro.a variants shipped in HexKL.
filegroup(
    name = "libhexkl_micro_tree",
    srcs = glob(["hexkl_addon/lib/**/libhexkl_micro.a"]),
)

cc_library(
    name = "hexkl_headers",
    hdrs = glob(["hexkl_addon/include/**"]),
    includes = [
        "external/_main~cellar_hexagon_repos~hexkl/hexkl_addon/include",
    ],
)

cc_import(
    name = "hexkl_micro_v73",
    hdrs = glob(["hexkl_addon/include/**"]),
    includes = [
        "external/_main~cellar_hexagon_repos~hexkl/hexkl_addon/include",
    ],
    static_library = "hexkl_addon/lib/hexagon_toolv19_v73/libhexkl_micro.a",
)

cc_import(
    name = "hexkl_micro_v75",
    hdrs = glob(["hexkl_addon/include/**"]),
    includes = [
        "external/_main~cellar_hexagon_repos~hexkl/hexkl_addon/include",
    ],
    static_library = "hexkl_addon/lib/hexagon_toolv19_v75/libhexkl_micro.a",
)

cc_import(
    name = "hexkl_micro_v79",
    hdrs = glob(["hexkl_addon/include/**"]),
    includes = [
        "external/_main~cellar_hexagon_repos~hexkl/hexkl_addon/include",
    ],
    static_library = "hexkl_addon/lib/hexagon_toolv19_v79/libhexkl_micro.a",
)

alias(
    name = "hexkl_micro",
    actual = ":hexkl_micro_v79",
)
""",
    )

hexkl_repository = repository_rule(
    implementation = _hexkl_repository_impl,
)

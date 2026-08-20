"""
Helpers to build simple Hexagon RPC sample apps (ARM host + DSP skeleton)
similar to the legacy `my_build.bash` script in testing/hexagon_loader.

This wires together:
- QAIC interface generation (stub/skel)
- ARM/aarch64 Android host binary
- Hexagon DSP shared library and a cross-platform alias to force the Hexagon
  toolchain
- Optional zipped "ship" bundle with both artifacts.
"""

load("//build_tools/bazel:hexagon_interface.bzl", "hexagon_interface")
load("@roof_mlir//:build_tools/bazel/platform_aliases.bzl", "hexagon_platform_alias")
load("@rules_cc//cc:cc_binary.bzl", "cc_binary")
load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_pkg//pkg:pkg.bzl", "pkg_zip")

filegroup = native.filegroup

def hexagon_rpc_app(
        *,
        name,
        interface_idl,
        interface_name = None,
        host_srcs,
        dsp_srcs,
        host_deps = None,
        dsp_deps = None,
        visibility = None,
        package_dir = "ship"):
    """Builds a QAIC interface, ARM host binary, and Hexagon DSP skeleton.

    Args:
      name: Base name for the targets. Creates:
        - :<name>                (ARM/aarch64 Android host binary)
        - :<name>_skel           (Hexagon DSP shared object)
        - :<name>_skel_cross     (alias that always builds with //platform:hexagon)
        - :<name>_ship           (zip containing host + DSP outputs under `package_dir`)
      interface_idl: Path to the .idl file.
      interface_name: Optional override for the QAIC interface name. Defaults to
        `name`.
      host_srcs: List of ARM source files for the host binary.
      dsp_srcs: List of DSP source files for the skeleton .so.
      host_deps: Additional deps for the host binary (defaults include QAIC stub
        library and Hexagon SDK helpers).
      dsp_deps: Additional deps for the DSP binary (defaults include QAIC skel
        library).
      visibility: Optional visibility list applied to generated targets.
      package_dir: Directory prefix inside the generated zip (default: "ship").
    """
    interface_name = interface_name or name
    host_deps = host_deps or []
    dsp_deps = dsp_deps or []
    host_target_compatible_with = [
        "@platforms//cpu:aarch64",
        "@platforms//os:android",
    ]
    dsp_target_compatible_with = [
        "//constraints:cpu_hexagon",
        "//constraints:os_qurt",
    ]
    gen_dir = name + "_gen"
    iface_target = name + "_iface"

    hexagon_interface(
        name = iface_target,
        input_idl = interface_idl,
        interface_name = interface_name,
        output_dir = gen_dir,
        visibility = visibility,
    )

    filegroup(
        name = iface_target + "_header",
        srcs = [iface_target],
        output_group = "header",
        visibility = visibility,
    )

    filegroup(
        name = iface_target + "_stub",
        srcs = [iface_target],
        output_group = "stub",
        visibility = visibility,
    )

    filegroup(
        name = iface_target + "_skel",
        srcs = [iface_target],
        output_group = "skel",
        visibility = visibility,
    )

    cc_library(
        name = name + "_host_interface",
        srcs = [iface_target + "_stub"],
        hdrs = [iface_target + "_header"],
        includes = [gen_dir],
        deps = ["@hexagon_sdk//:cdsprpc_android_aarch64"],
        target_compatible_with = host_target_compatible_with,
        visibility = visibility,
    )

    cc_library(
        name = name + "_dsp_interface",
        srcs = [iface_target + "_skel"],
        hdrs = [iface_target + "_header"],
        includes = [gen_dir],
        alwayslink = True,
        target_compatible_with = dsp_target_compatible_with,
        visibility = visibility,
    )

    cc_binary(
        name = name,
        srcs = host_srcs,
        copts = ["-std=gnu99"],
        deps = ["@hexagon_sdk//:cdsprpc_android_aarch64", name + "_host_interface"] + host_deps,
        target_compatible_with = host_target_compatible_with,
        visibility = visibility,
    )

    dsp_binary_name = name + "_skel"
    cc_binary(
        name = dsp_binary_name,
        srcs = dsp_srcs,
        copts = ["-std=gnu99"],
        linkopts = ["-Wl,-soname=lib%s_skel.so" % interface_name],
        linkshared = True,
        target_compatible_with = dsp_target_compatible_with,
        deps = [name + "_dsp_interface"] + dsp_deps,
        visibility = visibility,
    )

    hexagon_platform_alias(
        name = dsp_binary_name + "_cross",
        actual = dsp_binary_name,
    )

    filegroup(
        name = name + "_ship_files",
        srcs = [
            name,
            dsp_binary_name + "_cross",
        ],
        visibility = visibility,
    )

    pkg_zip(
        name = name + "_ship",
        srcs = [name + "_ship_files"],
        package_dir = package_dir,
        visibility = visibility,
    )

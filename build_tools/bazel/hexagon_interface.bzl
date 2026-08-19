"""
This file provides a rule to generate RPC interfaces to the Hexagon DSP.
"""

load("@bazel_skylib//lib:paths.bzl", "paths")

def _hexagon_interface_impl(ctx):
    """Generate stub and skel interface files from an interface definition.

    Args:
        ctx: The context.
            .input_idl (str): Path of the interface definition file (*.idl).
            .interface_name (str): Name of the interface in the *.idl file.
            .output_dir (str): Path of the output directory.
    """
    inc_flags = []
    for inc_file in ctx.files._incs_tree:
        inc_flag = "-I" + paths.dirname(inc_file.path)
        if inc_flag not in inc_flags:
            inc_flags.append(inc_flag)
    output_name_base = paths.join(ctx.attr.output_dir, ctx.attr.interface_name)
    header = ctx.actions.declare_file(output_name_base + ".h")
    stub = ctx.actions.declare_file(output_name_base + "_stub.c")
    skel = ctx.actions.declare_file(output_name_base + "_skel.c")
    outputs = [header, skel, stub]
    ctx.actions.run(
        executable = ctx.file._qaic,
        arguments = ["-mdll", "-o", header.dirname] + inc_flags + [ctx.file.input_idl.path],
        inputs = [ctx.file.input_idl],
        outputs = outputs,
        mnemonic = "qaic",
        progress_message = "Generating interface from %s" % ctx.file.input_idl.short_path,
        tools = [ctx.file._qaic] + ctx.files._incs_tree,
    )

    return [DefaultInfo(files = depset(outputs)), OutputGroupInfo(header = depset([header]), stub = depset([stub]), skel = depset([skel]))]

hexagon_interface = rule(
    implementation = _hexagon_interface_impl,
    attrs = {
        "_qaic": attr.label(default = Label("@hexagon_sdk//:qaic"), allow_single_file = True),
        "_incs_tree": attr.label(default = Label("@hexagon_sdk//:incs_tree"), allow_files = True),
        "input_idl": attr.label(allow_single_file = True, mandatory = True),
        "interface_name": attr.string(mandatory = True),
        "output_dir": attr.string(mandatory = True),
    },
)

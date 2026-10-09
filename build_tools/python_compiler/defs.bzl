"""Build matching native bindings and assemble a relocatable compiler package."""

load("@iree//build_tools/bazel:renamed_link.bzl", "renamed_cc_info")
load("@rules_cc//cc:cc_binary.bzl", "cc_binary")
load("@rules_cc//cc:cc_library.bzl", "cc_library")

PYTHON_COPTS = ["-fexceptions", "-frtti", "-DMLIR_PYTHON_PACKAGE_PREFIX=iree.compiler."]
PYTHON_FEATURES = ["-use_header_modules"]

# Upstream filegroups include optional wrappers outside the dialect subset in
# IREE's compiler/bindings/python/CMakeLists.txt. Their IRDL/Debug/Tune dependency
# closure is not packaged here; do not ship wrappers that cannot be imported.
_EXCLUDED_WRAPPERS = [
    "iree/compiler/dialects/ext.py",
    "iree/compiler/dialects/transform/debug.py",
    "iree/compiler/dialects/transform/tune.py",
]

def mlir_python_extension(name, src, deps = []):
    """Keep binding objects, resolving their MLIR implementation from IREE."""
    cc_library(
        name = name + "_objects",
        srcs = [src],
        copts = PYTHON_COPTS,
        features = PYTHON_FEATURES,
        deps = [
            "@llvm-project//mlir:MLIRBindingsPythonNanobindHeaders",
            "@llvm-project//mlir:MLIRBindingsPythonLibHeaders",
        ] + deps,
    )
    renamed_cc_info(
        name = name + "_renamed",
        deps = [":" + name + "_objects"],
        compiler = "@iree//lib:IREECompilerShared",
        force_alwayslink = True,
    )
    cc_binary(
        name = name + ".so",
        srcs = [],
        linkshared = True,
        linkstatic = True,
        linkopts = ["-Wl,-rpath,$ORIGIN"],
        deps = [":" + name + "_renamed", "@iree//lib:IREECompilerShared"] + ([] if name == "_mlir" else [":core_shared"]),
    )

def _destination(file):
    # short_path uses ../<canonical repository>/ for external files. Routing
    # on source roots is independent of Bazel's canonical repository spelling.
    path = file.short_path
    for prefix, replacement in [
        ("compiler/bindings/python/iree/", "iree/"),
        ("mlir/python/mlir/", "iree/compiler/"),
        ("python/torch_mlir/extras/", "iree/compiler/extras/"),
    ]:
        marker = "/" + prefix
        if marker in path:
            destination = replacement + path.split(marker, 1)[1]
            return None if destination in _EXCLUDED_WRAPPERS else destination
    return "iree/compiler/_mlir_libs/" + file.basename

def _package_impl(ctx):
    out = ctx.actions.declare_directory(ctx.label.name)
    manifest = ctx.actions.declare_file(ctx.label.name + ".inputs.json")
    files = depset(transitive = [src[DefaultInfo].files for src in ctx.attr.srcs])
    ctx.actions.write(manifest, json.encode([
        {"source": file.path, "destination": _destination(file)}
        for file in files.to_list()
        if _destination(file) != None
    ]))
    ctx.actions.run(
        executable = ctx.executable._assembler,
        arguments = ["--manifest", manifest.path, "--metadata", ctx.file.metadata.path, "--output", out.path],
        inputs = depset([manifest, ctx.file.metadata], transitive = [files]),
        outputs = [out],
        mnemonic = "CompilerPythonPackage",
        progress_message = "Assembling source-built iree.compiler and content provenance",
    )
    return [DefaultInfo(files = depset([out]))]

compiler_python_package = rule(
    implementation = _package_impl,
    attrs = {
        "srcs": attr.label_list(allow_files = True),
        "metadata": attr.label(allow_single_file = True, mandatory = True),
        "_assembler": attr.label(default = "//build_tools/python_compiler:assemble", executable = True, cfg = "exec"),
    },
)

# Compiler Python package

This package keeps Turbine's FX importer and the native compiler aligned with
this checkout's IREE, LLVM and torch-mlir sources and patches.

Create or refresh the editable test environment:

```sh
build_tools/create_test_venv.sh
```

Python sources are copied into Bazel outputs. Rerun the helper after compiler or
importer changes; rebuilding only `iree-compile` does not refresh these bindings.
The helper checks the installed package's origin and native module imports.

Build an installable Python 3.11 Linux wheel:

```sh
bazel build //build_tools/python_compiler:wheel
build_tools/create_test_venv.sh --compiler-wheel <wheel-path>
```

Assembly adapts upstream IREE/MLIR Bazel targets to the `iree.compiler` namespace,
following the package-tree approach used in roof-mlir. Wheel creation uses
[rules_python's `py_wheel`](https://rules-python.readthedocs.io/en/latest/api/rules_python/python/packaging.html),
with no custom wheel writer. IREE's own Python packaging uses CMake/setuptools.
Native bindings share the packaged compiler library; optional AMDGPU/SMT Python
bindings are omitted.

The distribution is `iree-base-compiler==3.12.0+source`. Import manifests record
its precomputed content digest. External dependencies stay locked, including the
host Python runtime used by Turbine. The dependency lock also serves Bazel;
the venv helper replaces only its compiler wheel entry.

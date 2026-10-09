# torch-mlir patches

The checkout is `third-party/torch-mlir`. Bazel reads this working tree through
`torch-mlir-raw`; CMake uses the linked checkout under IREE. Apply these patches
with `build_tools/apply_submodule_patches.sh` before building.

The compiler Python package includes this checkout's FX importer and its helpers
in the `iree.compiler` namespace. Rebuild the package after importer edits.

| Patch | Category | Reason |
| --- | --- | --- |
| [python_importer_exports.patch](python_importer_exports.patch) | Bazel Python packaging | Export the FX importer, its `fx_as_strided.py` helper and the ONNX importer so the source-built `iree-base-compiler` package can declare them as build inputs. |
| [scatter_normalize_negative_dim.patch](scatter_normalize_negative_dim.patch) | Crash fix (ROO-2038) | Normalize negative dimensions in the scatter, scatter_add and scatter_reduce lowerings before indexing per-dimension arrays. |

The scatter patch validates the normalized dimension and includes regression
tests in `test/Conversion/TorchToTMTensor/basic.mlir`. It allows the Gemma-4
model to use its original `scatter(-1, ...)` call.

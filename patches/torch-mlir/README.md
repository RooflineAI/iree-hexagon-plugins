# torch-mlir patches

These patches fix crashes in torch-mlir's Torch-to-Linalg and Torch-to-TMTensor
conversions. `iree-compile` runs those conversions when it imports a torch
model (`--iree-input-type=torch`), so the crashes stop our models from
compiling at all. Both fixes are small and self-contained. Neither is in
upstream torch-mlir yet; each section below says what upstreaming would take.

The checkout is `third-party/torch-mlir`, the iree-org fork pinned by IREE. It
is upstream llvm/torch-mlir 528d3dcb (2026-10-02) plus three integration
commits. Bazel reads it through `torch-mlir-raw` in `MODULE.bazel`. The CMake
build reads it through the `third-party/iree/third_party/torch-mlir` symlink
that `build_tools/update_submodules.py` creates. Both builds therefore see the
patched working tree that `build_tools/apply_submodule_patches.sh` produces.

| Patch | Category | Reason |
| --- | --- | --- |
| [unflatten_static_input_dynamic_result.patch](unflatten_static_input_dynamic_result.patch) | Crash fix (ROO-2037) | When lowering `aten.unflatten.int`, expand a static input dim into static result dims and cast to the (partly dynamic) result type. Expanding it straight into a dynamic result dim is invalid `tensor.expand_shape`, and the shape-inferring builder segfaults on it. |
| [scatter_normalize_negative_dim.patch](scatter_normalize_negative_dim.patch) | Crash fix (ROO-2038) | Normalize a negative `dim` (e.g. `-1`) in the `aten.scatter.src`, `aten.scatter_add` and `aten.scatter_reduce.two` lowerings. Unnormalized, it is used as an array index and corrupts memory. |
| [python_importer_exports.patch](python_importer_exports.patch) | Bazel Python packaging | Export the FX importer, its `fx_as_strided.py` helper and the ONNX importer so the source-built `iree-base-compiler` package can declare them as build inputs. |

## unflatten_static_input_dynamic_result.patch

**Bug.** Converting an `aten.unflatten.int` whose input dim is static but whose
result type has a dynamic dim crashes with a segfault:

```mlir
%0 = torch.aten.unflatten.int %arg0, %int0, %sizes
    : !torch.vtensor<[52],f32>, !torch.int, !torch.list<int>
    -> !torch.vtensor<[1,1,?,1],f32>
```

The stack is `arith::DivSIOp` fold `<- inferExpandShapeOutputShape <-
tensor::ExpandShapeOp::build <- ConvertAtenUnflattenIntOp`. Torch's shape
inference produces such IR when the sizes come from a computation, e.g. a
`view(-1, ...)` after an index gather. It reached our Qwen3-ASR and Qwen3-VL
vision-encoder models through torch-mlir's old `aten.as_strided`
decomposition (see "Why the models no longer hit it" below).

**Root cause.** `ConvertAtenUnflattenIntOp`
(`lib/Conversion/TorchToLinalg/DataMovement.cpp`) builds the
`tensor.expand_shape` with the overload that infers the output shape. That
overload calls `mlir::inferExpandShapeOutputShape`
(`mlir/lib/Dialect/Arith/Utils/Utils.cpp`). For every reassociation group that
contains a dynamic result dim, it does
`cast<Value>(inputShape[inputIndex])`, "under the assumption that we're not
casting dynamism". When the corresponding input dim is static,
`getMixedSizes` returns an Attribute there, not a Value. In a release build the
cast is unchecked and yields a garbage Value, which the `DivSIOp` folder then
dereferences.

The same happens for a dynamic result dim outside the unflattened group, as
in `[4,12] -> [?,3,?]`. That case crashes with a corrupted stack.

The deeper problem is the result type itself. `tensor.expand_shape` requires a
static input dim to expand into static result dims only: its verifier rejects
`tensor<52xf32> into tensor<1x1x?x1xf32>` ("expected dimension 0 of collapsed
type to be dynamic"). So passing an explicit output shape is not enough. A
first version of this patch did that; it traded the crash for a verifier
error.

**Fix.** Before building the op, compute a refined result type. Each dynamic
result dim whose input dim is static gets its static size:

- outside the unflattened group, the size of the corresponding input dim;
- inside the group (at most one dynamic dim there on this path), the input
  dim divided by the product of the group's other sizes. A size that does not
  divide is a match failure instead of a crash.

The `tensor.expand_shape` produces this refined type, through the same
inferring builder as before. Its assumption, that every dynamic result dim
comes from a dynamic input dim, now holds. A `tensor.cast` then converts to
the converted torch result type, which keeps its `?`:

```mlir
%e = tensor.expand_shape %in [[0, 1, 2, 3]] output_shape [1, 1, 52, 1]
    : tensor<52xf32> into tensor<1x1x52x1xf32>
%c = tensor.cast %e : tensor<1x1x52x1xf32> to tensor<1x1x?x1xf32>
```

This mirrors the existing code a few lines above it, which casts the *input*
to a static type when the output is all static. Unchanged cases produce no
cast. The path with two or more dynamic dims in the group (`tensor.reshape`)
is untouched.

**Tests.** Two cases in `test/Conversion/TorchToLinalg/datamovement.mlir`:
the in-group case above and the outer-dim case `[4,12] -> [?,3,?]`. Both
crash the unpatched compiler (checked with the iree-base-compiler 3.12.0
`iree-opt`). With the patch, all of `test/Conversion/TorchToLinalg` and
`test/Conversion/TorchToTMTensor` pass under the Bazel-built `iree-opt`,
except `sparse.mlir`, which needs the `sparse_tensor` dialect that `iree-opt`
does not register.

**Alternatives considered.**

- [llvm/torch-mlir#4565](https://github.com/llvm/torch-mlir/pull/4565)
  ("Avoid inferred expand_shape in unflatten lowering", open since 2026-05-08,
  no reviews, no tests) passes an explicit output shape. By reading, it keeps
  the dynamic result type, so it would likely hit the same verifier error as
  this patch's first version; that is not verified. It is also broader: it
  *requires* the sizes to come from a `prim.ListConstruct` for any dynamic dim
  (it was only required for two or more), and derives the size from the
  literal `-1` operand. Its null check for a non-list `sizes` operand is a
  real but separate latent bug, which this patch does not touch. Its
  description attributes the crash to a use-after-free; the cause above is
  different and reproducible without one.
- Fixing MLIR instead: `inferExpandShapeOutputShape` could use
  `getValueOrCreateConstantIndexOp` instead of `cast<Value>`. That would turn
  the segfault into the verifier error above, not into a working lowering.
  The invalid result type comes from torch-mlir, so the fix belongs there.

**Why the models no longer hit it.** torch-mlir
[#4585](https://github.com/llvm/torch-mlir/pull/4585) (in our pin since the
2026-10 IREE bump) deleted the Torch-IR `aten.as_strided` decomposition. That
decomposition was the producer of the crashing `unflatten`. The FX importer
in iree-base-compiler 3.12.0 now rewrites `as_strided` to `aten.index.Tensor`
before import. So this patch guards against a known crash with no current
producer in our model suite. It is cheap insurance and a clean upstream fix.

**Upstream.** torch-mlir main still has the unpatched code. The patch plus
its tests should be suitable for a PR to llvm/torch-mlir as-is.

## scatter_normalize_negative_dim.patch

**Bug.** `torch.aten.scatter.src` with `dim = -1` segfaults:

```mlir
%0 = torch.aten.scatter.src %self, %int-1, %index, %src
    : !torch.vtensor<[2,8],f32>, !torch.int, !torch.vtensor<[2,3],si64>,
      !torch.vtensor<[2,3],f32> -> !torch.vtensor<[2,8],f32>
```

The stack is `SmallVector::growAndPushBack <- linalg::GenericOp::create <-
convertTorchScatterIndexAndSrcToTMScatterIndexAndSrc <- ConvertAtenScatterOp`.
`scatter_add` aborts and `scatter_reduce.two` segfaults the same way. Writing
`tensor.scatter(-1, ...)` is idiomatic PyTorch; our Gemma-4 drafter does it
in `_masked_embedder_forward`.

**Root cause.** `ConvertAtenScatterOp` and `ConvertAtenScatterReduceTwoOp`
(`lib/Conversion/TorchToTMTensor/TorchToTMTensor.cpp`) pass the constant `dim`
straight to `convertTorchScatterIndexAndSrcToTMScatterIndexAndSrc`. Inside
its `linalg.generic` body builder, that function writes `yieldVals[dim]`. With
`dim = -1` the write lands just before the vector's elements, over its
size/capacity header. The next `push_back` then reallocates from garbage. The
other `dim` users in the same file already call `toPositiveDim`; these two
patterns were missed.

**Fix.** Normalize with `toPositiveDim(dim, rank)` and reject an
out-of-range dim with `isValidDim` in both patterns. The lit tests in
`test/Conversion/TorchToTMTensor/basic.mlir` check that `dim = -1` produces
exactly the lowering of the equivalent positive dim, for `scatter.src`,
`scatter_add` and `scatter_reduce.two`. The CHECK lines were validated
against the unpatched lowering of the positive dim, and they reject the
lowering of the wrong dim (0).

**Upstream.** Unfixed on torch-mlir main. No issue or PR found. This is a
small, obviously-correct PR for llvm/torch-mlir.

## Not patched: boolean-mask indexing (ROO-2039)

`hidden[mask, :] += feats` with a data-dependent boolean mask fails before
`iree-compile` is ever run. It fails in torch-mlir's **FX importer**
(`torch_mlir/extras/fx_importer.py`, packaged as
`iree.compiler.extras.fx_importer`) with
`Unimplemented call_function: target=<built-in function ge>`.

The cause: the mask's unbacked size gets pinned to a constant (e.g. by a
broadcast). The size assertions torch.export emits then have a concrete
`True` value instead of a symbolic one, and the importer only accepts
symbolic builtins. The source-built compiler Python package now includes this
checkout's FX importer, so a local importer patch would reach integration tests
after rebuilding the package. No importer fix is included here. A second bug
sits behind it: TorchToLinalg emits invalid `arith.cmpi` for a boolean-mask
`aten.index.Tensor`; only `index_put` is decomposed through `nonzero`. The
Qwen3-VL model works around both with a static gather, which also avoids
dynamic shapes on Hexagon. See
`integration_tests/models/Qwen/Qwen3-VL-2B-Instruct/model.py`.

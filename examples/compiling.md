# Compiling an example for Hexagon

No Hexagon hardware is required for anything on this page.

## 1. Build the compiler

As a first step, build the iree-compile executable:

```sh
bazel build @iree//tools:iree-compile
```

The bazel output points to the compiled binary. It can also be resolved through:

```sh
IREE_COMPILE="$PWD/$(bazel cquery --output=files @iree//tools:iree-compile 2>/dev/null | head -n1)"
"$IREE_COMPILE" --version
```

## 2. Choose an example

Every command below is written against these variables, so pick an example once here
and the rest of the page applies unchanged:

```sh
EXAMPLE=matmul            # or: EXAMPLE=attention_layer

MODEL="$PWD/examples/$EXAMPLE/$EXAMPLE.mlir"
OUT="$PWD/examples/$EXAMPLE/dump"
```

Both examples live at `examples/<name>/<name>.mlir` and compile with identical flags,
so switching between them only means re-running the block above.

## 3. Compile for Hexagon

```sh
mkdir -p "$OUT"

"$IREE_COMPILE" "$MODEL" \
  -o "$OUT/$EXAMPLE.vmfb" \
  --iree-hal-target-device=hexagon \
  --iree-input-type=auto \
  --iree-hexagon-v=79 \
  --iree-hexagon-features=+hvxv79,+hvx-length128b \
  --iree-opt-data-tiling=false \
  --iree-stream-resource-min-offset-alignment=128 \
  --iree-hexagon-launch-config-selector=hexagon
```

You should get `$OUT/$EXAMPLE.vmfb`. That file contains the Hexagon shared object for
the dispatch plus the VM module that drives it.

### What the flags do

| Flag | Why |
|---|---|
| `--iree-hal-target-device=hexagon` | Selects the Hexagon target registered by the compiler plugin. |
| `--iree-hexagon-v=79` | Hexagon architecture version. |
| `--iree-hexagon-features=+hvxv79,+hvx-length128b` | Enables HVX with a 128-byte vector length. |
| `--iree-opt-data-tiling=false` | IREE's encoding/data-tiling path is not yet supported in Hexagon; leave it off. |
| `--iree-stream-resource-min-offset-alignment=128` | Matches the DSP's buffer alignment requirement. |
| `--iree-hexagon-launch-config-selector=hexagon` | Uses the Hexagon lowering-strategy selector. Defaults to `llvmcpu`, which routes through the a generic upstream LLVMCPU selector instead. |

## 4. Read the compilation log

Nothing above prints the lowering. To capture it, add the dump flags and redirect —
note `--mlir-disable-threading`, without which the interleaved output from parallel
compilation is unreadable:

```sh
mkdir -p "$OUT/compilation_phases"

"$IREE_COMPILE" "$MODEL" \
  -o "$OUT/$EXAMPLE.vmfb" \
  --iree-hal-target-device=hexagon \
  --iree-input-type=auto \
  --iree-hexagon-v=79 \
  --iree-hexagon-features=+hvxv79,+hvx-length128b \
  --iree-opt-data-tiling=false \
  --iree-stream-resource-min-offset-alignment=128 \
  --iree-hexagon-launch-config-selector=hexagon \
  --mlir-disable-threading \
  --mlir-print-ir-after-all \
  --mlir-elide-elementsattrs-if-larger=16 \
  --mlir-elide-resource-strings-if-larger=16 \
  --dump-compilation-phases-to="$OUT/compilation_phases" \
  --iree-hal-dump-executable-intermediates-to="$OUT" \
  --iree-hal-dump-executable-binaries-to="$OUT" \
  > "$OUT/log.mlir" 2>&1
```

This produces:

| Path | Contents |
|---|---|
| `$OUT/log.mlir` | The full IR after every pass. Note that this file may become very large. |
| `$OUT/compilation_phases/` | One file per top-level phase (input, ABI, flow, stream, HAL, VM). |
| `$OUT/*.bc`, `*.ll` | LLVM bitcode / IR for the dispatch, from `--iree-hal-dump-executable-intermediates-to`. |
| `$OUT/*.s` | Hexagon assembly for the generated kernel. Requires `--iree-hal-dump-executable-binaries-to` — the intermediates flag alone does **not** produce it. |
| `$OUT/*.so` | The linked Hexagon ELF, same flag as the assembly. |
| `$OUT/$EXAMPLE.vmfb` | The final artifact. |

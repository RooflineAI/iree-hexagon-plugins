# Hexagon integration tests

End-to-end tests that compile a model with this repo's `iree-compile` and run it
on a physical Hexagon DSP over ADB, checking the result against a reference
produced by running the same model eagerly in torch.

A model is **two files and no test code**:

```
models/<org>/<name>/model.yaml   what to run, how close it has to be, known failures
models/<org>/<name>/model.py     get_model() -> torch.nn.Module
```

The path under `models/` **is** the model's name - what `--model` takes and what
the test id shows. The hub id and its pinned revision live in `model.py`, which
is what fetches the weights; `model.yaml` never repeats them.

## Running them

The device suite runs under **plain pytest**, not `bazel test`. The compiler binary,
runtime package and the device-tools zip are expected to be built separately and
passed as an argument to the pytest command.

```sh
build_tools/create_test_venv.sh                 # creates ./.venv from requirements.txt
.venv/bin/pytest integration_tests/ -v -rA \
    --iree-compile=<path> --runtime-zip=<path> --device-tools-zip=<path> --lld=<path>
```

Omit the four paths and they will try to be resolved with `bazel cquery`:

```sh
bazel build @iree//tools:iree-compile \
            //plugins/runtime/hexagon:hexagon_runtime_aarch64_android \
            //integration_tests/device/tools:device_tools_aarch64_android \
            @llvm-project//lld:ld.lld
pytest integration_tests/ -v -rA --model=microsoft/resnet-50
```

Options, all in `conftest.py`: `--model` and `--compile-case` (both
repeatable), `--keep-device-dir`, `--device-root`, and the four artifact paths.

## Requirements

* An Android phone with a Hexagon DSP, in `adb devices` state `device`. **The
  adb server is assumed to be up already** - nothing here manages keys, servers
  or connections.
* With no usable phone the device tests **fail**.
* Network, on first run: models come from the HuggingFace hub. Set `HF_HOME` to
  a persistent directory so weights are fetched once.

## Layout

| | |
|---|---|
| `models/` | **data only**: `<org>/<name>/{model.yaml, model.py}` |
| `modeltree/` | code *about* that data - `spec.py` (schema), `discover.py` (finding and selecting), `outcomes.py` (known failures) |
| `stages/` | the pipeline, in order - `importing.py`, `compiling.py`, `running.py`, `checking.py`, plus `artifacts.py` |
| `device/` | the only package that knows a phone exists - `adb.py`, `deploy.py`, and `tools/` (the cross-compiled helper) |
| `tool_paths.py` | resolve an artifact path, or ask Bazel |
| `conftest.py` | options and fixtures, the only pytest glue |
| `test_models.py` | the device test, parametrized over the tree |

## On the device

```
/data/local/tmp/hexagon_it/runs/<run-id>/
    bin/    iree-run-module, iree-benchmark-module, limit_lifetime
    lib/    the DSP skel and libc++_shared
    cases/  one directory per (model, compile case)
```

**There is no device lock.** This suite assumes one runner executing serially.

## Adding a model

`models/HuggingFaceTB/SmolLM2-135M/model.yaml`:

```yaml
dtype: float16
inputs:
  - {shape: [1, 128], dtype: int64, generator: uniform, low: 0, high: null, seed: 0}
  - {shape: [1, 128], dtype: int64, generator: ones}
tolerances: {atol: 2.0e-1, rtol: 5.0e-2, relative_l2_percent: 0.2}
```

beside a `model.py` that pins what it fetches:

```python
_MODEL_ID = "HuggingFaceTB/SmolLM2-135M"
_REVISION = "93efa2f097d58c2a74874c7e644dbc9b0cee75a2"
```

With no `compile_cases`, a model runs both `hexagon` and
`hexagon-vtcm-hmx`. Add an explicit list to narrow or extend that matrix. The
`llvmcpu-baseline` case is opt-in; for example:

* `hexagon`: Hexagon launch configuration, with VTCM tiling and HMX disabled.

* `hexagon-vtcm-hmx`: Hexagon launch configuration, with both features enabled.

* `llvmcpu-baseline`: LLVMCPU launch configuration, with both features disabled.

```yaml
compile_cases: [hexagon, hexagon-vtcm-hmx, llvmcpu-baseline]
```

Use `--compile-case=hexagon-vtcm-hmx` to filter a test invocation to one of a
model's configured cases.

For a classifier on a real photograph, `generator: image` plus a
`semantic_check`:

```yaml
inputs:
  - {shape: [1, 3, 224, 224], dtype: float32, generator: image, image: keyboard.png}
semantic_check:
  expected_label: "computer keyboard, keypad"
```

## Known failures

A model may record known failures:

```yaml
expected_outcomes:
  - case: llvmcpu-baseline   # or "*" for every case
    status: ACCURACY_FAILURE # or COMPILE_FAILURE, RUNTIME_FAILURE
    reason: "relative L2 error 2.9"
    comment: "generic LLVMCPU launch-config selector; measured 2026-08-27"
```

`reason` is a substring that **must appear in the log the failure produces**.

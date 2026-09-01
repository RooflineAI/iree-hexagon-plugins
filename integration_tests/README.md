# Hexagon integration tests

End-to-end tests that compile a model with this repo's `iree-compile` and run it
on a physical Hexagon DSP over ADB, checking the result against a reference
produced by running the same model eagerly in torch.

A model is **two files and no test code**:

```
models/<org>/<name>/model.yaml   what to run, how close it has to be, known failures
models/<org>/<name>/model.py     get_model() -> torch.nn.Module
```

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

Options, all in `conftest.py`: `--model` (repeatable), `--include-tag` /
`--exclude-tag`, `--keep-device-dir`, `--device-root`, and the four artifact
paths.

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
| `modeltree/` | code *about* that data - `spec.py` (schema), `discover.py` (tags), `outcomes.py` (known failures) |
| `stages/` | the pipeline, in order - `importing.py`, `compiling.py`, `running.py`, `checking.py`, plus `artifacts.py` |
| `device/` | the only package that knows a phone exists - `adb.py`, `deploy.py`, and `tools/` (the cross-compiled helper) |
| `tool_paths.py` | resolve an artifact path, or ask Bazel |
| `conftest.py` | options and fixtures, the only pytest glue |
| `test_models.py` | the device test, parametrized over the tree |
| `test_model_tree.py` | the checks that need no phone |

## On the device

```
/data/local/tmp/hexagon_it/runs/<run-id>/
    bin/    iree-run-module, iree-benchmark-module, limit_lifetime
    lib/    the DSP skel and libc++_shared
    params/ parameter archives, one directory per model
    cases/  one directory per (model, compile case)
```

**There is no device lock.** This suite assumes one runner executing serially.

## Adding a model

```yaml
model: HuggingFaceTB/SmolLM2-135M
revision: 93efa2f097d58c2a74874c7e644dbc9b0cee75a2   # pin it
dtype: float16
inputs:
  - {shape: [1, 128], dtype: int64, generator: uniform, low: 0, high: null, seed: 0}
  - {shape: [1, 128], dtype: int64, generator: ones}
tolerances: {atol: 2.0e-1, rtol: 5.0e-2, relative_l2_percent: 0.2}
tags: [hexagon, llm]
```

Add `compile_cases: [new-tiling-heuristics]` to run only some of the three
configurations.

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
  - case: default            # or "*" for every case
    status: ACCURACY_FAILURE # or COMPILE_FAILURE, RUNTIME_FAILURE
    reason: "relative L2 error 2.9"
    comment: "generic LLVMCPU launch-config selector; measured 2026-08-27"
```

`reason` is a substring that **must appear in the log the failure produces**.

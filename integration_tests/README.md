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
both runtime packages (plain and with Tracy) and the device-tools zip are
expected to be built separately and passed as an argument to the pytest command.

```sh
build_tools/create_test_venv.sh                 # creates ./.venv from requirements.txt
.venv/bin/pytest integration_tests/ -v -rA \
    --iree-compile=<path> --runtime-zip=<path> --runtime-zip-tracy=<path> \
    --device-tools-zip=<path> --lld=<path>
```

Omit the paths and they will try to be resolved with `bazel cquery`:

```sh
bazel build @iree//tools:iree-compile \
            //plugins/runtime/hexagon:hexagon_runtime_aarch64_android \
            //plugins/runtime/hexagon:hexagon_runtime_aarch64_android_tracy \
            //integration_tests/device/tools:device_tools_aarch64_android \
            @llvm-project//lld:ld.lld
pytest integration_tests/ -v -rA --model=microsoft/resnet-50
```

Both runtimes are deployed once per session, each in its own run directory; the
profile run cases (see [Profiling](#profiling)) use the Tracy one, all others
the plain one.

Options, all in `conftest.py`: `--model`, `--compile-case` and `--run-case`
(all repeatable), `--run-skipped`, `--keep-device-dir`, `--device-root`,
`--rsync`, `--tracy-port` (default 8086), and the five artifact paths.

## Requirements

* An Android phone with a Hexagon DSP, in `adb devices` state `device`. **The
  adb server is assumed to be up already** - nothing here manages keys, servers
  or connections.
* With no usable phone the device tests **fail**.
* Network, on first run: models come from the HuggingFace hub. Set `HF_HOME` to
  a persistent directory so weights are fetched once.
* `rsync` 3.2 or newer on the host, with zstd and xxHash support (as in
  Ubuntu's package), on PATH or given with `--rsync`. Large modules go to the
  phone as rsync deltas; the phone's end ships in the device-tools zip, which is a
  separate CI artifact and not part of the release.

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
/data/local/tmp/hexagon_it/
    runs/<run-id>/          removed when the run ends
        bin/    iree-run-module, iree-benchmark-module, iree-run-llm,
                limit_lifetime, rsync
        lib/    the DSP skel and libc++_shared
        cases/  one directory per (model, compile case)
    modules/<case>/module.vmfb   kept across runs
```

Modules of 64 MiB and more are kept in `modules/` and sent with rsync, as a
delta against the module the same (model, compile case) sent last time. A
recompile usually leaves the weights - nearly all of a module - byte-identical,
so it sends little more than the changed code. The first transfer of a case,
with nothing in `modules/` yet, uses adb push, which is faster for sending a
whole file. Each case's `module.vmfb` is then a copy of its entry in
`modules/`, and the test fails unless that copy's sha256 matches the module
built on the host - the sign that another execution sharing the device root
replaced it. Deleting `modules/` is always safe; the next run of each case then
sends the whole module once.

Smaller modules go straight into the case directory with adb push: for them a
delta saves less than rsync's own round trips to the phone cost.

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
`hexagon-vtcm-hmx`. Add an explicit list to narrow that matrix:

* `hexagon`: Hexagon launch configuration, with VTCM tiling and HMX disabled.

* `hexagon-vtcm-hmx`: Hexagon launch configuration, with both features enabled.

```yaml
compile_cases: [hexagon, hexagon-vtcm-hmx]
```

Use `--compile-case=hexagon-vtcm-hmx` to filter a test invocation to one of a
model's configured cases.

Each compile case is run on the device once per run case, a runtime
configuration passed as flags to `iree-benchmark-module`. With no `run_cases`, a model
runs only `parallel`:

* `parallel`: the runtime default, executing **up to** one dispatch per HVX context
  at the same time.

* `sequential`: `--hexagon_max_concurrent_dispatches=1`, one dispatch at a time.

* `parallel-profile`, `sequential-profile`: the same, run with the Tracy runtime
  and profiled, see [Profiling](#profiling).

* `parallel-debug-malloc`, `sequential-debug-malloc`: the same, run under
  bionic's [malloc debug](https://android.googlesource.com/platform/bionic/+/main/libc/malloc_debug/README.md)
  (`verify_pointers abort_on_error`), so a free of an invalid pointer aborts
  the run. Only for smaller models: malloc debug fails every single allocation
  of 2 GiB or more, which crashes iree-benchmark-module on a module that large.
  `parallel-profile-debug-malloc` and `sequential-profile-debug-malloc` are
  the profile run cases under malloc debug; it applies to iree-benchmark-module only,
  not to the Tracy tools.

```yaml
run_cases: [parallel, sequential]
```

`--run-case=sequential` filters a test invocation the same way.
`expected_outcomes` entries are matched against the compile case only, since they shuld never differ, regardless of sequential or parallel execution

For a classifier on a real photograph, `generator: image` plus a
`semantic_check`:

```yaml
inputs:
  - {shape: [1, 3, 224, 224], dtype: float32, generator: image, image: keyboard.png}
semantic_check:
  expected_label: "computer keyboard, keypad"
```

A large model may need more than the default 600 seconds to push its module
and to run on the device:

```yaml
run_timeout: 3600
```

### Token generation: prefill and decode

By default the module has one function, `main`: the model's forward on the
spec's inputs. A `model.py` whose `get_model()` returns a `KvCacheLM`
(`models/kv_cache.py`) instead gets a causal LM exported for token generation,
with the KV cache in module state (globals): `prefill` on the prompt, and
`decode` of the next token, each with its own example arguments, dynamic
shapes and eager reference. The test compiles once and runs both in one adb
round trip, each in its own `iree-run-module`: `decode` runs only if prefill's
token matches the reference, compared on the device. The host then checks every
output. Both start from an empty cache, in the eager reference as on the
device, so `decode` checks the compiled function rather than a real
continuation.

A `generation` block adds a real continuation: after those two checks pass,
`iree-run-llm` (`tools/iree-run-llm-main.c`, on the inference engine in
`tools/llm/`) runs the same module on the device as an inference loop. It tokenizes the prompt itself, runs `prefill`,
then `decode` once per further token, and generates `max_new_tokens` tokens
greedily, ignoring stop tokens:

```yaml
generation:
  prompt: "def fibonacci(n):\n"
  max_new_tokens: 32
  max_tie_margin: 0.25  # the default
```

`model.py` then also defines `get_tokenizer_json()`, the path of the model's
`tokenizer.json`, which the device gets too. The reference is
`KvCacheLM.generate_greedy`: the same functions run eagerly, with the prompt
padded as on the device. The check, in `stages/checking.py`:

* the prompt's token ids have to match the `tokenizers` library's exactly,
  which checks the device's tokenizer;
* the generated tokens have to match up to the first that differs, where the
  reference's two most likely tokens have to be within `max_tie_margin` of each
  other in logits: a near-tie that rounding may tip either way. The tokens
  after it are not compared.

So pick a prompt whose reference has no near-tie, and any difference fails;
the import writes the margins to `generation_reference.json`. The log has
the text and, under `generation timings:`, the time of the whole
`iree-run-llm` run (`generate`) and its own times of `prefill` and of one
decode step (`decode_per_token`). They are not part of `run:`, the time
`perf_compare.py` tracks.

## Profiling

A profile run case runs the module with the Tracy runtime and
`TRACY_NO_EXIT=1`, captures it with `tracy-capture` and exports the DSP zones
with `tracy-csvexport -g`, both on the device, from the Tracy runtime package.
Only the CSV is pulled, to `device_outputs/profile.csv` in the test's work
directory, and its path is printed. Both run on the device because the Tracy
port is not reachable from the host when adb talks to a remote adb server, as
in CI, and because the trace is thousands of times the size of the CSV.
The trace is not kept: to look at a profile that fails a check, profile the
model by hand, see [below](#profiling-a-model-by-hand).

The CSV has one row per DSP zone: name, source file, start and duration in ns.
Kernel zones are named after their dispatch function, e.g.
`main$async_dispatch_1_conv_32x64x64x32x3x3_f16xf16xf32`; the other zones have
fixed names ("DSP execution", "Schedule", "Barrier", ...). The export has no
column for the DSP thread a zone ran on, so checks only see names and times.

`profile_checks` says what a model's profile has to show. Each entry names a
check from `stages/profile_checks.py`, its `args`, and optionally the profile
run case and compile case it applies to (default `"*"`, every one):

```yaml
run_cases: [parallel, sequential, parallel-profile, sequential-profile]
profile_checks:
  # the two convs run in parallel
  - check: overlap
    run_case: parallel-profile
    args: {a: dispatch_1_conv, b: dispatch_2_conv}
  # the max reads both conv results, so it must not overlap either of them
  - check: no_overlap
    args: {a: dispatch_3_reduction, b: "dispatch_[12]_conv"}
```

`a` and `b` are regular expressions searched in the kernel names. A pattern that
matches no kernel fails the check, so a renamed dispatch cannot pass silently.
The checks:

| check | args | passes if |
|---|---|---|
| `overlap` | `a`, `b` | a kernel matching `a` ran at the same time as another matching `b` |
| `no_overlap` | `a`, `b` | no kernel matching `a` ran at the same time as one matching `b` |
| `max_kernel_concurrency` | `max` | never more than `max` kernels ran at once |
| `min_kernel_concurrency` | `min` | at some point at least `min` kernels ran at once |
| `kernel_count` | `count` | exactly `count` kernels ran |

A new check is a function in `stages/profile_checks.py` decorated with
`@profile_check("<name>")`, taking the `Profile` and its args and raising
`ProfileError`. The args in `model.yaml` are validated against its signature
when the model tree is loaded.


A check that does not pass is a `PROFILE_FAILURE`; a profile that could not be
captured or exported is a `CAPTURE_PROFILE_FAILURE`.
A profile run case with no checks still produces the CSV.

`test_profile_checks.py` tests the checks on hand-written CSVs, without a device.

## Performance

CI compares the run time of every passed test against its average in the same
job of the latest completed main runs, in a step of its own after the tests:
`perf_compare.py` parses the `-rA` output of the session, the `run:` line under `timings:` of each passed test, and reads the earlier main runs' output from their GitHub logs with `gh`.
When no time can be found for any passed test, the check fails.

The time is one invocation of the function, as `iree-benchmark-module` measures
it on the device (`--benchmark_min_time=1x`), without starting the process or
loading the module. The rest is printed as `adb_setup:`, but not compared: the
host's wall clock around the adb command minus the invocation, so the adb
round trip, starting the process, opening the DSP session and loading the
module. When there is no plausible device time, the
host's is printed as `run_host:` instead, and the test is not compared.
Profile and debug-malloc run cases are not compared.

The thresholds when a warning is raised are command-line arguments.
To compare a local session:
```sh
# collect the log from stdout
.venv/bin/pytest integration_tests/ -v -rA ... | tee pytest-output.log
# compare against CI's main
python -m integration_tests.perf_compare --current pytest-output.log \
    --num-recent-runs 5 --max-runs-searched 10 \
    --warn-percent 5 --fail-percent 30 --min-delta-s 0.3 \
    --job test-bazel # or --baseline-log <file>, repeatable
```

## Known failures

A model may record known failures:

```yaml
expected_outcomes:
  - case: hexagon-vtcm-hmx   # or "*" for every case
    status: ACCURACY_FAILURE # or COMPILE_FAILURE, RUNTIME_FAILURE, PROFILE_FAILURE,
                             #    CAPTURE_PROFILE_FAILURE
    reason: "relative L2 error 2.9"
    comment: "measured 2026-08-27"
```

`reason` is a substring that **must appear in the log the failure produces**.

## Skipping a model

A model that is too expensive to run while a known failure stands can opt out
of the default run, while staying in the tree:

```yaml
skip: "known COMPILE_FAILURE; tracked by micro/conv-gelu-4x32x8x50-f16"
```

Its tests are reported as skipped with that reason, before any fixture runs,
so the model is not even imported. `--run-skipped` runs them anyway, e.g.
`--model=Qwen/Qwen3-ASR-0.6B --run-skipped`. Keep its `expected_outcomes`
current, so that a run with `--run-skipped` still checks the failure.


## Note
Note that if you run a model sequentially and parallel, its compiled twice at the moment, so this is only recommended for smaller models

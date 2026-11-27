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
| `prefetch.py` | import and compile the next test on the host while the current one runs on the device |
| `tool_paths.py` | resolve an artifact path, or ask Bazel |
| `conftest.py` | options and fixtures, the only pytest glue |
| `test_models.py` | the device test, parametrized over the tree |
| `test_kv_cache.py` | a host test of `models/kv_cache.py`, no device needed |

## On the device

```
/data/local/tmp/hexagon_it/
    runs/<id>[-tracy]/      hash of host user and repo root; emptied when
                            a run starts, removed when it ends
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
with the KV cache in module state (globals): `prefill` on a prompt, at
position 0 or after the tokens already in the cache, and `decode` of the next
token, with dynamic shapes. The spec's inputs, a prompt of token ids, only
serve as the example arguments of the export.

The two functions are not run one by one, each from an empty cache, but
together, as they are meant to: `iree-run-llm` (`tools/iree-run-llm-main.c`, on
the inference engine in `tools/llm/`) runs the module on the device as an
inference loop. It tokenizes the prompt itself, runs `prefill`, then `decode`
once per further token, and generates `max_new_tokens` tokens greedily,
ignoring stop tokens. Then it does the same for each further prompt, as a
further turn of a conversation: the prompt is prefilled after the tokens so
far, preceded by the last generated token, which no decode has written to the
cache yet. Such a model needs a `generation` list, one entry per prompt:

```yaml
generation:
  - prompt: "def fibonacci(n):\n"
    max_new_tokens: 32  # the default
    min_margin: 0.25    # the default
  - prompt: "\n\ndef factorial(n):\n"
    max_new_tokens: 16
```

All prompts, each padded, and their tokens have to fit in the export's
`max_seq_len`; the import fails if they do not.

`model.py` then also defines `get_tokenizer_json()`, the path of the model's
`tokenizer.json`, which the device gets too. The reference is
`KvCacheLM.generate_greedy`: the same functions run eagerly, with the prompts
padded as on the device. At every step, its most likely token has to lead the
runner-up by at least the prompt's `min_margin` in logits. Otherwise it is a
near-tie that rounding on the device may tip either way, after which the two
sides continue from different text, and the import fails with a `NearTieError`
naming the prompt, the step and both tokens: change the prompt, or shorten its
`max_new_tokens` to end before the near-tie. The import writes the margins to
`generation_reference.json`.

The check, in `stages/checking.py`, for each prompt:

* the prompt's token ids have to match the `tokenizers` library's exactly,
  which checks the device's tokenizer (the later prompts' without special
  tokens);
* prefill's logits, which `iree-run-llm --prefill_logits_output` writes, have
  to match the reference's within the model's `tolerances` at the positions
  prefill ran on (the padding's are never used);
* the generated tokens have to match the reference's exactly.

The reference's prefill and decode, mask and cache positions included, are
checked on the host by `test_kv_cache.py`, against a forward of the whole
history without a cache.

The log has the text and, under `generation timings:`, the time of the whole
`iree-run-llm` run (`generate`) and its own times of `prefill` and of one
decode step (`decode_per_token`), of the first prompt, and the same of each
further prompt N as `prefill_turnN` and `decode_per_token_turnN`.
`perf_compare.py` compares those of the first and second prompt against the
recent main runs, each as `<test id>:<name>`, as it compares the `run:` of
other models.

These models cannot use a profile run case (e.g. `parallel-profile`):
`iree-run-llm` does not run under Tracy, and the test rejects it.

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
`perf_compare.py` parses the `-rA` output of the session, the `run:` line under `timings:` of each passed test (or the `prefill:` and `decode_per_token:` lines under `generation timings:` of a token-generation test), and reads the earlier main runs' output from their GitHub logs with `gh`.
When no time can be found for any passed test, the check fails.

The time is one invocation of the function, as `iree-benchmark-module` measures
it on the device (`--benchmark_min_time=1x`), without starting the process or
loading the module. The rest is printed as `adb_setup:`, but not compared: the
host's wall clock around the adb command minus the invocation, so the adb
round trip, starting the process, opening the DSP session and loading the
module. When there is no plausible device time, the
host's is printed as `run_host:` instead, and the test is not compared.
Profile and debug-malloc run cases are not compared.

A token-generation test has two times, each compared on its own as
`<test id>:<name>`: `iree-run-llm`'s own times of `prefill` and of one decode
step, `decode_per_token`. `generate`, the host's wall clock around the whole
`iree-run-llm` run, is printed but not compared, as `adb_setup:` is not.

The thresholds when a warning is raised are command-line arguments.
To compare a local session:
```sh
# collect the log from stdout
.venv/bin/pytest integration_tests/ -v -rA ... | tee pytest-output.log
# compare against CI's main
python -m integration_tests.perf_compare --current pytest-output.log \
    --num-recent-runs 5 --max-runs-searched 10 \
    --warn-percent 5 --fail-percent 10 --min-delta-s 0.005 \
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

## Marks

A model can carry pytest marks, which `-m` selects:

```yaml
marks: [smoke]
```

`smoke` is a small subset for a quick check: all `micro/` models and
`google/gemma-3-270m-it`. CI runs it in `test-cmake` on PRs
(`pytest -m smoke`), as the bazel build covers the whole suite.


## Host and device overlap

The device stage (push, run, pull) is serial, one test at a time. The host
stages are not: at the start of the session, the imports and compiles of all
collected tests are queued, in test order, to one background thread
(`prefetch.py`), which runs ahead of the device. It does one import or compile
at a time. Each model is imported once, and each (model, compile case) compiled
once, shared by its run cases.

The imports run in one long-lived child process, so torch loads and warms up
once per session. Each import's whole output (warnings, progress bars, C++
logs) goes to `import.log` next to its `model.mlirbc`, and is printed by the first
test of that model. The child is replaced when its memory grows past 4 GiB (`_RECYCLE_RSS_BYTES`), and after a crash.

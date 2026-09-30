## Measurement Scripts for register pressure estimation tests

These scripts where used to get the actual measured register usage annotated to the testcases.

Setup:
* Build  bazel target `@llvm-project//llvm:llc`: we need llc with assertions enabled

Usage - one test file at a time
* `measure_all.py ../<TestCaseFile>.cpp` measures the real HvxVR for every
  MLIR snippet and every `TileConfig` the file declares, writes one row per config to
  `results.csv`, and prints every case where the
  `note` field's claimed measured value disagrees with what the compiler
  produces now. See the manual steps below for explanation on how this is done

Usage - one snippet/tiling at a time (what `measure_all.py` automates)
* `extract_test_ir.py ../<TestCaseFile>.cpp` (`--name` to say which
  one to) writes that test's IR to `example.mlir`  (Or write `example.mlir` by hand.)
* `to_config.sh` compiles `example.mlir` to `config.mlir` (`--compile-to=executable-configurations`)
* Adjust the tile sizes in `config.mlir` attributes if necessary: overwrite
  the op's `lowering_config`'s `vector_common_parallel`/`vector_reduction`
* `get_reg_pressure.sh` compiles the config and runs the resulting `.opt.ll` through `llc` including  the machine scheduler, recording per-region register pressure after machine scheduling. It stores this information to `pressure.csv`

## On-device benchmark suite (vector tile search, `RegisterEstimation/TileSelectionReport.md`)

`kernels/*.mlir` is a suite of standalone dispatch kernels (one `func.func`
per file), covering the per_op/dispatch gtest suites plus a few kernel
shapes those don't (non-power-of-two and prime-dimensioned matmuls, an
f16-throughout matmul, NHWC conv2d, softmax, RMSNorm, a row-mean, a
tanh/GELU body). `to_exe.sh` and `benchmark.sh` compile and run them
individually, same as `example.mlir`; `bench_all.py` automates the whole
suite:

* `bench_all.py --variant <label> --out results/<label>.csv [--extra-flags
  "..."] [--kernel k1 k2 ...]` compiles each kernel (`to_config.sh` +
  `to_exe.sh`, with `--extra-flags` passed through to `to_config.sh` as
  `EXTRA_FLAGS`), runs it on device (`benchmark.sh`, median of
  `--repetitions`, default 3), and diffs its output against
  `results/baseline_outputs/<kernel>.output.npy` (written by the
  `--variant baseline` run, via `run_module.sh`) to catch a correctness
  regression, not only a performance one.
* `results/baseline.csv` and `results/final.csv` are the heuristic and
  vector-tile-search (`--iree-hexagon-experimental-vector-tile-search=true`) runs behind
  `TileSelectionReport.md` section 10 (and section 11's calibration/tuning
  follow-up)'s numbers, at the shipped defaults (register margin 8,
  calibrated `math.tanh`/`math.rsqrt`); `results/summary.csv` merges the
  two into one row per kernel with the speedup. Section 11 also covers two
  approaches that were tried and dropped - a roofline cost model and a
  higher register margin (12) - with the on-device numbers that ruled them
  out, but their intermediate result CSVs are not kept around.

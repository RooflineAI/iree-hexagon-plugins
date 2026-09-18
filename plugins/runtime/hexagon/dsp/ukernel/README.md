# Hexagon DSP microkernels

This directory contains native DSP kernels that implement compiler-generated
calls.

## Why the HMX kernels live in the runtime

The HMX implementation cannot be compiled with the open source LLVM backend, used
by the codegen. It requires the use of the proprietary binaries used for compiling the
runtime instead. Nevertheless, once these closed-source changes are upstreamed to the
public llvm version, these kernels should be moved to the codegen side using the same
patterns that other IREE backends are using.

For now, the kernels are compiled into the DSP runtime skeleton. Codegen emits
calls to their stable C ABI and leaves those symbols undefined in
the executable; the DSP loader resolves them from `libhexagon_dsp_skel.so`.
This makes the compiler and runtime ABI version-coupled. If executable-side
linking of SDK-compiled objects is added later, these kernels can move to that
path without changing the HMX IR contracts.

HMX microkernels have their own Bazel library so target features and
force-linking do not leak into the general runtime helpers.

## Testing

The `hmx/test` package cross-compiles small shared objects and executes them in
the Hexagon v79 simulator. Each module acquires VTCM, powers HVX/HMX, and locks
HMX using the resource lifecycle originally validated by the
`L-roro/07-29-hmx-matmul-playground` benchmark driver.

The tests compare packing, unpacking, and accumulator/matmul behavior against
independent scalar implementations. They validate instruction-level functional
behavior in the simulator, not physical device power behavior, resource
contention, or performance.

### `run_main_on_hexagon_sim`

`hexagon-sim` executes an ELF, but the three Bazel-built test artifacts are
shared objects. The SDK's v79 `run_main_on_hexagon_sim` is the executable ELF
that starts in QuRT, loads one test module, resolves its HAP and
compute-resource symbols, calls its `main`, and propagates the result. It is
needed by this shared-object test architecture, not by `hexagon-sim` in
general.

The compute-resource fixture also needs the SDK's complete QuRT simulator
environment:

- `--cosim_file` supplies the timer and interrupt-controller cosims used by
  QuRT.
- `--l2tcm_base` and `--subsystem_base` select the v79 TCM and subsystem address
  map.
- `--rtos` loads the QuRT simulator model.
- `runelf.pbn` starts the QuRT user process in which
  `run_main_on_hexagon_sim` loads the module.

Without this environment, ordinary standalone code can execute, but the HAP
compute-resource services needed to acquire VTCM and lock HMX are not
available.

### Why the simulator test has a launcher

Bazel executes the shared Patio `hexagon_sim_test` runner as the host test
program; that launcher then stages the cross-compiled module and invokes
`hexagon-sim`. It creates SDK configuration files containing runfile-specific
paths, provides the simulator's ncurses compatibility links, enforces the
timeout, and retains complete diagnostics on failure.

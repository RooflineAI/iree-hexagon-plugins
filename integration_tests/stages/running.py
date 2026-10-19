# Copyright 2026 RooflineAI GmbH
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Running a compiled module on a device with iree-run-module."""

from __future__ import annotations

import concurrent.futures
import shlex
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

from integration_tests.device import adb
from integration_tests.device.adb import AdbError
from integration_tests.device.deploy import Deployment


@dataclass(frozen=True)
class RunCase:
    """A runtime configuration: extra flags for iree-run-module."""

    name: str
    flags: tuple[str, ...] = ()
    # Run with the Tracy runtime and capture a profile, exported as CSV.
    profile: bool = False


_SEQUENTIAL_FLAGS = ("--hexagon_max_concurrent_dispatches=1",)

RUN_CASES: tuple[RunCase, ...] = (
    # The runtime default: as many concurrent dispatches as HVX contexts.
    RunCase(name="parallel"),
    RunCase(name="sequential", flags=_SEQUENTIAL_FLAGS),
    RunCase(name="parallel-profile", profile=True),
    RunCase(name="sequential-profile", flags=_SEQUENTIAL_FLAGS, profile=True),
)

RUN_CASES_BY_NAME = {case.name: case for case in RUN_CASES}
RUN_CASE_NAMES = tuple(RUN_CASES_BY_NAME)

# Every model runs with the runtime defaults unless its manifest asks for more.
DEFAULT_RUN_CASE_NAMES = ("parallel",)


@dataclass(frozen=True)
class ProfilerTools:
    """The host side of a Tracy profile: capture and CSV export."""

    tracy_capture: Path
    tracy_csvexport: Path
    port: int = 8086


# tracy-capture's exit code when the connection drops before the handshake.
# Through `adb forward` the local connect always succeeds, so this is what happens
# if tracy tries to connect before the device process listens looks like.
_CAPTURE_HANDSHAKE_DROPPED = 3


@dataclass
class RunModuleResult:
    remote_dir: str
    exit_code: int
    log: str
    logcat: str
    output_files: list[Path] = field(default_factory=list)
    timings: dict[str, float] = field(default_factory=dict)
    # Only set for a profiled run.
    profile_trace: Path | None = None
    profile_csv: Path | None = None
    # Why the profile could not be captured or exported, if it could not.
    profile_error: str | None = None

    def describe(self) -> str:
        return (
            f"iree-run-module exited {self.exit_code} in {self.remote_dir}\n"
            f"--- stdout/stderr ---\n{self.log}\n"
            f"--- logcat (adsprpc) ---\n{self.logcat}"
        )


def _filter_logcat_lines(logcat: str) -> str:
    """
    Filter lines from logcat.
    Remove lines that clutter the log, appear often and don't bring information
    useful for debugging.
    """
    lines = logcat.split("\n")
    remove_patterns = (
        ": adsp_process_group_mem_map",
        ": adsp_process_group_mem_unmap",
        ": fastrpc_invoke_fd_mmap_create: ",
        ": fastrpc_invoke_fd_mmap_destroy ",
        ": fastrpc_invoke_fd_mmap_get for fd ",
        ": fastrpc_invoke_fd_mmap_put for fd ",
        ": fastrpc_invoke_static_map_fd_alloc: ",
        ": fastrpc_invoke_static_map_fd_free ",
        ": no matching pending jobs, collect would deadlock",
    )
    lines = [
        line
        for line in lines
        if not any(rem_pat in line for rem_pat in remove_patterns)
    ]
    return "\n".join(lines)


def run_module_on_device(
    deployment: Deployment,
    case_dir_name: str,
    module_vmfb: Path,
    function: str,
    input_files: list[Path],
    output_names: list[str],
    local_output_dir: Path,
    extra_flags: tuple[str, ...] = (),
    timeout: float = 900.0,
    profiler: ProfilerTools | None = None,
) -> RunModuleResult:
    """Push a module plus its inputs, run it on the DSP, and pull the outputs.

    `timeout` bounds both the module push and the run.

    With `profiler`, the deployment must hold the Tracy runtime. The run is then
    captured into `local_output_dir/profile.tracy`and exported to `local_output_dir/profile.csv`.

    The command runs under `limit_lifetime`, so that when `timeout` fires and
    Python SIGKILLs the adb client, the on-device process group is killed too.
    Without it a timed-out run leaves iree-run-module alive holding the DSP, and
    every subsequent test fails for an unrelated reason.
    """
    timings: dict[str, float] = {}

    push_total_start = time.time()
    remote_dir = deployment.case_dir(case_dir_name)
    adb.shell(f"rm -rf {remote_dir}; mkdir -p {remote_dir}")
    # A large model's module takes a while to push, so it gets the same budget
    # as the run.
    push_module_start = time.time()
    adb.push([module_vmfb], f"{remote_dir}/module.vmfb", timeout=timeout)
    timings["push_module"] = time.time() - push_module_start
    if input_files:
        adb.push(input_files, remote_dir + "/")
    timings["push_total"] = time.time() - push_total_start

    argv = [
        deployment.limit_lifetime,
        deployment.run_module,
        "--module=module.vmfb",
        f"--function={function}",
        "--device=hexagon",
    ]
    argv += [f"--input=@{path.name}" for path in input_files]
    argv += [f"--output=@{name}" for name in output_names]
    argv += list(extra_flags)

    # DSP_LIBRARY_PATH is where the DSP side looks for the skel and for the
    # FARF mask file; see examples/running.md.
    env = ""
    if profiler is not None:
        # TRACY_NO_EXIT keeps the process alive until the capture has all data.
        env = f"TRACY_NO_EXIT=1 TRACY_PORT={profiler.port} "
    script = (
        f"cd {remote_dir} && "
        f"export DSP_LIBRARY_PATH={deployment.dsp_library_path} && "
        + env
        + shlex.join(argv)
    )

    local_output_dir.mkdir(parents=True, exist_ok=True)
    adb.logcat_clear()
    run_start = time.time()
    if profiler is None:
        exit_code, log = adb.shell_exit_code(script, timeout=timeout)
        profile_error = None
    else:
        exit_code, log, profile_error = _run_with_capture(
            script, profiler, local_output_dir / "profile.tracy", timeout
        )
    timings["run"] = time.time() - run_start
    logcat = adb.logcat_dump()
    pulled: list[Path] = []
    pull_start = time.time()
    for name in output_names:
        remote_output = f"{remote_dir}/{name}"
        local_output = local_output_dir / name
        # Outputs are pulled one at a time and a miss is tolerated: when the run
        # failed there may be no output at all, and the exit code plus the log
        # are the useful diagnostics, not a pull error on top of them.
        try:
            adb.pull([remote_output], local_output_dir)
        except AdbError:
            continue
        pulled.append(local_output)
    timings["pull"] = time.time() - pull_start

    profile_trace = profile_csv = None
    if profiler is not None:
        profile_trace = local_output_dir / "profile.tracy"
        if profile_error is None:
            export_start = time.time()
            profile_csv = local_output_dir / "profile.csv"
            profile_error = _export_csv(profiler, profile_trace, profile_csv)
            timings["csvexport"] = time.time() - export_start
    return RunModuleResult(
        remote_dir=remote_dir,
        exit_code=exit_code,
        log=log,
        logcat=_filter_logcat_lines(logcat),
        output_files=pulled,
        timings=timings,
        profile_trace=profile_trace,
        profile_csv=profile_csv,
        profile_error=profile_error,
    )


def _run_with_capture(
    script: str,
    profiler: ProfilerTools,
    trace: Path,
    timeout: float,
) -> tuple[int, str, str | None]:
    """Run `script` on the device while tracy-capture records it into `trace`.

    Returns the run's exit code and log, and why the capture failed, if it did.
    """
    capture_argv = [
        str(profiler.tracy_capture),
        "-o",
        str(trace),
        "-f",
        "-p",
        str(profiler.port),
    ]
    deadline = time.time() + timeout
    capture_log = ""
    capture_error: str | None = None
    adb.forward(profiler.port, profiler.port)
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            run = pool.submit(adb.shell_exit_code, script, timeout)
            # Retry until the device process listens. Thanks to TRACY_NO_EXIT
            # it cannot exit before a capture got its data, unless it fails.
            while True:
                try:
                    capture = subprocess.run(
                        capture_argv,
                        capture_output=True,
                        text=True,
                        timeout=max(deadline - time.time(), 1.0),
                    )
                except subprocess.TimeoutExpired:
                    capture_error = f"tracy-capture timed out after {timeout:.0f}s"
                    break
                capture_log = capture.stdout + capture.stderr
                if capture.returncode != _CAPTURE_HANDSHAKE_DROPPED:
                    break
                if run.done():
                    break
                time.sleep(0.2)
            exit_code, log = run.result()
    finally:
        adb.forward_remove(profiler.port)

    if capture_error is None:
        if capture.returncode != 0:
            capture_error = f"tracy-capture exited {capture.returncode}:\n{capture_log}"
        # tracy-capture exits 0 even when saving the trace failed.
        elif not trace.exists() or trace.stat().st_size == 0:
            capture_error = f"tracy-capture wrote no trace to {trace}:\n{capture_log}"
    return exit_code, log, capture_error


def _export_csv(profiler: ProfilerTools, trace: Path, csv: Path) -> str | None:
    """Export the GPU zones of `trace` to `csv`, returning why it failed, if it did.

    The DSP zones are Tracy GPU zones, one GPU context per DSP thread. The
    export has no column for the context, so the threads are merged.
    """
    export = subprocess.run(
        [str(profiler.tracy_csvexport), "-g", str(trace)],
        capture_output=True,
        text=True,
    )
    if export.returncode != 0:
        return f"tracy-csvexport exited {export.returncode}:\n{export.stderr}"
    csv.write_text(export.stdout)
    return None

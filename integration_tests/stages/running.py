# Copyright 2026 RooflineAI GmbH
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Running a compiled module on a device with iree-run-module."""

from __future__ import annotations

import shlex
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


@dataclass
class RunModuleResult:
    remote_dir: str
    exit_code: int
    log: str
    logcat: str
    output_files: list[Path] = field(default_factory=list)
    timings: dict[str, float] = field(default_factory=dict)
    # Only set for a profiled run.
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
    tracy_port: int | None = None,
) -> RunModuleResult:
    """Push a module plus its inputs, run it on the DSP, and pull the outputs.

    `timeout` bounds both the module push and the run.

    With `tracy_port`, the deployment must hold the Tracy runtime.
    The port is then used on device to connect `tracy-capture` to `iree-module-run`.
    It is not used on the host.

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
    script = (
        f"cd {remote_dir} && export DSP_LIBRARY_PATH={deployment.dsp_library_path} && "
    )
    if tracy_port is None:
        script += shlex.join(argv)
    else:
        script += _capture_script(deployment, argv, tracy_port)

    local_output_dir.mkdir(parents=True, exist_ok=True)
    adb.logcat_clear()
    run_start = time.time()
    exit_code, log = adb.shell_exit_code(script, timeout=timeout)
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

    profile_csv = profile_error = None
    if tracy_port is not None:
        pull_profile_start = time.time()
        profile_error = _pull_profile(remote_dir, local_output_dir)
        timings["pull_profile"] = time.time() - pull_profile_start
        if profile_error is None:
            profile_csv = local_output_dir / _PROFILE_CSV
    return RunModuleResult(
        remote_dir=remote_dir,
        exit_code=exit_code,
        log=log,
        logcat=_filter_logcat_lines(logcat),
        output_files=pulled,
        timings=timings,
        profile_csv=profile_csv,
        profile_error=profile_error,
    )


# Files the on-device capture and export leave in the case directory.
_PROFILE_TRACE = "profile.tracy"
_PROFILE_CSV = "profile.csv"
_CAPTURE_LOG = "capture.log"
_CSVEXPORT_LOG = "csvexport.log"
# "<tracy-capture exit code> <tracy-csvexport exit code>", the latter "-" when
# it did not run.
_PROFILE_EXIT_CODES = "profile.exit_codes"


def _capture_script(deployment: Deployment, argv: list[str], port: int) -> str:
    """The shell script running `argv` while tracy-capture records it, and
    tracy-csvexport exporting the recording.

    tracy-capture starts first and keeps retrying to connect until
    iree-run-module listens. TRACY_NO_EXIT keeps iree-run-module alive until the
    capture has all data, so after a successful run the capture finishes on its
    own. A failed run may never have listened, so then the capture is stopped.
    Everything runs under `limit_lifetime`, so a timeout kills it all.
    The script exits with the exit code of the run.

    The export keeps the GPU zones only (`-g`): the DSP zones are Tracy GPU
    zones, one GPU context per DSP thread. The export has no column for the
    context, so the threads are merged.
    """
    capture_argv = [
        deployment.limit_lifetime,
        deployment.tracy_capture,
        "-o",
        _PROFILE_TRACE,
        "-f",
        "-p",
        str(port),
    ]
    csvexport_argv = [
        deployment.limit_lifetime,
        deployment.tracy_csvexport,
        "-g",
        _PROFILE_TRACE,
    ]
    # The braces keep the `&` from sending the commands before it, which the
    # caller chains with `&&`, to the background too.
    return (
        f"{{ {shlex.join(capture_argv)} > {_CAPTURE_LOG} 2>&1 & capture_pid=$!; "
        f"TRACY_NO_EXIT=1 TRACY_PORT={port} {shlex.join(argv)}; run_exit_code=$?; "
        "if [ $run_exit_code -ne 0 ]; then kill $capture_pid; fi; "
        "wait $capture_pid; capture_exit_code=$?; csvexport_exit_code=-; "
        "if [ $capture_exit_code -eq 0 ]; then "
        f"{shlex.join(csvexport_argv)} > {_PROFILE_CSV} 2> {_CSVEXPORT_LOG}; "
        "csvexport_exit_code=$?; fi; "
        f"echo $capture_exit_code $csvexport_exit_code > {_PROFILE_EXIT_CODES}; "
        "exit $run_exit_code; }"
    )


def _pull_profile(remote_dir: str, local_output_dir: Path) -> str | None:
    """Pull the CSV the on-device export wrote, returning why there is none, if
    there is none.
    """
    status, exit_codes = adb.shell_exit_code(f"cat {remote_dir}/{_PROFILE_EXIT_CODES}")
    if status != 0:
        return f"the profiling left no exit codes:\n{exit_codes}"
    capture_exit_code, csvexport_exit_code = exit_codes.split()
    if capture_exit_code != "0":
        # tracy-capture redraws its progress line with carriage returns.
        log = _device_log(remote_dir, _CAPTURE_LOG).replace("\r", "\n")
        return f"tracy-capture exited {capture_exit_code}:\n{log}"
    if csvexport_exit_code != "0":
        log = _device_log(remote_dir, _CSVEXPORT_LOG)
        return f"tracy-csvexport exited {csvexport_exit_code}:\n{log}"
    try:
        adb.pull([f"{remote_dir}/{_PROFILE_CSV}"], local_output_dir)
    except AdbError as err:
        return f"could not pull the CSV tracy-csvexport wrote:\n{err}"
    return None


def _device_log(remote_dir: str, name: str) -> str:
    """A log left in `remote_dir`, for an error message."""
    _, log = adb.shell_exit_code(f"cat {remote_dir}/{name}")
    return log

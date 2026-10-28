# Copyright 2026 RooflineAI GmbH
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Running a compiled module on a device with iree-run-module."""

from __future__ import annotations

import hashlib
import posixpath
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
    # Run iree-run-module under bionic's malloc debug, see _DEBUG_MALLOC_OPTIONS.
    debug_malloc: bool = False


_SEQUENTIAL_FLAGS = ("--hexagon_max_concurrent_dispatches=1",)

RUN_CASES: tuple[RunCase, ...] = (
    # The runtime default: as many concurrent dispatches as HVX contexts.
    RunCase(name="parallel"),
    RunCase(name="sequential", flags=_SEQUENTIAL_FLAGS),
    RunCase(name="parallel-profile", profile=True),
    RunCase(name="sequential-profile", flags=_SEQUENTIAL_FLAGS, profile=True),
    RunCase(name="parallel-debug-malloc", debug_malloc=True),
    RunCase(name="sequential-debug-malloc", flags=_SEQUENTIAL_FLAGS, debug_malloc=True),
    RunCase(name="parallel-profile-debug-malloc", profile=True, debug_malloc=True),
    RunCase(
        name="sequential-profile-debug-malloc",
        flags=_SEQUENTIAL_FLAGS,
        profile=True,
        debug_malloc=True,
    ),
)

# Bionic's malloc debug: abort on a free/realloc of a pointer that is not a live
# allocation. It refuses any single allocation above 2^31 - 1 bytes with a
# silent ENOMEM (PointerInfoType::MaxSize() in libc/malloc_debug/PointerData.h),
# and iree-run-module reads the whole module into one, so a model whose .vmfb
# is 2 GiB or larger crashes under it. Only for smaller models.
_DEBUG_MALLOC_OPTIONS = "verify_pointers abort_on_error"

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


# Modules at least this large are kept on the device across runs and sent as
# rsync deltas. Below it a plain adb push is faster: in CI each adb round trip
# to the phone rig costs ~1.5 s, and rsync plus the copy-and-check take a few of
# them, while adb push sends a module this size in about one.
_KEEP_MODULE_MIN_BYTES = 64 << 20


class ModuleMismatchError(RuntimeError):
    """The module the device would run is not the module that was built."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _check_module_after_rsync(
    module_vmfb: Path, kept_module: str, remote_module: str
) -> None:
    """Copy the kept module into the case directory, and check it is the build.

    A copy rather than a link, so the case directory keeps the module it ran
    after a later run of this case has replaced kept_module. It is cheap on the
    device (~0.3 s for 270 MB), and shares one adb round trip with the checksum.

    The transfer to kept_module is itself verified, so a mismatch here means the
    kept module changed between the transfer and the copy: another execution of
    the integration tests, using the same device root, sent its own build of the
    same case. The device root has no lock, so this is the point where that
    sharing would otherwise go unnoticed and the test would run someone else's
    module.
    """
    expected = _sha256(module_vmfb)
    actual = adb.shell(
        f"cp {kept_module} {remote_module} && sha256sum {remote_module}"
    ).split()[0]
    if actual != expected:
        raise ModuleMismatchError(
            f"{remote_module} on the device does not match the module built at "
            f"{module_vmfb}:\n"
            f"  device: sha256 {actual}\n"
            f"  host:   sha256 {expected}\n"
            "Another execution of the integration tests is probably using the "
            "same device root concurrently. The device root has no lock - give "
            "each execution its own --device-root, or run them one at a time."
        )


def push_module(
    module_vmfb: Path,
    remote_module: str,
    kept_module: str,
    host_rsync: Path,
    device_rsync: str,
    timeout: float,
    setup: str = "",
    timings: dict[str, float] | None = None,
) -> None:
    """Make `remote_module` on the device a copy of the `module_vmfb` built.

    A module of at least `_KEEP_MODULE_MIN_BYTES` goes to `kept_module` (see
    `integration_tests/device/deploy.py`) with `host_rsync`, as a delta against
    the module sent there last time, or with adb push if there is none yet.
    `remote_module` then gets its own copy of it, and `ModuleMismatchError` is
    raised unless that copy's checksum matches `module_vmfb`, defending against a different user doing this at teh same time.
    A smaller module is pushed straight to `remote_module`.

    `device_rsync` is the command that runs rsync on the device. `setup` is a
    shell script run first, in the same adb shell that looks for a kept
    module. `timeout` bounds the transfer. With `timings`, the durations of the
    transfer and of the check are added to it.
    """
    if timings is None:
        timings = {}
    should_keep_module = module_vmfb.stat().st_size >= _KEEP_MODULE_MIN_BYTES
    # One adb round trip runs the setup and, for a kept module, finds out
    # whether an earlier run left one to diff against.
    script = setup or "true"
    if should_keep_module:
        script += (
            f"; mkdir -p {posixpath.dirname(kept_module)}"
            f"; if [ -f {kept_module} ]; then echo kept; fi"
        )
    has_kept_module = adb.shell(script).strip() == "kept"
    push_module_start = time.time()
    if not should_keep_module:
        adb.push([module_vmfb], remote_module, timeout=timeout)
    elif has_kept_module:
        # A delta against the module this case sent last time, so a rebuilt
        # module only sends what changed. With `device_rsync` under
        # limit_lifetime, the device side ends if the transfer times out.
        adb.sync(
            host_rsync=host_rsync,
            device_rsync=device_rsync,
            local_path=module_vmfb,
            remote_path=kept_module,
            timeout=timeout,
        )
    else:
        # Nothing to diff against, and for a full transfer adb push is faster
        # than rsync over `adb shell`.
        adb.push([module_vmfb], kept_module, timeout=timeout)
    timings["push_module"] = time.time() - push_module_start
    if should_keep_module:
        # otherwise, it is already pushed to the case dir, and no need to verify, that no one else overrides it in between
        verify_module_start = time.time()
        _check_module_after_rsync(module_vmfb, kept_module, remote_module)
        timings["verify_module"] = time.time() - verify_module_start


def run_module_on_device(
    deployment: Deployment,
    case_dir_name: str,
    module_vmfb: Path,
    function: str,
    input_files: list[Path],
    output_names: list[str],
    local_output_dir: Path,
    host_rsync: Path,
    extra_flags: tuple[str, ...] = (),
    timeout: float = 900.0,
    tracy_port: int | None = None,
    debug_malloc: bool = False,
) -> RunModuleResult:
    """Push a module plus its inputs, run it on the DSP, and pull the outputs.

    A module of at least `_KEEP_MODULE_MIN_BYTES` is kept on the device across
    runs (see `integration_tests/device/deploy.py`) and goes over with
    `host_rsync`, as a delta against the one this case sent last time, or with
    adb push if there is none yet. The case directory then gets its own copy of
    it, and `ModuleMismatchError` is raised unless that copy's checksum matches
    the `module_vmfb` built on the host. A smaller module is pushed straight
    into the case directory.

    `timeout` bounds both the module push and the run.

    With `debug_malloc`, iree-run-module runs under bionic's malloc debug, see
    `_DEBUG_MALLOC_OPTIONS`.

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
    # The first transfer of a large module takes a while, so it gets the same
    # budget as the run.
    push_module(
        module_vmfb=module_vmfb,
        remote_module=f"{remote_dir}/module.vmfb",
        kept_module=deployment.kept_module(case_dir_name),
        host_rsync=host_rsync,
        device_rsync=f"{deployment.limit_lifetime} {deployment.rsync}",
        timeout=timeout,
        setup=f"rm -rf {remote_dir}; mkdir -p {remote_dir}",
        timings=timings,
    )
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
    # Set for the run only, not for the Tracy tools of a profiled run.
    env = (
        f"LIBC_DEBUG_MALLOC_OPTIONS={shlex.quote(_DEBUG_MALLOC_OPTIONS)} "
        if debug_malloc
        else ""
    )
    if tracy_port is None:
        script += env + shlex.join(argv)
    else:
        script += _capture_script(deployment, env + shlex.join(argv), tracy_port)

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


def _capture_script(deployment: Deployment, run_command: str, port: int) -> str:
    """The shell script running `run_command` while tracy-capture records it, and
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
    return f"""{{
    {shlex.join(capture_argv)} > {_CAPTURE_LOG} 2>&1 &
    capture_pid=$!
    TRACY_NO_EXIT=1 TRACY_PORT={port} {run_command}
    run_exit_code=$?
    if [ $run_exit_code -ne 0 ]; then
      kill $capture_pid
    fi
    wait $capture_pid
    capture_exit_code=$?
    csvexport_exit_code=-
    if [ $capture_exit_code -eq 0 ]; then
      {shlex.join(csvexport_argv)} > {_PROFILE_CSV} 2> {_CSVEXPORT_LOG}
      csvexport_exit_code=$?
    fi
    echo $capture_exit_code $csvexport_exit_code > {_PROFILE_EXIT_CODES}
    exit $run_exit_code
    }}"""


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

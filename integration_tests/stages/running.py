# Copyright 2026 RooflineAI GmbH
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Running a compiled module on a device with iree-run-module."""

from __future__ import annotations

import hashlib
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


RUN_CASES: tuple[RunCase, ...] = (
    # The runtime default: as many concurrent dispatches as HVX contexts.
    RunCase(name="parallel"),
    RunCase(name="sequential", flags=("--hexagon_max_concurrent_dispatches=1",)),
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


class ModuleMismatchError(RuntimeError):
    """The module the device would run is not the module that was built."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _check_module_copy(module_vmfb: Path, remote_module: str) -> None:
    """Fail unless the device's copy of the module is byte-identical to the build.

    rsync verifies its own transfer, so a mismatch here means the kept module
    changed between the sync and the copy: another execution of the integration
    tests, using the same device root, sent its own build of the same case. The
    device root has no lock, so this is the point where that sharing would
    otherwise go unnoticed and the test would run someone else's module.
    """
    expected = _sha256(module_vmfb)
    actual = adb.shell(f"sha256sum {remote_module}").split()[0]
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
) -> RunModuleResult:
    """Push a module plus its inputs, run it on the DSP, and pull the outputs.

    The module goes over with `host_rsync`, as a delta against the one this
    case sent last time; see `integration_tests/device/deploy.py`. The case
    directory then gets its own copy of it. Raises `ModuleMismatchError` unless
    that copy's checksum matches the `module_vmfb` built on the host.

    `timeout` bounds both the module push and the run.

    The command runs under `limit_lifetime`, so that when `timeout` fires and
    Python SIGKILLs the adb client, the on-device process group is killed too.
    Without it a timed-out run leaves iree-run-module alive holding the DSP, and
    every subsequent test fails for an unrelated reason.
    """
    timings: dict[str, float] = {}

    push_total_start = time.time()
    remote_dir = deployment.case_dir(case_dir_name)
    adb.shell(f"rm -rf {remote_dir}; mkdir -p {remote_dir}")
    # rsync against the module this case sent last time, so a rebuilt module
    # only sends what changed. The first transfer of a case sends it all, which
    # for a large model takes a while, so it gets the same budget as the run.
    # limit_lifetime ends the device side if the transfer times out.
    push_module_start = time.time()
    kept_module = deployment.kept_module(case_dir_name)
    adb.sync(
        host_rsync=host_rsync,
        device_rsync=f"{deployment.limit_lifetime} {deployment.rsync}",
        local_path=module_vmfb,
        remote_path=kept_module,
        timeout=timeout,
    )
    # A copy rather than a link, so the case directory keeps the module it ran
    # after a later run of this case has replaced kept_module. It is cheap on
    # the device (~0.3 s for 270 MB).
    remote_module = f"{remote_dir}/module.vmfb"
    adb.shell(f"cp {kept_module} {remote_module}")
    timings["push_module"] = time.time() - push_module_start
    verify_module_start = time.time()
    _check_module_copy(module_vmfb, remote_module)
    timings["verify_module"] = time.time() - verify_module_start
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
        f"cd {remote_dir} && "
        f"export DSP_LIBRARY_PATH={deployment.dsp_library_path} && " + shlex.join(argv)
    )

    adb.logcat_clear()
    run_start = time.time()
    exit_code, log = adb.shell_exit_code(script, timeout=timeout)
    timings["run"] = time.time() - run_start
    logcat = adb.logcat_dump()

    local_output_dir.mkdir(parents=True, exist_ok=True)
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
    return RunModuleResult(
        remote_dir=remote_dir,
        exit_code=exit_code,
        log=log,
        logcat=_filter_logcat_lines(logcat),
        output_files=pulled,
        timings=timings,
    )

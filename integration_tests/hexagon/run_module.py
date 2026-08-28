# Copyright 2026 RooflineAI GmbH
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Running a compiled module on a device with iree-run-module."""

from __future__ import annotations

import shlex
from dataclasses import dataclass, field
from pathlib import Path

from integration_tests.hexagon.adb_device import AdbDevice, AdbError
from integration_tests.hexagon.hexagon_runtime import DeployedRuntime


@dataclass
class RunModuleResult:
    remote_dir: str
    exit_code: int
    log: str
    logcat: str
    output_files: list[Path] = field(default_factory=list)

    def describe(self) -> str:
        return (
            f"iree-run-module exited {self.exit_code} in {self.remote_dir}\n"
            f"--- stdout/stderr ---\n{self.log}\n"
            f"--- logcat (adsprpc) ---\n{self.logcat}"
        )


def run_module_on_device(
    runtime: DeployedRuntime,
    case_dir_name: str,
    module_vmfb: Path,
    function: str,
    input_files: list[Path],
    output_names: list[str],
    local_output_dir: Path,
    extra_flags: tuple[str, ...] = (),
    timeout: float = 900.0,
) -> RunModuleResult:
    """Push a module plus its inputs, run it on the DSP, and pull the outputs.

    Each case gets its own subdirectory of the runtime directory so that a
    stale artifact from another case can never be picked up, and so a failure
    can be inspected on the device afterwards.
    """
    device: AdbDevice = runtime.device
    remote_dir = f"{runtime.remote_dir}/cases/{case_dir_name}"
    device.shell(f"rm -rf {remote_dir}; mkdir -p {remote_dir}")
    device.push([module_vmfb], f"{remote_dir}/module.vmfb")
    if input_files:
        device.push(input_files, remote_dir + "/")

    input_args = " ".join(
        f"--input=@{shlex.quote(path.name)}" for path in input_files
    )
    output_args = " ".join(f"--output=@{shlex.quote(name)}" for name in output_names)
    # DSP_LIBRARY_PATH is where the DSP side looks for the skel and for the
    # FARF mask file; see examples/running.md.
    script = (
        f"cd {remote_dir} && "
        f"export DSP_LIBRARY_PATH={runtime.dsp_library_path} && "
        f"{runtime.run_module} --module=module.vmfb "
        f"--function={shlex.quote(function)} --device=hexagon "
        f"{input_args} {output_args} "
        + " ".join(shlex.quote(flag) for flag in extra_flags)
    )

    device.logcat_clear()
    exit_code, log = device.shell_exit_code(
        script, exit_code_file=f"{remote_dir}/exit-code", timeout=timeout
    )
    logcat = device.logcat_dump()

    local_output_dir.mkdir(parents=True, exist_ok=True)
    pulled: list[Path] = []
    for name in output_names:
        remote_output = f"{remote_dir}/{name}"
        local_output = local_output_dir / name
        # Outputs are pulled one at a time and a miss is tolerated: when the run
        # failed there may be no output at all, and the exit code plus the log
        # are the useful diagnostics, not a pull error on top of them.
        if device.exists(remote_output):
            try:
                device.pull([remote_output], local_output_dir)
            except AdbError:
                continue
            pulled.append(local_output)
    return RunModuleResult(
        remote_dir=remote_dir,
        exit_code=exit_code,
        log=log,
        logcat=logcat,
        output_files=pulled,
    )

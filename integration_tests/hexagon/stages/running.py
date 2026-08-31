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

from integration_tests.hexagon.device import adb
from integration_tests.hexagon.device.adb import AdbError
from integration_tests.hexagon.device.deploy import Deployment


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
    deployment: Deployment,
    case_dir_name: str,
    module_vmfb: Path,
    function: str,
    input_files: list[Path],
    output_names: list[str],
    local_output_dir: Path,
    parameter_file: str | None = None,
    parameter_scope: str = "model",
    parameter_mode: str = "file",
    extra_flags: tuple[str, ...] = (),
    timeout: float = 900.0,
) -> RunModuleResult:
    """Push a module plus its inputs, run it on the DSP, and pull the outputs.

    The command runs under `limit_lifetime`, so that when `timeout` fires and
    Python SIGKILLs the adb client, the on-device process group is killed too.
    Without it a timed-out run leaves iree-run-module alive holding the DSP, and
    every subsequent test fails for an unrelated reason.
    """
    remote_dir = deployment.case_dir(case_dir_name)
    adb.shell(f"rm -rf {remote_dir}; mkdir -p {remote_dir}")
    adb.push([module_vmfb], f"{remote_dir}/module.vmfb")
    if input_files:
        adb.push(input_files, remote_dir + "/")

    argv = [
        deployment.limit_lifetime,
        deployment.run_module,
        "--module=module.vmfb",
        f"--function={function}",
        "--device=hexagon",
    ]
    argv += [f"--input=@{path.name}" for path in input_files]
    argv += [f"--output=@{name}" for name in output_names]
    if parameter_file is not None:
        # The imported IR names its scope: #flow.parameter.named<"model"::"...">.
        argv += [
            f"--parameters={parameter_scope}={parameter_file}",
            f"--parameter_mode={parameter_mode}",
        ]
    argv += list(extra_flags)

    # DSP_LIBRARY_PATH is where the DSP side looks for the skel and for the
    # FARF mask file; see examples/running.md.
    script = (
        f"cd {remote_dir} && "
        f"export DSP_LIBRARY_PATH={deployment.dsp_library_path} && " + shlex.join(argv)
    )

    adb.logcat_clear()
    exit_code, log = adb.shell_exit_code(script, timeout=timeout)
    logcat = adb.logcat_dump()

    local_output_dir.mkdir(parents=True, exist_ok=True)
    pulled: list[Path] = []
    for name in output_names:
        remote_output = f"{remote_dir}/{name}"
        local_output = local_output_dir / name
        # Outputs are pulled one at a time and a miss is tolerated: when the run
        # failed there may be no output at all, and the exit code plus the log
        # are the useful diagnostics, not a pull error on top of them.
        if adb.exists(remote_output):
            try:
                adb.pull([remote_output], local_output_dir)
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

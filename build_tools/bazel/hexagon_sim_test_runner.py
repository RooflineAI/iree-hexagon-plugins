#!/usr/bin/env python3
"""Runs a shared-object test module in the Hexagon v79 QuRT simulator."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

_COMPAT_LIBS = ("libpanel", "libncurses", "libtinfo", "libform", "libmenu")


def _create_compat_libs(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    system_lib_dirs = (
        Path("/lib/x86_64-linux-gnu"),
        Path("/usr/lib/x86_64-linux-gnu"),
    )
    for library in _COMPAT_LIBS:
        source = next(
            (
                candidate
                for system_dir in system_lib_dirs
                for candidate in sorted(system_dir.glob(f"{library}.so.6*"))
                if candidate.is_file()
            ),
            None,
        )
        if source is not None:
            (directory / f"{library}.so.5").symlink_to(source)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--module", required=True, type=Path)
    parser.add_argument("--simulator", required=True, type=Path)
    parser.add_argument("--run-main", required=True, type=Path)
    parser.add_argument("--runelf", required=True, type=Path)
    parser.add_argument("--qurt-model", required=True, type=Path)
    args = parser.parse_args()
    module_input = args.module.resolve()
    simulator = args.simulator.resolve()
    run_main = args.run_main.resolve()
    runelf = args.runelf.resolve()
    qurt_model = args.qurt_model.resolve()

    test_tmpdir = os.environ.get("TEST_TMPDIR")
    with tempfile.TemporaryDirectory(
        dir=test_tmpdir, prefix="hexagon_sim_test_"
    ) as tmp:
        workdir = Path(tmp)
        module = workdir / module_input.name
        shutil.copy2(module_input, module)

        compat_dir = workdir / "compat"
        _create_compat_libs(compat_dir)
        simulator_lib_dir = simulator.parents[1] / "lib"

        # Reproduce the SDK's runHexagonSim environment. qtimer and l2vic
        # provide QuRT's timer/interrupt devices, while the OS-awareness model
        # provides the simulated QuRT kernel services used by resource APIs.
        q6ss_config = workdir / "q6ss.cfg"
        q6ss_config.write_text(
            f"{simulator_lib_dir}/iss/qtimer.so --csr_base=0xFC900000 "
            "--irq_p=3 --freq=19200000 --cnttid=1\n"
            f"{simulator_lib_dir}/iss/l2vic.so 32 0xFC910000\n"
        )
        osam_config = workdir / "osam.cfg"
        osam_config.write_text(f"{qurt_model}\n")
        env = dict(os.environ)
        library_paths = [str(compat_dir), str(simulator_lib_dir)]
        if env.get("LD_LIBRARY_PATH"):
            library_paths.append(env["LD_LIBRARY_PATH"])
        env["LD_LIBRARY_PATH"] = os.pathsep.join(library_paths)

        # The base addresses match the SDK v79 memory map. runelf boots the
        # QuRT user process; run_main then dlopens the staged test module and
        # forwards its return value to --simulated_returnval.
        command = [
            str(simulator),
            "-mv79",
            "--simulated_returnval",
            "--usefs",
            str(workdir),
            "--cosim_file",
            str(q6ss_config),
            "--l2tcm_base",
            "0xd800",
            "--subsystem_base",
            "0xFC90",
            "--rtos",
            str(osam_config),
            str(runelf),
            "--",
            str(run_main),
            "--",
            module.name,
        ]
        try:
            result = subprocess.run(
                command,
                cwd=workdir,
                env=env,
                capture_output=True,
                text=True,
                timeout=300,
                check=False,
            )
        except subprocess.TimeoutExpired as error:
            print("Hexagon simulator timed out after 300 seconds", file=sys.stderr)
            print(f"command: {' '.join(command)}", file=sys.stderr)
            print(f"stdout:\n{error.stdout or '<empty>'}", file=sys.stderr)
            print(f"stderr:\n{error.stderr or '<empty>'}", file=sys.stderr)
            return 1

        if result.stdout:
            print(result.stdout, end="" if result.stdout.endswith("\n") else "\n")

        if result.returncode != 0:
            print(
                f"Hexagon simulator test failed with return code {result.returncode}",
                file=sys.stderr,
            )
            print(f"command: {' '.join(command)}", file=sys.stderr)
            print(f"stdout:\n{result.stdout or '<empty>'}", file=sys.stderr)
            print(f"stderr:\n{result.stderr or '<empty>'}", file=sys.stderr)
            return 1
        return 0


if __name__ == "__main__":
    raise SystemExit(main())

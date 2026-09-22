# Copyright 2026 RooflineAI GmbH
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""The Hexagon compilation configurations exercised by these tests."""

from __future__ import annotations

import dataclasses
import subprocess
from pathlib import Path

BASE_FLAGS: tuple[str, ...] = (
    "--iree-hal-target-device=hexagon",
    "--iree-input-type=auto",
    "--iree-hexagon-v=79",
    "--iree-hexagon-features=+hvxv79,+hvx-length128b",
    "--iree-opt-data-tiling=false",
    "--iree-stream-resource-min-offset-alignment=128",
)


@dataclasses.dataclass(frozen=True)
class CompileCase:
    name: str
    extra_flags: tuple[str, ...] = ()

    def flags(self, model_flags: tuple[str, ...] = ()) -> list[str]:
        return [*BASE_FLAGS, *self.extra_flags, *model_flags]


COMPILE_CASES: tuple[CompileCase, ...] = (
    CompileCase(
        name="hexagon",
        extra_flags=(
            "--iree-hexagon-launch-config-selector=hexagon",
            "--iree-hexagon-enable-vtcm-tiling=false",
            "--iree-hexagon-enable-hmx-matmul=false",
        ),
    ),
    CompileCase(
        name="hexagon-vtcm-hmx",
        extra_flags=(
            "--iree-hexagon-launch-config-selector=hexagon",
            "--iree-hexagon-enable-vtcm-tiling=true",
            "--iree-hexagon-enable-hmx-matmul=true",
        ),
    ),
    CompileCase(
        name="llvmcpu-baseline",
        extra_flags=(
            "--iree-hexagon-launch-config-selector=llvmcpu",
            "--iree-hexagon-enable-vtcm-tiling=false",
            "--iree-hexagon-enable-hmx-matmul=false",
        ),
    ),
)

COMPILE_CASES_BY_NAME = {case.name: case for case in COMPILE_CASES}
COMPILE_CASE_NAMES = tuple(COMPILE_CASES_BY_NAME)

# Every model exercises both custom Hexagon configurations unless its manifest
# deliberately narrows the matrix. The LLVMCPU selector is a sparse comparison
# baseline and must be requested by name.
DEFAULT_COMPILE_CASE_NAMES = ("hexagon", "hexagon-vtcm-hmx")


class CompilationError(RuntimeError):
    """Raised when iree-compile fails. Carries the log, for reason matching."""

    def __init__(self, message: str, log: str = "") -> None:
        super().__init__(message)
        self.log = log


def linker_flags(linker: Path) -> list[str]:
    """Point both linkers at `linker` instead of letting them search PATH."""
    return [
        f"--iree-hexagon-linker-path={linker}",
        f"--iree-llvmcpu-embedded-linker-path={linker}",
    ]


def compile_model(
    iree_compile: Path,
    model_mlir: Path,
    output_vmfb: Path,
    compile_case: CompileCase,
    log_file: Path | None = None,
    linker: Path | None = None,
    model_flags: tuple[str, ...] = (),
) -> Path:
    """Compile `model_mlir` for Hexagon, returning the vmfb path."""
    output_vmfb.parent.mkdir(parents=True, exist_ok=True)
    argv = [
        str(iree_compile),
        str(model_mlir),
        "-o",
        str(output_vmfb),
        *compile_case.flags(model_flags),
        *(linker_flags(linker) if linker is not None else []),
    ]
    result = subprocess.run(argv, capture_output=True, text=True)
    log = result.stdout + result.stderr
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        log_file.write_text(log)
    if result.returncode != 0:
        detail = f"see {log_file}" if log_file is not None else log[-4000:]
        raise CompilationError(
            f"iree-compile failed with exit code {result.returncode} for "
            f"{model_mlir} (case '{compile_case.name}'): {detail}",
            log=log,
        )
    if not output_vmfb.exists():
        raise CompilationError(
            f"iree-compile reported success but produced no {output_vmfb}", log=log
        )
    return output_vmfb

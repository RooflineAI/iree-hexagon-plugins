# Copyright 2026 RooflineAI GmbH
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Encoding a model's external weights into the archive the device loads.

With external weights the import writes the weights to an archive of their own
(`importing.WEIGHTS_ARCHIVE_NAME`) and the MLIR only names them. The compile
then splits off an encoder module (`--iree-parameter-encoder-output-file`): the
work the module would otherwise do on the weights when it starts up (packing,
relayout, other computations on constants), and the weights themselves. Here
the encoder module is compiled for the host CPU and run by
`iree-encode-parameters` on the import's archive. Its output, in the default
`consolidate` mode, is the only archive the module needs at runtime, under
`ENCODED_SCOPE`.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from integration_tests.stages.compiling import CompilationError, linker_flags

# The scope iree-turbine names the weights under at import, as in
# `#flow.parameter.named<"model"::"...">`.
IMPORT_SCOPE = "model"

# The scope the encoded parameters are written to and loaded from, as in
# `--parameters=encoded=<file>`: iree-compile's default
# `--iree-parameter-encoder-output-scope`.
ENCODED_SCOPE = "encoded"

# The encoder module runs on the host. A generic CPU (no
# --iree-llvmcpu-target-cpu) keeps the encoded archive, and so its hash, the
# same on every host.
_ENCODER_TARGET_FLAGS = (
    "--iree-hal-target-device=local",
    "--iree-hal-local-target-device-backends=llvm-cpu",
)


def encoder_output_flags(encoder_mlir: Path) -> list[str]:
    """iree-compile flags that split the encoder module off into `encoder_mlir`."""
    return [f"--iree-parameter-encoder-output-file={encoder_mlir}"]


def _run(argv: list[str], log_file: Path, what: str) -> None:
    result = subprocess.run(argv, capture_output=True, text=True)
    log = result.stdout + result.stderr
    log_file.parent.mkdir(parents=True, exist_ok=True)
    log_file.write_text(log)
    if result.returncode != 0:
        raise CompilationError(
            f"{what} failed with exit code {result.returncode}: see {log_file}",
            log=log,
        )


def encode_parameters(
    iree_compile: Path,
    iree_encode_parameters: Path,
    encoder_mlir: Path,
    weights_archive: Path,
    output_archive: Path,
    linker: Path | None = None,
) -> Path:
    """Compile `encoder_mlir` for the host and run it on `weights_archive`.

    Returns `output_archive`, the encoded parameters. The encoder module and
    its vmfb are deleted afterwards. Failures raise `CompilationError`, as they
    are part of building the module.
    """
    work_dir = output_archive.parent
    encoder_vmfb = work_dir / "encoder.vmfb"
    # A stale archive from an earlier run into the same directory would
    # otherwise pass for this one's.
    output_archive.unlink(missing_ok=True)
    _run(
        [
            str(iree_compile),
            str(encoder_mlir),
            "-o",
            str(encoder_vmfb),
            *_ENCODER_TARGET_FLAGS,
            *(linker_flags(linker) if linker is not None else []),
        ],
        work_dir / "encoder-compile.log",
        "compiling the parameter encoder",
    )
    _run(
        [
            str(iree_encode_parameters),
            f"--module={encoder_vmfb}",
            # Map the encoder module instead of reading it into memory: it
            # can hold constants of the model.
            "--module_mode=mmap",
            f"--parameters={IMPORT_SCOPE}={weights_archive}",
            f"--output={ENCODED_SCOPE}={output_archive}",
        ],
        work_dir / "encode-parameters.log",
        "iree-encode-parameters",
    )
    if not output_archive.exists():
        raise CompilationError(
            f"iree-encode-parameters reported success but wrote no {output_archive}"
        )
    encoder_vmfb.unlink()
    encoder_mlir.unlink()
    return output_archive

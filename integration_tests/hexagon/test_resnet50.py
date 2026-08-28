# Copyright 2026 RooflineAI GmbH
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""End-to-end test: resnet-50 imported from torch, run on the Hexagon DSP."""

from __future__ import annotations

import pathlib

import numpy as np
import pytest

from integration_tests.hexagon.check_outputs import AccuracyError, check_output_files
from integration_tests.hexagon.compile_cases import (
    CompileCase,
    cases_with_known_failures,
    compile_model,
)
from integration_tests.hexagon.hexagon_runtime import DeployedRuntime
from integration_tests.hexagon.run_module import run_module_on_device
from integration_tests.hexagon.torch_model_utils import (
    ImportedModel,
    import_torch_model,
)

_MODEL_ID = "microsoft/resnet-50"

# ImageNet NCHW input. f32 throughout:
# the reference then comes from torch's own f32 path, so any difference is the
# DSP's, not a dtype conversion's.
_INPUT_SHAPE = (1, 3, 224, 224)

# Loose enough that a legitimate difference in accumulation order does not fail,
# tight enough that a wrong kernel does. Measured on this model: relative L2
# error 0.0000%, max abs diff 6.7e-06.
_ATOL = 1e-2
_RTOL = 1e-2

# The "default" case is the one that does *not* pass
# --iree-hexagon-launch-config-selector=hexagon, so it routes through the
# generic upstream LLVMCPU lowering-strategy selector. On this model it runs to
# completion and returns plausible logits, but they are wrong: relative L2 error
# 2.98%, max abs diff 0.83 on logits spanning roughly +-8, spread broadly over
# the output (mean signed diff -0.018, std 0.22) rather than concentrated in a
# few elements. The classification still happens to survive - both torch and the
# device pick class 911 - which is exactly why this needs a numeric check and
# not just a smoke test.
#
# Unresolved: whether that is a bug in the generic path for Hexagon or whether
# the generic path is simply not meant to be numerically usable here. Until that
# is decided the case stays xfail(strict), so the suite is actionable and a fix
# announces itself.
_KNOWN_FAILURES = {
    "default": (
        "generic LLVMCPU launch-config selector gives 2.98% relative L2 error "
        "on resnet-50 logits (measured 2026-08-27); see comment above"
    ),
}


@pytest.fixture(scope="session")
def resnet50(tmp_path_factory: pytest.TempPathFactory) -> ImportedModel:
    """Import resnet-50 once for the whole session.

    Session-scoped because the import is the expensive part (weights load plus
    export, and a ~200 MB MLIR file) while the compile it feeds is seconds.
    """
    torch = pytest.importorskip("torch", reason="torch is needed to import the model")
    transformers = pytest.importorskip(
        "transformers", reason="transformers is needed to build resnet-50"
    )

    model = transformers.AutoModelForImageClassification.from_pretrained(
        _MODEL_ID, dtype=torch.float32
    )
    model.eval()
    torch.manual_seed(0)
    inputs = (torch.randn(*_INPUT_SHAPE, dtype=torch.float32),)

    output_dir = tmp_path_factory.mktemp("resnet50_import")
    imported = import_torch_model(model, inputs, output_dir)
    print(
        f"imported {_MODEL_ID} to {imported.model_mlir} "
        f"({imported.model_mlir.stat().st_size} bytes)"
    )
    return imported


def _top1(logits: np.ndarray) -> tuple[int, float]:
    """Index of the largest logit and its margin over the runner-up."""
    flat = logits.reshape(-1)
    order = np.argsort(flat)[::-1]
    margin = float(flat[order[0]] - flat[order[1]]) if flat.size > 1 else float("inf")
    return int(order[0]), margin


@pytest.mark.hexagon
@pytest.mark.parametrize(
    "compile_case", cases_with_known_failures(_KNOWN_FAILURES, raises=AccuracyError)
)
def test_resnet50_on_device(
    compile_case: CompileCase,
    resnet50: ImportedModel,
    iree_compile: pathlib.Path,
    lld: pathlib.Path | None,
    hexagon_runtime: DeployedRuntime,
    tmp_path: pathlib.Path,
) -> None:
    vmfb = compile_model(
        iree_compile=iree_compile,
        model_mlir=resnet50.model_mlir,
        output_vmfb=tmp_path / "model.vmfb",
        compile_case=compile_case,
        log_file=tmp_path / "iree-compile.log",
        linker=lld,
    )
    print(f"compiled {compile_case.name} -> {vmfb} ({vmfb.stat().st_size} bytes)")

    result = run_module_on_device(
        runtime=hexagon_runtime,
        case_dir_name=f"resnet50_{compile_case.name}",
        module_vmfb=vmfb,
        function=resnet50.function,
        input_files=resnet50.input_files,
        output_names=resnet50.output_names,
        local_output_dir=tmp_path / "device_outputs",
    )
    assert result.exit_code == 0, result.describe()
    print(result.log)

    check_output_files(
        reference_files=resnet50.output_files,
        actual_files=result.output_files,
        atol=_ATOL,
        rtol=_RTOL,
    )

    # The classification itself, which is what the numbers are for. Asserted
    # only when the reference is decisive, so that a future change of input or
    # seed cannot turn a near-tie into a mystery failure.
    reference_top1, margin = _top1(np.load(resnet50.output_files[0]))
    device_top1, _ = _top1(np.load(result.output_files[0]))
    print(
        f"top-1 class: reference {reference_top1}, device {device_top1} (reference margin {margin:.4g})"
    )
    if margin > 10 * _ATOL:
        assert device_top1 == reference_top1, (
            f"device picked class {device_top1}, torch picked {reference_top1} "
            f"(reference margin over runner-up was {margin:.4g})"
        )


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-vv", *sys.argv[1:]]))

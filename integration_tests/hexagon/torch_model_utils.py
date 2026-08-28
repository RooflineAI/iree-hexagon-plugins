# Copyright 2026 RooflineAI GmbH
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Importing a torch model to MLIR, plus its reference outputs.

The exported entry point is named `main`, which is iree-turbine's default.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from iree.turbine import aot
from iree.turbine.aot import extend_aot_decompositions

_aten = torch.ops.aten

# Ops that must be decomposed before the importer sees them, because torch-mlir
# either cannot lower them or lowers them badly.
#
# This is the subset of roof-mlir's `attic_torch/decompositions.py`
# TORCH_DECOMP_EXTENSIONS that a conv net or a transformer block needs. That
# file is 206 lines of accumulated knowledge and the rest of it should be
# ported on demand, when a model actually fails, rather than pre-emptively:
# each entry there carries a reason, and copying reasons we have not hit yet
# would be cargo cult.
#
# One entry deserves its comment repeated, since it is the least guessable:
# convolutions have to reach the importer as the generic `aten.convolution`,
# because the passes that lift them back to conv1d/conv2d/conv3d are only
# implemented for the generic form.
DEFAULT_DECOMPOSITIONS = [
    _aten.conv1d,
    _aten.conv1d.default,
    _aten.conv2d,
    _aten.conv2d.default,
    _aten.conv3d,
    _aten.conv3d.default,
    # hardtanh likewise gets decomposed to a clamp.
    _aten.hardtanh,
    _aten.hardtanh.default,
    # Since torch 2.6 linear is no longer decomposed automatically, and
    # torch-mlir's own decomposition turns it into a batch_matmul with batch
    # size 1 rather than a matmul.
    _aten.linear.default,
    _aten.matmul.default,
    _aten.split,
    _aten.split_with_sizes.default,
    _aten.concat.default,
    # Attention: keep the math form rather than the fused CPU kernels, which
    # have no torch-mlir lowering.
    _aten.scaled_dot_product_attention.default,
    _aten._scaled_dot_product_attention_math.default,
    _aten._scaled_dot_product_flash_attention_for_cpu.default,
    _aten._safe_softmax,
]

# Custom decomposition *implementations* (roof-mlir's ATTIC_CUSTOM_TABLE:
# erfinv, a torch-2.5 split_with_sizes, concat -> cat) are not portable through
# this API: the public extend_aot_decompositions takes a list of ops, while the
# attic fork extended it to take a dict of op -> implementation. If one is ever
# needed, the public route is
# torch.export.export(...).run_decompositions(table) before aot.export.


@dataclass(frozen=True)
class ImportedModel:
    """An imported model and the data needed to run and check it."""

    model_mlir: Path
    input_files: list[Path]
    output_files: list[Path]
    function: str = "main"

    @property
    def output_names(self) -> list[str]:
        return [path.name for path in self.output_files]


def _as_tensor_tuple(outputs: Any) -> tuple[torch.Tensor, ...]:
    if hasattr(outputs, "items"):  # transformers ModelOutput
        outputs = tuple(value for _, value in outputs.items())
    elif not isinstance(outputs, (tuple, list)):
        outputs = (outputs,)
    for output in outputs:
        if not isinstance(output, torch.Tensor):
            raise TypeError(f"model returned a non-tensor output: {type(output)}")
    return tuple(outputs)


def import_torch_model(
    model: torch.nn.Module,
    inputs: tuple[torch.Tensor, ...],
    output_dir: Path,
    decompositions: list[Any] | None = None,
) -> ImportedModel:
    """Export `model` to MLIR and record its inputs and eager outputs.

    Writes `model.mlir`, `input{N}.npy` and `output{N}.npy` into `output_dir`.
    The outputs come from running the model eagerly in torch on the same inputs,
    which is what makes them a reference rather than just another result.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    with extend_aot_decompositions(
        from_current=True,
        add_ops=decompositions
        if decompositions is not None
        else DEFAULT_DECOMPOSITIONS,
    ):
        exported = aot.export(model, args=inputs)

    model_mlir = output_dir / "model.mlir"
    exported.save_mlir(model_mlir)

    input_files = []
    for index, tensor in enumerate(inputs):
        input_file = output_dir / f"input{index}.npy"
        np.save(input_file, tensor.detach().cpu().numpy())
        input_files.append(input_file)

    with torch.no_grad():
        outputs = _as_tensor_tuple(model(*inputs))
    output_files = []
    for index, tensor in enumerate(outputs):
        output_file = output_dir / f"output{index}.npy"
        np.save(output_file, tensor.detach().cpu().numpy())
        output_files.append(output_file)

    return ImportedModel(
        model_mlir=model_mlir,
        input_files=input_files,
        output_files=output_files,
    )

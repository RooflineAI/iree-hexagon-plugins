# Copyright 2026 RooflineAI GmbH
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Report the host-side torch reference of one model, without a device.

The integration tests compare the device output against a reference computed
by torch on the host, in the model's dtype. For float16 that reference depends
on the host CPU (the kernels torch picks), so the same test can pass on one CI
runner and fail on another. This prints the host, the reference's checksum and
values at chosen elements, and how far it is from a float32 reference, so that
runs on different machines can be compared.

    .venv/bin/python -m integration_tests.report_reference \
        --model=google/gemma-4-E2B-it-assistant-patched --save-dir=/tmp/ref
    .venv/bin/python -m integration_tests.report_reference --compare-dir=/tmp/ref
"""

from __future__ import annotations

import argparse
import hashlib
import platform
import sys
from pathlib import Path

import numpy as np
import torch

# Run as a file (`python integration_tests/report_reference.py`), the repo root
# is not on sys.path, so the integration_tests package would not be found.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from integration_tests.modeltree import discover
from integration_tests.stages.importing import (
    _as_tensor_tuple,
    generate_inputs,
    load_model,
)

# The worst elements of the google/gemma-4-E2B-it-assistant-patched failure in
# CI run 40552156845.
_DEFAULT_INDICES = (153019, 151363, 138133, 221674, 223170)
# output0 of that model on the local phone (hexagon-vtcm-hmx, 1 or 4 threads
# alike) at _DEFAULT_INDICES; CI's device output had the same values there.
_DEVICE_OUTPUT_SHA256 = "4f35876c70c44bee"
_DEVICE_OUTPUT = (2.578125, 0.1453857421875, -14.0, -2.64453125, -1.357421875)
_CPU_FLAGS = ("avx2", "avx512f", "avx512_bf16", "avx512_fp16", "amx_tile")


def _host_report() -> list[str]:
    model_name = platform.processor()
    flags: set[str] = set()
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        for line in cpuinfo.read_text().splitlines():
            key, _, value = line.partition(":")
            if key.strip() == "model name":
                model_name = value.strip()
            elif key.strip() == "flags":
                flags = set(value.split())
    present = [flag for flag in _CPU_FLAGS if flag in flags]
    return [
        f"cpu: {model_name}",
        f"cpu flags: {' '.join(present) or '-'}",
        f"torch {torch.__version__}, capability {torch.backends.cpu.get_cpu_capability()}, "
        f"{torch.get_num_threads()} threads, mkldnn {torch.backends.mkldnn.is_available()}",
    ]


def _run(model: torch.nn.Module, inputs: tuple[torch.Tensor, ...]) -> list[np.ndarray]:
    with torch.no_grad():
        return [output.numpy() for output in _as_tensor_tuple(model(*inputs))]


def _describe(name: str, array: np.ndarray, indices: tuple[int, ...]) -> str:
    flat = array.reshape(-1)
    values = ", ".join(f"[{i}]={float(flat[i]):.5g}" for i in indices if i < flat.size)
    digest = hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()[:16]
    return f"{name}: sha256 {digest}, {values}"


def _distance(name: str, actual: np.ndarray, reference: np.ndarray) -> str:
    act = actual.astype(np.float64)
    ref = reference.astype(np.float64)
    max_abs = float(np.abs(act - ref).max())
    relative = float(np.linalg.norm(act - ref) / np.linalg.norm(ref) * 100.0)
    return f"{name}: max abs diff {max_abs:.6g}, relative L2 {relative:.4f}%"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default="google/gemma-4-E2B-it-assistant-patched")
    parser.add_argument(
        "--index",
        type=int,
        action="append",
        help="flat output element to print (repeatable; default: the CI failure's)",
    )
    parser.add_argument(
        "--save-dir", type=Path, help="write the outputs here as output<i>.npy"
    )
    parser.add_argument(
        "--compare-dir",
        type=Path,
        help="a --save-dir from another machine to compare the outputs against",
    )
    args = parser.parse_args()
    indices = tuple(args.index or _DEFAULT_INDICES)

    for line in _host_report():
        print(line)

    spec = discover.select(names=[args.model])[0]
    model = load_model(spec)
    inputs = generate_inputs(spec, model)
    # As the integration test computes it: in the model's dtype.
    references = _run(model, inputs)
    # float32 does not depend on the host the way float16 does.
    model = model.float()
    inputs32 = tuple(t.float() if t.is_floating_point() else t for t in inputs)
    references32 = _run(model, inputs32)

    for i, (reference, reference32) in enumerate(zip(references, references32)):
        name = f"output{i}"
        print(_describe(f"{name} ({reference.dtype})", reference, indices))
        print(_describe(f"{name} (float32)", reference32, indices))
        print(_distance(f"{name} {reference.dtype} vs float32", reference, reference32))
        if args.compare_dir is not None:
            other = np.load(args.compare_dir / f"{name}.npy")
            print(_distance(f"{name} vs {args.compare_dir}", reference, other))
            print(
                f"{name} bitwise equal to {args.compare_dir}: "
                f"{np.array_equal(reference, other)}"
            )
        if args.save_dir is not None:
            args.save_dir.mkdir(parents=True, exist_ok=True)
            np.save(args.save_dir / f"{name}.npy", reference)

    values = ", ".join(
        f"[{i}]={v:.5g}" for i, v in zip(_DEFAULT_INDICES, _DEVICE_OUTPUT)
    )
    print(f"output0 (local device): sha256 {_DEVICE_OUTPUT_SHA256}, {values}")


if __name__ == "__main__":
    main()

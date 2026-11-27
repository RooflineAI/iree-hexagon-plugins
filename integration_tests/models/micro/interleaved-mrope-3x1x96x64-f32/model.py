# Copyright 2026 RooflineAI GmbH
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import torch


class InterleavedMRope(torch.nn.Module):
    """Qwen3-VL's interleaved M-RoPE (`apply_interleaved_mrope` in
    transformers), on a runtime `freqs` of shape [3, batch, positions, 32].

    Each strided slice assignment becomes a dispatch that copies 20 strided
    floats per row, which is what this checks.
    """

    def __init__(self, mrope_section: tuple[int, int, int]) -> None:
        super().__init__()
        self.mrope_section = mrope_section

    def forward(self, freqs: torch.Tensor) -> torch.Tensor:
        # Clone so the input is not written to in place.
        freqs_t = freqs.clone()[0]
        for dim, offset in enumerate((1, 2), start=1):
            idx = slice(offset, self.mrope_section[dim] * 3, 3)
            freqs_t[..., idx] = freqs[dim, ..., idx]
        return freqs_t


def get_model() -> torch.nn.Module:
    return InterleavedMRope(mrope_section=(24, 20, 20))

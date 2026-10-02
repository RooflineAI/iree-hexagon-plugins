# Copyright 2026 RooflineAI GmbH
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""A strided 3x3 conv with an exact (erf) GELU fused in, in f16.

The reduced form of Qwen3-ASR's audio-encoder conv stem
(Qwen/Qwen3-ASR-0.6B), so that the LLVM Hexagon ISel failure it hits is
tracked without importing the whole model. Each part alone compiles: the bare
GELU, and the same conv at batch 1.
"""

import torch
import torch.nn.functional as F


class ConvGelu(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = torch.nn.Conv2d(32, 32, 3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(self.conv(x))


def get_model() -> torch.nn.Module:
    torch.manual_seed(0)
    return ConvGelu().to(torch.float16)

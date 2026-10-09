# Copyright 2026 RooflineAI GmbH
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""A strided 3x3 conv with an exact (erf) GELU fused in, in f16.

Exercises compilation of the fused conv and GELU without importing the full
Qwen3-ASR audio encoder.
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

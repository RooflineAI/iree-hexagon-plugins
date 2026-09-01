# Copyright 2026 RooflineAI GmbH
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import torch
import torch.nn.functional as F


class Ternary(torch.nn.Module):
    """One attention block, straight onto F.scaled_dot_product_attention."""

    def __init__(self, fn) -> None:
        super().__init__()
        self.fn = fn

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ) -> torch.Tensor:
        return self.fn(q, k, v)


def get_model() -> torch.nn.Module:
    return Ternary(F.scaled_dot_product_attention)

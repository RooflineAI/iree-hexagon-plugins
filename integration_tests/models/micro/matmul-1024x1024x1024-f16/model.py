# Copyright 2026 RooflineAI GmbH
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import torch


class Binary(torch.nn.Module):
    """A single op, so a codegen regression points at the op and not at a net."""

    def __init__(self, fn) -> None:
        super().__init__()
        self.fn = fn

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return self.fn(a, b)


def get_model() -> torch.nn.Module:
    return Binary(torch.matmul)

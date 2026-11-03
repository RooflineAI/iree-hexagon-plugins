# Copyright 2026 RooflineAI GmbH
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""`scatter` along a negative dim, as in Gemma-4's masked embedder.

torch-mlir must normalize `dim = -1` before lowering `aten.scatter.src` to
TMTensor. The indices are a fixed buffer without duplicates, so the result
does not depend on the order of the writes.
"""

import torch


class ScatterNegativeDim(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("index", torch.tensor([[0, 3, 5], [1, 2, 7]]))

    def forward(self, base, src):
        return base.scatter(-1, self.index, src)


def get_model() -> torch.nn.Module:
    return ScatterNegativeDim()

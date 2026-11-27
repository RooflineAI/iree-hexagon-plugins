# Copyright 2026 RooflineAI GmbH
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Layer replacements that keep an f16 model accurate once compiled."""

import torch
import torch.nn.functional as F


class F32LayerNorm(torch.nn.LayerNorm):
    """`nn.LayerNorm`, computed in f32.

    iree-compile lowers an f16 aten.layer_norm with f16 accumulation, which
    alone puts Qwen3-VL's merged image embeddings 100% (relative L2) off.
    Computing it in f32 differs from torch's own f16 CPU kernel by at most one
    f16 ulp per norm (measured 2026-10-01). Swap it in with
    `module.__class__ = F32LayerNorm` on a plain `nn.LayerNorm`.
    """

    def forward(self, x):
        return F.layer_norm(
            x.float(),
            self.normalized_shape,
            self.weight.float(),
            self.bias.float(),
            self.eps,
        ).to(x.dtype)

# Copyright 2026 RooflineAI GmbH
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""probs @ V in f32 with an f16 result, as in Qwen3-VL's vision attention.

V is a slice of the f16 qkv buffer, transposed to [heads, seq, head_dim] and
widened to f32 in the same dispatch as the batch matmul, whose result is
truncated to f16 and transposed back. The 4-D matmul keeps that widening out
of the matmul's operand, as in the encoder. The 32-lane fptrunc epilogue
keeps one half of a refilled HVX register pair and redefines the other as
qf32, which Hexagon's post-RA qfloat handling must not convert a second time.
"""

import torch


class ProbsV(torch.nn.Module):
    def forward(self, probs, qkv):
        v = qkv[2].transpose(0, 1).unsqueeze(0).float()
        return (probs @ v).squeeze(0).transpose(0, 1).half()


def get_model() -> torch.nn.Module:
    return ProbsV()

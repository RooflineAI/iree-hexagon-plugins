# Copyright 2026 RooflineAI GmbH
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import torch


class SlicedCacheAttention(torch.nn.Module):
    """Grouped-query attention over the first `context` rows of a KV cache."""

    def __init__(self, context: int) -> None:
        super().__init__()
        self.context = context

    def forward(
        self, query: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor
    ) -> torch.Tensor:
        key = k_cache[:, :, : self.context, :]
        value = v_cache[:, :, : self.context, :]
        # The query heads that share a KV head form the M dim of one matmul.
        batch, q_heads, q_len, head_dim = query.shape
        kv_heads = key.shape[1]
        group = q_heads // kv_heads
        query = query.reshape(batch, kv_heads, group * q_len, head_dim)
        scores = torch.matmul(query, key.transpose(2, 3)).float() * head_dim**-0.5
        weights = torch.softmax(scores, dim=-1).to(query.dtype)
        out = torch.matmul(weights, value)
        out = out.view(batch, kv_heads, group, q_len, head_dim)
        return out.permute(0, 3, 1, 2, 4).reshape(batch, q_len, q_heads, head_dim)


def get_model() -> torch.nn.Module:
    return SlicedCacheAttention(context=64)

# Copyright 2026 RooflineAI GmbH
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""A causal LM exported as `prefill` and `decode`, the KV cache in module state.

  * `prefill(input_ids, cache_position, input_length) -> (next_token, logits)`
    runs the prompt, its length dynamic in multiples of `block_seq_stride`.
    `input_length` is the index of the last real token; padding after it is
    written to the cache but never attended to.
  * `decode(input_ids, cache_position) -> next_token` runs one token at
    position `len(cache_position) - 1` against the cache.

The cache is registered buffers, which the export turns into globals that both
functions share, initially zero. Each function runs from that zero cache, both
in the eager reference and on the device (a separate `iree-run-module` each),
so `decode` is checked against an empty cache rather than the prompt's.
`prefill` reads only the positions it has just written, so the initial cache
does not change its result.

The test harness recognizes a `KvCacheLM` from `get_model()` and exports, runs
and checks both functions (see `stages/importing.py`).

Attention is grouped-query matmuls without `repeat_kv`, and the greedy argmax
is part of the graph. Only full attention is supported.
"""

from __future__ import annotations

from typing import Any

import torch
from transformers import Cache
from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS, eager_mask
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

BATCH_SIZE = 1
# Head room past max_seq_len that the symbolic shapes of the export need.
_CONTEXT_PADDING = 32
GROUPED_QUERY_ATTENTION = "grouped_query_matmul"


def grouped_query_attention(
    module: Any,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float | None = None,
    dropout: float = 0.0,
    **kwargs: Any,
) -> tuple[torch.Tensor, None]:
    """Folds the query heads that share a KV head into the M dim of one matmul.

    Query head `h` attends to KV head `h // group_size`, as with `repeat_kv`.
    Matmuls stay in the model dtype; scaling, mask and softmax run in float32.
    """
    for ignored in ("position_ids", "sliding_window", "use_cache"):
        kwargs.pop(ignored, None)
    if dropout != 0.0 or kwargs:
        raise ValueError(f"unsupported: dropout={dropout}, {sorted(kwargs)}")

    batch_size, num_q_heads, q_len, head_dim = query.shape
    num_kv_heads, kv_len = key.shape[1], key.shape[2]
    group_size = num_q_heads // num_kv_heads
    scale = scaling if scaling is not None else float(head_dim) ** -0.5

    query = query.reshape(batch_size, num_kv_heads, group_size * q_len, head_dim)
    scores = torch.matmul(query, key.transpose(2, 3)).float() * scale
    if attention_mask is not None:
        # Additive [batch, 1, q_len, kv_len], broadcast over heads and group.
        mask = attention_mask[..., :kv_len].unsqueeze(1)
        scores = scores.view(batch_size, num_kv_heads, group_size, q_len, kv_len)
        scores = (scores + mask).view(
            batch_size, num_kv_heads, group_size * q_len, kv_len
        )
    weights = torch.softmax(scores, dim=-1).to(query.dtype)
    out = torch.matmul(weights, value)
    out = out.view(batch_size, num_kv_heads, group_size, q_len, head_dim)
    out = out.permute(0, 3, 1, 2, 4).reshape(batch_size, q_len, num_q_heads, head_dim)
    return out.contiguous(), None


class SliceUpdateKvCache(torch.nn.Module, Cache):
    """A KV cache in registered buffers, written in place at `past_seen_tokens`.

    transformers' own caches branch on Python state, which the export freezes
    at its traced value; here the write position comes from the shape of
    `cache_position`, set by the wrapper before every call.
    """

    def __init__(self, config: Any, context_size: int, dtype: torch.dtype) -> None:
        super().__init__()
        self.past_seen_tokens: Any = 0
        num_kv_heads = getattr(config, "num_key_value_heads", None)
        num_kv_heads = num_kv_heads or config.num_attention_heads
        head_dim = getattr(config, "head_dim", None)
        head_dim = head_dim or config.hidden_size // config.num_attention_heads
        shape = (BATCH_SIZE, num_kv_heads, context_size, head_dim)
        for index in range(config.num_hidden_layers):
            self.register_buffer(f"k_{index}", torch.zeros(shape, dtype=dtype))
            self.register_buffer(f"v_{index}", torch.zeros(shape, dtype=dtype))

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        *args: Any,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        begin = self.past_seen_tokens
        end = begin + key_states.shape[-2]
        k_cache = getattr(self, f"k_{layer_idx}")
        v_cache = getattr(self, f"v_{layer_idx}")
        k_cache[:, :, begin:end, :] = key_states
        v_cache[:, :, begin:end, :] = value_states
        return k_cache[:, :, :end, :], v_cache[:, :, :end, :]

    def get_seq_length(self, layer_idx: int = 0) -> torch.Tensor:
        # A tensor, so that the mask code's `arange + offset` is a tensor add,
        # which the torch importer accepts.
        return torch.tensor(self.past_seen_tokens)

    def get_mask_sizes(self, q_length: int, layer_idx: int) -> tuple[int, int]:
        return q_length, 0


# The model with a kv cache
class KvCacheLM(torch.nn.Module):
    def __init__(
        self, model: torch.nn.Module, max_seq_len: int, block_seq_stride: int
    ) -> None:
        super().__init__()
        config = model.config.get_text_config()
        window = getattr(config, "sliding_window", None)
        layer_types = set(getattr(config, "layer_types", None) or ["full_attention"])
        if layer_types - {"full_attention", "sliding_attention"} or (
            window is not None and window < max_seq_len + _CONTEXT_PADDING
        ):
            raise NotImplementedError(
                f"only full attention: layer types {sorted(layer_types)}, "
                f"sliding_window {window}"
            )
        ALL_ATTENTION_FUNCTIONS[GROUPED_QUERY_ATTENTION] = grouped_query_attention
        ALL_MASK_ATTENTION_FUNCTIONS.register(GROUPED_QUERY_ATTENTION, eager_mask)
        model.set_attn_implementation({"": GROUPED_QUERY_ATTENTION})
        self.model = model
        # For the input generator, which reads config.vocab_size.
        self.config = model.config
        self.max_seq_len = max_seq_len
        self.block_seq_stride = block_seq_stride
        self.kv_cache = SliceUpdateKvCache(
            config, max_seq_len + _CONTEXT_PADDING, model.dtype
        )

    def _logits(
        self, input_ids: torch.Tensor, cache_position: torch.Tensor, mask: Any
    ) -> torch.Tensor:
        self.kv_cache.past_seen_tokens = cache_position.shape[-1] - input_ids.shape[-1]
        return self.model(
            input_ids,
            attention_mask=mask,
            past_key_values=self.kv_cache,
            use_cache=True,
        ).logits

    @torch.no_grad()
    def prefill(
        self,
        input_ids: torch.Tensor,
        cache_position: torch.Tensor,
        input_length: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Real tokens only; transformers adds the causal part.
        positions = torch.arange(input_ids.shape[-1])
        mask = positions[None, :] <= input_length[:, None]
        logits = self._logits(input_ids, cache_position, mask)
        next_token = torch.argmax(logits[0, input_length])[None, None]
        return next_token, logits

    @torch.no_grad()
    def decode(
        self, input_ids: torch.Tensor, cache_position: torch.Tensor
    ) -> torch.Tensor:
        # One new token sees every cached position.
        mask = torch.zeros((input_ids.shape[0], 1, 1, 1), dtype=self.model.dtype)
        logits = self._logits(input_ids, cache_position, mask)
        return torch.argmax(logits[0, 0])[None, None]

    def reset_cache(self) -> None:
        """Zero the cache, the state each function starts from on the device."""
        for buffer in self.kv_cache.buffers():
            buffer.zero_()

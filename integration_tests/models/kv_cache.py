# Copyright 2026 RooflineAI GmbH
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""A causal LM exported as `prefill` and `decode`, the KV cache in module state.

  * `prefill(input_ids, cache_position, input_length) -> (next_token, logits)`
    runs the prompt, its length dynamic in multiples of `block_seq_stride`, at
    positions `len(cache_position) - len(input_ids)` onwards, against the
    cache before them: 0 for a new sequence, or the number of cached tokens to
    continue one with another prompt. `input_length` is the index of the last
    real token in `input_ids`; padding after it is written to the cache but
    never attended to.
  * `decode(input_ids, cache_position) -> next_token` runs one token at
    position `len(cache_position) - 1` against the cache.

Only the lengths of `cache_position` matter, not its values. The cache is
registered buffers, which the export turns into globals that both functions
share, initially zero. Both read only the positions written before or by
themselves, so neither the initial cache nor padding changes their results.

The test harness recognizes a `KvCacheLM` from `get_model()`, exports both
functions (see `stages/importing.py`) and checks them together:
`iree-run-llm` has to generate as `generate_greedy` does.

Attention is grouped-query matmuls without `repeat_kv`, and the greedy argmax
is part of the graph. Only full attention is supported.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
from transformers import Cache
from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS, eager_mask
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

from integration_tests.stages.artifacts import GeneratedTokenInfo

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
        self.past_seen_tokens: int | torch.SymInt = 0
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
        self, input_ids: torch.Tensor, kv_length: int | torch.SymInt, mask: torch.Tensor
    ) -> torch.Tensor:
        """The logits of `input_ids` at the last of `kv_length` positions.

        `input_ids` are written to the cache right before `kv_length`, and
        attend to all cache positions before it as `mask` allows.
        """
        self.kv_cache.past_seen_tokens = kv_length - input_ids.shape[-1]
        return self.model(
            input_ids,
            attention_mask=mask,
            past_key_values=self.kv_cache,
            use_cache=True,
        ).logits

    def _prefill_logits(
        self,
        input_ids: torch.Tensor,
        kv_length: int | torch.SymInt,
        input_length: torch.Tensor,
    ) -> torch.Tensor:
        """`prefill`'s logits, see `prefill`."""
        seq_len = input_ids.shape[-1]
        # For the export, which cannot know these relations of the two dims.
        torch._check(kv_length >= seq_len)
        torch._check(kv_length <= self.max_seq_len)
        # Additive [batch, 1, seq_len, kv_length]: query i, at position
        # start + i, sees the earlier cache positions and the real prompt
        # tokens up to itself, but no padding. Positions are relative to
        # start, so that no tensor is added to a symbolic int, which the torch
        # importer does not accept.
        queries = torch.arange(seq_len)[:, None]
        keys = torch.arange(seq_len - kv_length, seq_len)[None, :]
        allowed = (keys <= queries) & (keys <= input_length[:, None])
        mask = torch.zeros(allowed.shape, dtype=self.model.dtype).masked_fill(
            ~allowed, torch.finfo(self.model.dtype).min
        )
        return self._logits(input_ids, kv_length, mask[None, None])

    @torch.no_grad()
    def prefill(
        self,
        input_ids: torch.Tensor,
        cache_position: torch.Tensor,
        input_length: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self._prefill_logits(input_ids, cache_position.shape[-1], input_length)
        next_token = torch.argmax(logits[0, input_length])[None, None]
        return next_token, logits

    @torch.no_grad()
    def decode(
        self, input_ids: torch.Tensor, cache_position: torch.Tensor
    ) -> torch.Tensor:
        # One new token sees every cached position.
        mask = torch.zeros((input_ids.shape[0], 1, 1, 1), dtype=self.model.dtype)
        logits = self._logits(input_ids, cache_position.shape[-1], mask)
        return torch.argmax(logits[0, 0])[None, None]

    @torch.no_grad()
    def generate_greedy(
        self,
        turns: Sequence[tuple[Sequence[int], int]],
        min_prefill_blocks: int = 2,
        pad_token_id: int = 0,
    ) -> list[tuple[list[GeneratedTokenInfo], torch.Tensor]]:
        """Greedy generation as iree-run-llm runs it, from an empty cache.

        `turns` are (prompt, max_new_tokens) pairs, run one after the other on
        the same cache, as iree-run-llm runs several prompts. Each prompt is
        padded to a multiple of `block_seq_stride` of at least
        `min_prefill_blocks` blocks for `prefill`, at the position the previous
        turn left off at, then `decode` runs once per further token. A turn
        after the first is prefilled with the previous turn's last token in
        front of its prompt: that token was generated but, no decode having run
        on it, is not in the cache yet. Stop tokens are ignored, as with
        iree-run-llm's --ignore_eos, and a turn ends early only at
        `max_seq_len`.

        Returns, per turn, one `GeneratedTokenInfo` per generated token, in
        order, and prefill's logits, `[1, padded length, vocab]` as `prefill`
        returns them. Raises `ValueError` if a padded prefill does not fit in
        `max_seq_len`.
        """
        stride = self.block_seq_stride
        results: list[tuple[list[GeneratedTokenInfo], torch.Tensor]] = []
        # The number of tokens in the cache, and the one generated last.
        position = 0
        pending: int | None = None
        self.reset_cache()
        try:
            for turn, (prompt, max_new_tokens) in enumerate(turns, start=1):
                tokens = list(prompt) if pending is None else [pending, *prompt]
                blocks = max(-(-len(tokens) // stride), min_prefill_blocks)
                padded_length = blocks * stride
                if position + padded_length > self.max_seq_len:
                    raise ValueError(
                        f"turn {turn}: {len(tokens)} tokens, {padded_length} "
                        f"padded, at position {position}, end past "
                        f"max_seq_len={self.max_seq_len}"
                    )
                input_ids = torch.full(
                    (1, padded_length), pad_token_id, dtype=torch.int64
                )
                input_ids[0, : len(tokens)] = torch.tensor(tokens, dtype=torch.int64)
                prefill_logits = self._prefill_logits(
                    input_ids,
                    position + padded_length,
                    torch.tensor([len(tokens) - 1]),
                )
                logits = prefill_logits[0, len(tokens) - 1]
                position += len(tokens)
                generated: list[GeneratedTokenInfo] = []
                while True:
                    top2 = torch.topk(logits.float(), 2)
                    generated.append(
                        GeneratedTokenInfo(
                            selected_id=int(torch.argmax(logits)),
                            runner_up_id=int(top2.indices[1]),
                            margin=float(top2.values[0] - top2.values[1]),
                        )
                    )
                    if len(generated) == max_new_tokens or position >= self.max_seq_len:
                        break
                    mask = torch.zeros((1, 1, 1, 1), dtype=self.model.dtype)
                    logits = self._logits(
                        torch.tensor([[generated[-1].selected_id]]),
                        position + 1,
                        mask,
                    )[0, 0]
                    position += 1
                pending = generated[-1].selected_id
                results.append((generated, prefill_logits))
        finally:
            self.reset_cache()
        return results

    def reset_cache(self) -> None:
        """Zero the cache, the state each function starts from on the device."""
        for buffer in self.kv_cache.buffers():
            buffer.zero_()

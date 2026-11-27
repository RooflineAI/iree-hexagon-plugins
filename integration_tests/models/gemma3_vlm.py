# Copyright 2026 RooflineAI GmbH
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Gemma 3 as one module with three functions: vision, prefill and decode.

As for Qwen3-VL (models/qwen3_vl.py), the vision encoder and the language
model are separate functions of one module:

  * `Gemma3VisionEncoder`: one `side` x `side` image (896 for Gemma 3) in, its
    256 image embeddings out: the SigLIP tower, then the projector's 4x4
    average pool, RMSNorm and projection into the language model's width.
  * `Gemma3KvCacheLM`: the language model as a `KvCacheLM` whose `prefill`
    also takes the image embeddings, plus the encoder as its `vision`
    function. Positions are plain 1-D, the image tokens counted like text.
    Unlike text, the image tokens attend to each other bidirectionally in
    prefill, as `Gemma3Model` does with `token_type_ids`.

The residual stream outgrows f16, so it is carried scaled down (see
`scale_residual`).
"""

from __future__ import annotations

from typing import Any

import torch
from transformers.models.gemma3.modeling_gemma3 import Gemma3RMSNorm

from integration_tests.models.f16_layers import F32LayerNorm
from integration_tests.models.kv_cache import KvCacheLM

# preprocessor_config.json of gemma-3-4b-it: rescale 1/255, then normalize.
_MEAN, _STD = 0.5, 0.5

# The residual stream of gemma-3-4b-it peaks at 101k (|x|, f32, last layer,
# measured 2026-11-05 on a text prompt), above f16's 65504: the f16 forward
# goes inf at layer 28 and every logit is NaN. The f32 peak / 8 = 12.7k leaves
# 5x headroom; a power of two keeps the rescale exact in f16.
RESIDUAL_SCALE = 8.0


class _ResidualRMSNorm(Gemma3RMSNorm):
    # Gemma3RMSNorm.forward, with the output scaled into the residual's range.
    def forward(self, x):
        output = self._norm(x.float()) * (1.0 + self.weight.float())
        return (output / RESIDUAL_SCALE).type_as(x)


def scale_residual(language_model: torch.nn.Module) -> None:
    """Carry the residual stream divided by `RESIDUAL_SCALE`.

    Every read of the residual goes through an RMSNorm, which is
    scale-invariant, so only what writes into it changes: the embedding scale
    and the two post-norms of each layer. Image embeddings that replace token
    embeddings have to be divided by it too.
    """
    language_model.embed_tokens.embed_scale /= RESIDUAL_SCALE
    for layer in language_model.layers:
        layer.post_attention_layernorm.__class__ = _ResidualRMSNorm
        layer.post_feedforward_layernorm.__class__ = _ResidualRMSNorm


class Gemma3VisionEncoder(torch.nn.Module):
    """`Gemma3Model.get_image_features` on one `side` x `side` image, taken
    as RGB in 0..1: the normalization is part of the graph."""

    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.vision_tower = model.model.vision_tower
        self.projector = model.model.multi_modal_projector
        self.side = int(model.config.vision_config.image_size)
        for module in self.vision_tower.modules():
            # See f16_layers.F32LayerNorm: an f16 layer norm compiles to f16
            # accumulation.
            if type(module) is torch.nn.LayerNorm:
                module.__class__ = F32LayerNorm

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor]:
        pixel_values = (image - _MEAN) / _STD
        hidden = self.vision_tower(pixel_values=pixel_values).last_hidden_state
        # [1, tokens, hidden] -> [tokens, hidden], as prefill takes it.
        return (self.projector(hidden)[0],)


class Gemma3KvCacheLM(KvCacheLM):
    """Gemma 3's language model, with the image embeddings as prefill input.

    * `prefill(input_ids, cache_position, input_length, image_embeds)`: the
      i-th image token of the prompt takes the i-th row of `image_embeds`
      ([256, hidden]). Text attends causally; image tokens also attend to
      every other image token.
    * `decode(input_ids, cache_position)`: text only.
    * `vision(image)`: the `Gemma3VisionEncoder` on one image
      ([1, 3, side, side] in 0..1), whose output is what `prefill` takes
      after `input_length`.

    The prompt must have exactly one image, as one contiguous run of image
    tokens, one per embedding.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        encoder: Gemma3VisionEncoder,
        max_seq_len: int,
        block_seq_stride: int,
    ) -> None:
        language_model = model.model.language_model
        scale_residual(language_model)
        # KvCacheLM sets the text model's attention implementation and builds
        # the cache from its config; the vision tower keeps its own.
        super().__init__(language_model, max_seq_len, block_seq_stride)
        self.lm_head = model.lm_head
        self.image_token_id = int(model.config.image_token_id)
        self.num_image_tokens = int(model.config.mm_tokens_per_image)
        self.hidden_size = int(self.config.hidden_size)
        # A submodule: its weights are part of the module, for `vision`.
        self.encoder = encoder

    def _run(
        self, inputs_embeds: torch.Tensor, cache_position: torch.Tensor, mask: Any
    ) -> torch.Tensor:
        self.kv_cache.past_seen_tokens = (
            cache_position.shape[-1] - inputs_embeds.shape[1]
        )
        out = self.model(
            input_ids=None,
            inputs_embeds=inputs_embeds,
            # A prepared 4-D mask for both layer types: the sliding window
            # (1024) is longer than the context, so they are the same.
            attention_mask={"full_attention": mask, "sliding_attention": mask},
            past_key_values=self.kv_cache,
            use_cache=True,
        )
        return self.lm_head(out.last_hidden_state)

    def _prefill_logits(
        self,
        input_ids: torch.Tensor,
        cache_position: torch.Tensor,
        input_length: torch.Tensor,
        *prefill_inputs: torch.Tensor,
    ) -> torch.Tensor:
        (image_embeds,) = prefill_inputs
        embeds = self.model.get_input_embeddings()(input_ids)
        is_image = input_ids[0] == self.image_token_id
        # The i-th image token takes the i-th embedding, as masked_scatter
        # does, with a static shape; the clamp only keeps the gather in range.
        index = (is_image.long().cumsum(-1) - 1).clamp(0, image_embeds.shape[0] - 1)
        gathered = image_embeds.index_select(0, index)[None] / RESIDUAL_SCALE
        embeds = torch.where(is_image[None, :, None], gathered.to(embeds.dtype), embeds)
        # Gemma3Model's prefill mask with one image: causal, or both query and
        # key image tokens; keys only up to the last real token.
        positions = torch.arange(input_ids.shape[-1])
        causal = positions[None, :] <= positions[:, None]
        allowed = (causal | (is_image[:, None] & is_image[None, :])) & (
            positions[None, :] <= input_length
        )
        mask = torch.where(
            allowed,
            torch.zeros((), dtype=embeds.dtype),
            torch.tensor(torch.finfo(embeds.dtype).min, dtype=embeds.dtype),
        )[None, None]
        return self._run(embeds, cache_position, mask)

    def _decode_logits(
        self, input_ids: torch.Tensor, cache_position: torch.Tensor
    ) -> torch.Tensor:
        embeds = self.model.get_input_embeddings()(input_ids)
        # One new token sees every cached position.
        mask = torch.zeros((input_ids.shape[0], 1, 1, 1), dtype=self.model.dtype)
        return self._run(embeds, cache_position, mask)

    def example_prefill_inputs(
        self,
    ) -> tuple[tuple[torch.Tensor, ...], dict[str, dict[int, Any]]]:
        # One fixed-size image: static.
        embeds = torch.zeros(
            self.num_image_tokens, self.hidden_size, dtype=self.model.dtype
        )
        return (embeds,), {"image_embeds": {}}

    @torch.no_grad()
    def prefill(
        self,
        input_ids: torch.Tensor,
        cache_position: torch.Tensor,
        input_length: torch.Tensor,
        image_embeds: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self._prefill_logits(
            input_ids, cache_position, input_length, image_embeds
        )
        next_token = torch.argmax(logits[0, input_length])[None, None]
        return next_token, logits

    @torch.no_grad()
    def vision(self, image: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """The encoder's output for `image` ([1, 3, side, side] in 0..1):
        what `prefill` takes after `input_length`."""
        return tuple(self.encoder(image))

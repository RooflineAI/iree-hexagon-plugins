# Copyright 2026 RooflineAI GmbH
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Qwen3-VL split into the two halves the device runs: vision encoder and LM.

The whole multimodal forward does not export as one graph (its shapes come
from the image grid through Python loops), so the halves are separate modules:

  * `VisionEncoder`: image in, the four tensors the language model consumes
    out, plus the merged image grid. One fixed resolution for now: the grid
    tensors transformers derives from the image size are baked in.
  * `Qwen3VLKvCacheLM`: the language model as a `KvCacheLM` whose `prefill`
    also takes the encoder's outputs. Unlike the encoder, it is not tied to one
    image size: the number of image tokens is dynamic and the M-RoPE positions
    are computed in the graph from the grid.

iree-run-llm chains the two on the device (`--vision_function`), or takes the
encoder's outputs from files (`--prefill_input`).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from transformers.models.qwen3_vl import modeling_qwen3_vl as mq
from transformers.vision_utils import (
    get_vision_cu_seqlens,
    get_vision_interpolation_indices_and_weights,
    get_vision_position_ids,
)

from integration_tests.models.kv_cache import KvCacheLM

# The chat template's placeholder for one image, which the prompt has once and
# which is expanded to one token per merged patch.
IMAGE_PAD = "<|image_pad|>"

# preprocessor_config.json of Qwen3-VL-2B/4B-Instruct.
_PATCH, _MERGE, _TEMPORAL, _MEAN, _STD = 16, 2, 2, 0.5, 0.5
# Image sides must be a multiple of this: one merged patch.
IMAGE_FACTOR = _PATCH * _MERGE

# ===----------------------------------------------------------------------=== #
# Vision encoder
# ===----------------------------------------------------------------------=== #


def _single_segment_attention(
    self, hidden_states, cu_seqlens, position_embeddings=None, **kwargs
):
    # Qwen3VLVisionAttention.forward for one image. Upstream splits q/k/v per
    # image with torch.split(..., cu_seqlens.tolist()); with one image that is
    # a no-op, but exported it becomes an index gather that segfaults
    # torch-mlir's ConvertAtenUnflattenIntOp (the same crash as Qwen3-ASR).
    seq_length = hidden_states.shape[0]
    q, k, v = (
        self.qkv(hidden_states)
        .reshape(seq_length, 3, self.num_heads, -1)
        .permute(1, 0, 2, 3)
        .unbind(0)
    )
    cos, sin = position_embeddings
    q, k = mq.apply_rotary_pos_emb_vision(q, k, cos, sin)
    q, k, v = (t.transpose(0, 1).unsqueeze(0) for t in (q, k, v))
    attention = mq.ALL_ATTENTION_FUNCTIONS.get_interface(
        self.config._attn_implementation, mq.eager_attention_forward
    )
    out = attention(
        self,
        q,
        k,
        v,
        attention_mask=None,
        scaling=self.scaling,
        dropout=0.0,
        is_causal=False,
    )[0]
    return self.proj(out.reshape(seq_length, -1).contiguous())


class _F32LayerNorm(torch.nn.LayerNorm):
    # iree-compile lowers an f16 aten.layer_norm with f16 accumulation, which
    # alone puts the merged embeddings 100% (relative L2) off. Computing it in
    # f32 differs from torch's own f16 CPU kernel by at most one f16 ulp per
    # norm (measured 2026-10-01); the single-segment attention is bit-identical.
    def forward(self, x):
        return F.layer_norm(
            x.float(),
            self.normalized_shape,
            self.weight.float(),
            self.bias.float(),
            self.eps,
        ).to(x.dtype)


def patch_vision_tower(visual: torch.nn.Module) -> None:
    """Make the vision tower exportable and accurate in f16 (see above)."""
    for module in visual.modules():
        if isinstance(module, mq.Qwen3VLVisionAttention):
            module.forward = _single_segment_attention.__get__(module)
        elif type(module) is torch.nn.LayerNorm:
            module.__class__ = _F32LayerNorm


class VisionEncoder(torch.nn.Module):
    """The vision tower on one `side` x `side` image.

    Outputs the merged image embeddings, the three deepstack features (each
    [tokens, hidden]) and the merged grid `[h, w]` (i64; tokens = h * w),
    which `Qwen3VLKvCacheLM.prefill` takes in this order.
    """

    def __init__(self, visual: torch.nn.Module, side: int) -> None:
        super().__init__()
        if side % IMAGE_FACTOR:
            raise ValueError(
                f"the image side {side} is not a multiple of {IMAGE_FACTOR}"
            )
        self.visual = visual
        self.side = side
        grid = torch.tensor([[1, side // _PATCH, side // _PATCH]])
        indices, weights = get_vision_interpolation_indices_and_weights(
            grid,
            num_grid_per_side=visual.num_grid_per_side,
            mode=visual.interpolation_mode,
            align_corners=visual.interpolation_align_corners,
            spatial_merge_size=visual.config.spatial_merge_size,
        )
        self.register_buffer("grid", grid, persistent=False)
        self.register_buffer("interp_indices", indices, persistent=False)
        self.register_buffer("interp_weights", weights, persistent=False)
        self.register_buffer(
            "position_ids",
            get_vision_position_ids(grid, visual.spatial_merge_size),
            persistent=False,
        )
        self.register_buffer(
            "cu_seqlens", get_vision_cu_seqlens(grid), persistent=False
        )
        merged = side // IMAGE_FACTOR
        self.register_buffer(
            "merged_grid", torch.tensor([merged, merged]), persistent=False
        )

    def forward(self, image: torch.Tensor):
        # image: [1, 3, H, W] in 0..1. The image processor's rescale is the
        # suite's /255; normalize and patchify follow
        # Qwen2VLImageProcessorPil.patchify: (gh, gw, mh, mw, C, t, ph, pw).
        x = (image[0] - _MEAN) / _STD
        c, g = x.shape[0], self.side // _PATCH
        x = x.reshape(c, g // _MERGE, _MERGE, _PATCH, g // _MERGE, _MERGE, _PATCH)
        x = x.permute(1, 4, 2, 5, 0, 3, 6).unsqueeze(5)
        x = x.expand(-1, -1, -1, -1, -1, _TEMPORAL, -1, -1)
        pixel_values = x.reshape(g * g, c * _TEMPORAL * _PATCH * _PATCH)
        out = self.visual(
            pixel_values,
            grid_thw=self.grid,
            interp_indices=self.interp_indices,
            interp_weights=self.interp_weights,
            position_ids=self.position_ids,
            cu_seqlens=self.cu_seqlens,
            return_dict=True,
        )
        # A copy, so that the export returns a value rather than a buffer.
        return (*(out.pooler_output, *out.deepstack_features), self.merged_grid + 0)


def load_image(path: Path, dtype: torch.dtype) -> torch.Tensor:
    """An image file as the encoder takes it: [1, 3, H, W], RGB in 0..1.

    The same decoding as the suite's `image` input generator, so that the
    encoder's own test and the generation test see the same pixels.
    """
    from PIL import Image

    array = np.asarray(Image.open(path).convert("RGB"))
    tensor = torch.from_numpy(array.astype(np.float32) / 255.0)
    return tensor.permute(2, 0, 1).unsqueeze(0).to(dtype)


# ===----------------------------------------------------------------------=== #
# Language model
# ===----------------------------------------------------------------------=== #


def static_deepstack(self, hidden_states, visual_pos_masks, visual_embeds):
    # Qwen3VLTextModel._deepstack_process does h[mask, :] += embeds. Its
    # data-dependent shape exports, but iree-turbine's importer rejects the
    # size assertion it carries ("Unimplemented call_function: target=ge").
    # Same values, static shape: gather the i-th embedding for the i-th masked
    # position (batch 1).
    visual_embeds = visual_embeds.to(hidden_states.dtype)
    index = (visual_pos_masks.long().cumsum(-1) - 1).clamp(min=0)
    gathered = visual_embeds.index_select(0, index[0]).unsqueeze(0)
    mask = visual_pos_masks.unsqueeze(-1)
    return hidden_states + torch.where(mask, gathered, torch.zeros_like(gathered))


def mrope_positions(
    input_ids: torch.Tensor,
    image_token_id: int,
    num_image_tokens: Any,
    image_grid: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Qwen3-VL's 3-D (t, h, w) positions of a prompt with at most one image.

    `get_rope_index` in closed form, for batch 1 and one image whose
    `num_image_tokens` (= h * w) tokens are contiguous in `input_ids`, starting
    at `s`, with merged grid `image_grid = [h, w]`:

      * text before the image: (i, i, i)
      * image token k:         (s, s + k // w, s + k % w)
      * text after the image:  i - h * w + max(h, w), on all three axes

    so decode continues at its cache position plus
    `rope_delta = max(h, w) - h * w`. Without an image token in the prompt,
    every position is (i, i, i) and the delta is 0. Elementwise ops only, so
    the prompt length and the image size can both be dynamic.

    Returns `position_ids [3, 1, P]` and `rope_delta [1]`.
    """
    is_image = input_ids[0] == image_token_id
    index = torch.arange(input_ids.shape[-1])
    has_image = is_image.any()
    # argmax returns the first maximal index: where the image starts.
    start = torch.argmax(is_image.to(torch.int32))
    h, w = image_grid[0], image_grid[1]
    delta = torch.maximum(h, w) - num_image_tokens
    after = has_image & (index >= start + num_image_tokens)
    text = torch.where(after, index + delta, index)
    k = (index - start).clamp(min=0)
    t_axis = torch.where(is_image, start, text)
    h_axis = torch.where(is_image, start + k // w, text)
    w_axis = torch.where(is_image, start + k % w, text)
    position_ids = torch.stack([t_axis, h_axis, w_axis])[:, None, :]
    rope_delta = torch.where(has_image, delta, torch.zeros_like(delta))
    return position_ids, rope_delta.reshape(1)


class Qwen3VLKvCacheLM(KvCacheLM):
    """Qwen3-VL's language model, with the image features as prefill inputs.

      * `prefill(input_ids, cache_position, input_length, image_embeds,
        deepstack0, deepstack1, deepstack2, image_grid)`: the image embeddings
        replace the prompt's image tokens, the deepstack features are added
        after the first three decoder layers, and the M-RoPE positions come
        from `mrope_positions`, whose `rope_delta` prefill keeps in module
        state.
      * `decode(input_ids, cache_position)`: text only, at the cache position
        plus `rope_delta` on all three axes.

    The vision tower is not part of the module: `image_features` runs it
    eagerly, for the reference.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        encoder: VisionEncoder,
        max_seq_len: int,
        block_seq_stride: int,
    ) -> None:
        # KvCacheLM sets the text model's attention implementation and builds
        # the cache from its config; the vision tower keeps its own.
        language_model = model.model.language_model
        language_model._deepstack_process = static_deepstack.__get__(language_model)
        super().__init__(language_model, max_seq_len, block_seq_stride)
        self.lm_head = model.lm_head
        self.image_token_id = int(model.config.image_token_id)
        self.hidden_size = int(self.config.hidden_size)
        self.register_buffer("rope_delta", torch.zeros(1, dtype=torch.int64))
        # Not a submodule: its weights must stay out of the exported module.
        self.__dict__["encoder"] = encoder

    def _run(
        self,
        inputs_embeds: torch.Tensor,
        cache_position: torch.Tensor,
        mask: Any,
        position_ids: torch.Tensor,
        visual_pos_masks: torch.Tensor | None = None,
        deepstack: list[torch.Tensor] | None = None,
    ) -> torch.Tensor:
        self.kv_cache.past_seen_tokens = (
            cache_position.shape[-1] - inputs_embeds.shape[1]
        )
        out = self.model(
            input_ids=None,
            inputs_embeds=inputs_embeds,
            attention_mask=mask,
            position_ids=position_ids,
            past_key_values=self.kv_cache,
            use_cache=True,
            visual_pos_masks=visual_pos_masks,
            deepstack_visual_embeds=deepstack,
        )
        return self.lm_head(out.last_hidden_state)

    def _prefill_logits(
        self,
        input_ids: torch.Tensor,
        cache_position: torch.Tensor,
        input_length: torch.Tensor,
        *prefill_inputs: torch.Tensor,
    ) -> torch.Tensor:
        image_embeds, deepstack0, deepstack1, deepstack2, image_grid = prefill_inputs
        num_image_tokens = image_embeds.shape[0]
        embeds = self.model.get_input_embeddings()(input_ids)
        is_image = (input_ids == self.image_token_id)[..., None]
        # The i-th image token takes the i-th embedding, as masked_scatter
        # does, with a static shape. The clamp keeps a prompt with more image
        # tokens than embeddings in bounds.
        index = (is_image[0, :, 0].long().cumsum(-1) - 1).clamp(
            min=0, max=num_image_tokens - 1
        )
        gathered = image_embeds.to(embeds.dtype).index_select(0, index)[None]
        embeds = torch.where(is_image, gathered, embeds)
        position_ids, rope_delta = mrope_positions(
            input_ids, self.image_token_id, num_image_tokens, image_grid
        )
        # Not copy_: iree-turbine materializes a module global only where a
        # function reads the buffer, and a store to one prefill never reads
        # fails to import ("Cannot store value to unmapped global"). This reads
        # it, and is exact for an integer.
        self.rope_delta.mul_(0).add_(rope_delta)
        positions = torch.arange(input_ids.shape[-1])
        mask = positions[None, :] <= input_length[:, None]
        return self._run(
            embeds,
            cache_position,
            mask,
            position_ids,
            visual_pos_masks=is_image[..., 0],
            deepstack=[deepstack0, deepstack1, deepstack2],
        )

    def _decode_logits(
        self, input_ids: torch.Tensor, cache_position: torch.Tensor
    ) -> torch.Tensor:
        embeds = self.model.get_input_embeddings()(input_ids)
        position = cache_position[-1:] + self.rope_delta
        position_ids = position.view(1, 1, 1).expand(3, 1, 1)
        mask = torch.zeros((input_ids.shape[0], 1, 1, 1), dtype=self.model.dtype)
        return self._run(embeds, cache_position, mask, position_ids)

    def example_prefill_inputs(
        self,
    ) -> tuple[tuple[torch.Tensor, ...], dict[str, dict[int, Any]]]:
        # 64 tokens, a 256x256 image; any count up to max_seq_len exports.
        tokens = torch.export.Dim("image_tokens", min=2, max=self.max_seq_len)
        features = torch.zeros(64, self.hidden_size, dtype=self.model.dtype)
        args = (features, features, features, features, torch.tensor([8, 8]))
        shapes = {
            "image_embeds": {0: tokens},
            "deepstack0": {0: tokens},
            "deepstack1": {0: tokens},
            "deepstack2": {0: tokens},
            "image_grid": {},
        }
        return args, shapes

    @torch.no_grad()
    def prefill(
        self,
        input_ids: torch.Tensor,
        cache_position: torch.Tensor,
        input_length: torch.Tensor,
        image_embeds: torch.Tensor,
        deepstack0: torch.Tensor,
        deepstack1: torch.Tensor,
        deepstack2: torch.Tensor,
        image_grid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self._prefill_logits(
            input_ids,
            cache_position,
            input_length,
            image_embeds,
            deepstack0,
            deepstack1,
            deepstack2,
            image_grid,
        )
        next_token = torch.argmax(logits[0, input_length])[None, None]
        return next_token, logits

    @torch.no_grad()
    def image_features(self, image: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """The encoder's outputs for `image` ([1, 3, H, W] in 0..1), eagerly:
        what `prefill` takes after `input_length`."""
        return tuple(self.encoder(image))

    def expand_prompt(self, prompt: str, num_image_tokens: int) -> str:
        """`prompt` with its one image placeholder expanded to one token per
        image embedding, as the processor does."""
        if prompt.count(IMAGE_PAD) != 1:
            raise ValueError(f"the prompt needs exactly one {IMAGE_PAD}: {prompt!r}")
        return prompt.replace(IMAGE_PAD, IMAGE_PAD * num_image_tokens)

    def reset_cache(self) -> None:
        super().reset_cache()
        self.rope_delta.zero_()

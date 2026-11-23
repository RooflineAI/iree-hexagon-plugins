# Copyright 2026 RooflineAI GmbH
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Qwen3-VL split into the two halves the device runs: vision encoder and LM.

The whole multimodal forward does not export as one graph (its shapes come
from the image grid through Python loops), so the halves are separate modules:

  * `VisionEncoder`: image in, the four tensors the language model consumes
    out, plus the merged image grid. Any image whose sides are multiples of
    `IMAGE_FACTOR` (32): the tensors transformers derives from the grid in
    Python are computed in the graph from the image's shape.
  * `Qwen3VLKvCacheLM`: the language model as a `KvCacheLM` whose `prefill`
    also takes the encoder's outputs: the number of image tokens is dynamic and
    the M-RoPE positions are computed in the graph from the grid.

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

from integration_tests.models.kv_cache import KvCacheLM

# The chat template's placeholder for one image, which the prompt has once and
# which is expanded to one token per merged patch.
IMAGE_PAD = "<|image_pad|>"

# preprocessor_config.json of Qwen3-VL-2B/4B-Instruct.
_PATCH, _MERGE, _TEMPORAL, _MEAN, _STD = 16, 2, 2, 0.5, 0.5
# Image sides must be a multiple of this: one merged patch.
IMAGE_FACTOR = _PATCH * _MERGE
# The largest image side the encoder is exported for, in pixels. 1024x1024 is
# 1024 image tokens, which Qwen/Qwen3-VL-2B-Instruct-kv-cache's max_seq_len is
# sized for; a non-square image has to stay within that token count too.
MAX_IMAGE_SIDE = 1024

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


def _grid_tensors(
    visual: torch.nn.Module,
    grid_h: torch.Tensor,
    grid_w: torch.Tensor,
    num_patches: Any,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The position-embedding interpolation indices and weights, and the
    rotary position ids, of one image's `grid_h` x `grid_w` patch grid: what
    transformers' get_vision_interpolation_indices_and_weights and
    get_vision_position_ids compute (for one image, bilinear, align_corners)
    with Python loops over the grid, here as tensor ops so that the grid can
    be dynamic.

    `grid_h`, `grid_w` are 0-d i64 tensors; `num_patches` (= their product) may
    be a symbolic size, used only as a shape. Every value is computed from
    tensors: arithmetic between a tensor and a symbolic size imports as
    invalid IR (see mrope_positions).
    """
    if (
        visual.interpolation_mode != "bilinear"
        or not visual.interpolation_align_corners
    ):
        raise NotImplementedError("only bilinear, align_corners=True interpolation")
    m = visual.config.spatial_merge_size
    side = visual.num_grid_per_side
    # Patches come in spatial-merge-block order: (block_row, block_col,
    # in_row, in_col), in_col fastest.
    index = torch.arange(num_patches)
    blocks_w = grid_w // m
    in_col = index % m
    in_row = (index // m) % m
    block_col = (index // (m * m)) % blocks_w
    block_row = index // (m * m) // blocks_w
    row = block_row * m + in_row
    col = block_col * m + in_col

    def taps(position: torch.Tensor, size: torch.Tensor):
        # Bilinear taps into the side x side learned table, align_corners:
        # position p of `size` maps to p * (side - 1) / (size - 1).
        source = position.float() * (side - 1) / torch.clamp(size - 1, min=1).float()
        floor = torch.floor(source)
        offsets = torch.arange(2)
        indices = (floor.long()[:, None] + offsets).clamp(0, side - 1)
        weights = (1 - (source[:, None] - floor[:, None] - offsets).abs()).clamp(min=0)
        return indices, weights

    h_taps, h_weights = taps(row, grid_h)
    w_taps, w_weights = taps(col, grid_w)
    indices = (h_taps[:, :, None] * side + w_taps[:, None, :]).reshape(-1, 4)
    weights = (h_weights[:, :, None] * w_weights[:, None, :]).reshape(-1, 4)
    return indices, weights, torch.stack([row, col], dim=-1)


class VisionEncoder(torch.nn.Module):
    """The vision tower on one image, [1, 3, H, W] RGB in 0..1, with H and W
    multiples of `IMAGE_FACTOR` (the caller resizes, as the image processor
    does).

    Outputs the merged image embeddings, the three deepstack features (each
    [tokens, hidden], tokens = H/32 * W/32) and the merged grid
    `[H/32, W/32]` (i64), which `Qwen3VLKvCacheLM.prefill` takes in this order.
    """

    def __init__(self, visual: torch.nn.Module) -> None:
        super().__init__()
        self.visual = visual

    @staticmethod
    def export_dynamic_shapes() -> dict[str, dict[int, Any]]:
        """The export's dynamic dims: H and W, multiples of IMAGE_FACTOR, from
        2 * IMAGE_FACTOR up to MAX_IMAGE_SIDE."""
        blocks = MAX_IMAGE_SIDE // IMAGE_FACTOR
        height = torch.export.Dim("image_height_blocks", min=2, max=blocks)
        width = torch.export.Dim("image_width_blocks", min=2, max=blocks)
        return {"image": {2: IMAGE_FACTOR * height, 3: IMAGE_FACTOR * width}}

    def forward(self, image: torch.Tensor):
        # The image processor's rescale is the suite's /255; normalize and
        # patchify follow Qwen2VLImageProcessorPil.patchify:
        # (gh, gw, mh, mw, C, t, ph, pw).
        _, c, height, width = image.shape
        gh, gw = height // _PATCH, width // _PATCH
        x = (image[0] - _MEAN) / _STD
        x = x.reshape(c, gh // _MERGE, _MERGE, _PATCH, gw // _MERGE, _MERGE, _PATCH)
        x = x.permute(1, 4, 2, 5, 0, 3, 6).unsqueeze(5)
        x = x.expand(-1, -1, -1, -1, -1, _TEMPORAL, -1, -1)
        pixel_values = x.reshape(gh * gw, c * _TEMPORAL * _PATCH * _PATCH)
        # The patch grid as tensors, the only form the importer takes for
        # arithmetic on it.
        grid_h = torch.ops.aten.scalar_tensor(gh, dtype=torch.int64)
        grid_w = torch.ops.aten.scalar_tensor(gw, dtype=torch.int64)
        indices, weights, position_ids = _grid_tensors(
            self.visual, grid_h, grid_w, gh * gw
        )
        one = torch.ones((), dtype=torch.int64)
        out = self.visual(
            pixel_values,
            grid_thw=torch.stack([one, grid_h, grid_w])[None],
            interp_indices=indices,
            interp_weights=weights,
            position_ids=position_ids,
            # One image: a single attention segment (the patched attention
            # ignores it; transformers still reads it).
            cu_seqlens=torch.stack([one - 1, grid_h * grid_w]).to(torch.int32),
            return_dict=True,
        )
        merged_grid = torch.stack([grid_h // _MERGE, grid_w // _MERGE])
        return (out.pooler_output, *out.deepstack_features, merged_grid)


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
    image_grid: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Qwen3-VL's 3-D (t, h, w) positions of a prompt with at most one image.

    `get_rope_index` in closed form, for batch 1 and one image with merged
    grid `image_grid = [h, w]`, whose h * w tokens are contiguous in
    `input_ids`, starting at `s`:

      * text before the image: (i, i, i)
      * image token k:         (s, s + k // w, s + k % w)
      * text after the image:  i - h * w + max(h, w), on all three axes

    so decode continues at its cache position plus
    `rope_delta = max(h, w) - h * w`. Without an image token in the prompt,
    every position is (i, i, i) and the delta is 0. Elementwise ops only, so
    the prompt length and the image size can both be dynamic. Everything
    derives from tensors: arithmetic between a tensor and a symbolic size
    (e.g. the number of image embeddings) imports as an invalid
    `torch.aten.sub.Tensor` with a `!torch.int` operand.

    Returns `position_ids [3, 1, P]` and `rope_delta [1]`.
    """
    is_image = input_ids[0] == image_token_id
    index = torch.arange(input_ids.shape[-1])
    has_image = is_image.any()
    # argmax returns the first maximal index: where the image starts.
    start = torch.argmax(is_image.to(torch.int32))
    h, w = image_grid[0], image_grid[1]
    num_image_tokens = h * w
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
        # iree-run-llm --image_pad_token: the device expands the prompt's one
        # placeholder itself, from the image grid.
        self.image_pad_token = IMAGE_PAD
        self.image_factor = IMAGE_FACTOR
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
        embeds = self.model.get_input_embeddings()(input_ids)
        is_image = (input_ids == self.image_token_id)[..., None]
        # The i-th image token takes the i-th embedding, as masked_scatter
        # does, with a static shape. The bounds keep a prompt with more image
        # tokens than the grid has in bounds (see mrope_positions on why they
        # come from a tensor).
        last = image_grid[0] * image_grid[1] - 1
        index = torch.minimum(
            (is_image[0, :, 0].long().cumsum(-1) - 1).clamp(min=0), last
        )
        gathered = image_embeds.to(embeds.dtype).index_select(0, index)[None]
        embeds = torch.where(is_image, gathered, embeds)
        position_ids, rope_delta = mrope_positions(
            input_ids, self.image_token_id, image_grid
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
        # One tensor per argument: torch.export treats example inputs that
        # alias each other as one value, so passing the same tensor four times
        # makes every feature argument read the same (the last) input.
        features = [
            torch.zeros(64, self.hidden_size, dtype=self.model.dtype) for _ in range(4)
        ]
        args = (*features, torch.tensor([8, 8]))
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

# Copyright 2026 RooflineAI GmbH
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Qwen3-VL-2B-Instruct's vision encoder, one 256x256 image.

The other half of Qwen/Qwen3-VL-2B-Instruct, whose entry runs only the
language model. Image in, the four tensors the language model consumes out:
the merged image embeddings and the three deepstack features, each [64, 2048].

The whole multimodal forward does not export as one graph (its shapes come
from the image grid through Python loops), so the encoder is tested on its own
at a fixed resolution. transformers anticipates this: every grid-derived
tensor can be precomputed (transformers/vision_utils.py) and is baked in here.
"""

import torch
import torch.nn.functional as F
from transformers import Qwen3VLForConditionalGeneration
from transformers.models.qwen3_vl import modeling_qwen3_vl as mq
from transformers.vision_utils import (
    get_vision_cu_seqlens,
    get_vision_interpolation_indices_and_weights,
    get_vision_position_ids,
)

_MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"
_REVISION = "89644892e4d85e24eaac8bacfd4f463576704203"
_SIDE = 256  # pixels; a multiple of patch_size * merge_size, so the processor would not resize it
# preprocessor_config.json of the pinned revision.
_PATCH, _MERGE, _TEMPORAL, _MEAN, _STD = 16, 2, 2, 0.5, 0.5


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


class _VisionEncoder(torch.nn.Module):
    def __init__(self, visual: torch.nn.Module) -> None:
        super().__init__()
        self.visual = visual
        grid = torch.tensor([[1, _SIDE // _PATCH, _SIDE // _PATCH]])
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

    def forward(self, image: torch.Tensor):
        # image: [1, 3, H, W] in 0..1. The image processor's rescale is the
        # suite's /255; normalize and patchify follow
        # Qwen2VLImageProcessorPil.patchify: (gh, gw, mh, mw, C, t, ph, pw).
        x = (image[0] - _MEAN) / _STD
        c, g = x.shape[0], _SIDE // _PATCH
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
        return (out.pooler_output, *out.deepstack_features)


def get_model() -> torch.nn.Module:
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        _MODEL_ID,
        revision=_REVISION,
        dtype=torch.float16,
        attn_implementation="sdpa",
    )
    model.to(dtype=torch.float16)
    visual = model.model.visual
    for module in visual.modules():
        if isinstance(module, mq.Qwen3VLVisionAttention):
            module.forward = _single_segment_attention.__get__(module)
        elif type(module) is torch.nn.LayerNorm:
            module.__class__ = _F32LayerNorm
    return _VisionEncoder(visual)

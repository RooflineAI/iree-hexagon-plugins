# Copyright 2026 RooflineAI GmbH
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Qwen3-VL-2B-Instruct exported for image + text token generation.

The language model as a `Qwen3VLKvCacheLM` (models/qwen3_vl.py): `prefill`
takes the vision encoder's outputs, `decode` continues from its KV cache. The
encoder is the Qwen/Qwen3-VL-2B-Instruct-vision-encoder-patched entry, which
the generation test compiles as the vision module; here it only computes the
reference's image features, eagerly.
"""

from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from transformers import Qwen3VLForConditionalGeneration

from integration_tests.models.qwen3_vl import (
    Qwen3VLKvCacheLM,
    VisionEncoder,
    patch_vision_tower,
)

_MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"
_REVISION = "89644892e4d85e24eaac8bacfd4f463576704203"
# The vision encoder entry's resolution.
_SIDE = 256


def get_model() -> torch.nn.Module:
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        _MODEL_ID,
        revision=_REVISION,
        dtype=torch.float16,
        attn_implementation="sdpa",
    )
    model.to(dtype=torch.float16)
    # KvCacheLM passes a cache to every call; this makes the config say so.
    model.config.get_text_config().use_cache = True
    visual = model.model.visual
    patch_vision_tower(visual)
    # A 256x256 image is 64 tokens of the ~80-token prompt; 256 leaves room for
    # larger images and for 100+ generated tokens.
    return Qwen3VLKvCacheLM(
        model,
        VisionEncoder(visual, _SIDE),
        max_seq_len=256,
        block_seq_stride=32,
    )


def get_tokenizer_json() -> Path:
    """The model's tokenizer, for the `generation` check."""
    return Path(hf_hub_download(_MODEL_ID, "tokenizer.json", revision=_REVISION))

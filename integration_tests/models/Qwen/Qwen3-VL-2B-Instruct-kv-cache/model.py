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
# Context length. Sized for images up to 1024x1024 pixels: 32 x 32 = 1024
# image tokens (one per 32x32 pixels after the encoder's 2x2 merge), plus 352
# for the chat template, the question and the answer. The bound is a choice,
# not a model limit (Qwen3-VL accepts larger images); lower it if the KV cache
# (28 layers x 2 x 8 heads x (1376 + 32) x 128 x f16 = 161 MB) or the export's
# dynamic-shape range becomes a problem on the device. iree-run-llm has to be
# told the same value (--max_seq_len).
#
# Not 1280: then hexagon-vtcm-hmx fails to compile an attention matmul,
# whose HMX result rows (1280 + 32 cache positions) are not known to be
# 128-byte aligned ("iree_hexagon.hmx.unpack op destination must be known to
# be 128-byte aligned"). 1248 and 1376 (caches of 1280 and 1408, multiples of
# 128) compile, and so does 256 (measured 2026-11-23, 4-layer model); check
# the compile before picking another value.
_MAX_SEQ_LEN = 1376


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
    return Qwen3VLKvCacheLM(
        model,
        VisionEncoder(visual),
        max_seq_len=_MAX_SEQ_LEN,
        block_seq_stride=32,
    )


def get_tokenizer_json() -> Path:
    """The model's tokenizer, for the `generation` check."""
    return Path(hf_hub_download(_MODEL_ID, "tokenizer.json", revision=_REVISION))

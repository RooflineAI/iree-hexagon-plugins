# Copyright 2026 RooflineAI GmbH
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""gemma-3-4b-it exported for image + text token generation.

One module, a `Gemma3KvCacheLM` (models/gemma3_vlm.py): `vision` runs the
SigLIP encoder on one 896x896 image, `prefill` takes its 256 embeddings, and
`decode` continues from the KV cache.
"""

from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from transformers import Gemma3ForConditionalGeneration

from integration_tests.models.gemma3_vlm import Gemma3KvCacheLM, Gemma3VisionEncoder
from integration_tests.models.vlm_preprocessor import Fit, Gemma3Processor

_MODEL_ID = "google/gemma-3-4b-it"
_REVISION = "093f9f388b31de276ce2de164bdc2081324b9767"
# config.json: vision_config.image_size, mm_tokens_per_image.
_SIDE = 896
_IMAGE_TOKENS = 256


def get_model() -> torch.nn.Module:
    model = Gemma3ForConditionalGeneration.from_pretrained(
        _MODEL_ID,
        revision=_REVISION,
        dtype=torch.float16,
        attn_implementation="sdpa",
    )
    # from_pretrained leaves some buffers in f32.
    model.to(dtype=torch.float16)
    # KvCacheLM passes a cache to every call; this makes the config say so.
    model.config.get_text_config().use_cache = True
    # A 276-token prompt (256 image tokens) pads to 288; 512 leaves room for
    # the generated tokens and longer prompts, below the sliding window
    # (1024).
    return Gemma3KvCacheLM(
        model,
        Gemma3VisionEncoder(model),
        max_seq_len=512,
        block_seq_stride=32,
    )


def get_vlm_preprocessor(fit: Fit = "stretch") -> Gemma3Processor:
    """Image and prompt as the exported modules take them, for `generation`
    and integration_tests/tools/preprocess_vlm.py."""
    return Gemma3Processor(_SIDE, _IMAGE_TOKENS, fit)


def get_tokenizer_json() -> Path:
    """The model's tokenizer, for the `generation` check."""
    return Path(hf_hub_download(_MODEL_ID, "tokenizer.json", revision=_REVISION))

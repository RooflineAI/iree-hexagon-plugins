# Copyright 2026 RooflineAI GmbH
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Qwen3-VL-2B-Instruct's vision encoder, one 256x256 image.

The other half of Qwen/Qwen3-VL-2B-Instruct, whose entry runs only the
language model. Image in, the four tensors the language model consumes out:
the merged image embeddings and the three deepstack features, each [64, 2048],
plus the merged grid [8, 8]. See models/qwen3_vl.py.

Qwen/Qwen3-VL-2B-Instruct-kv-cache compiles this entry as its vision module.
"""

import torch
from transformers import Qwen3VLForConditionalGeneration

from integration_tests.models.qwen3_vl import VisionEncoder, patch_vision_tower

_MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"
_REVISION = "89644892e4d85e24eaac8bacfd4f463576704203"
# Pixels; a multiple of patch_size * merge_size, so the processor would not
# resize it.
_SIDE = 256


def get_model() -> torch.nn.Module:
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        _MODEL_ID,
        revision=_REVISION,
        dtype=torch.float16,
        attn_implementation="sdpa",
    )
    model.to(dtype=torch.float16)
    visual = model.model.visual
    patch_vision_tower(visual)
    return VisionEncoder(visual, _SIDE)

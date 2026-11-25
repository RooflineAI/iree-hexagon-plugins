# Copyright 2026 RooflineAI GmbH
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM

from integration_tests.models.kv_cache import KvCacheLM

_MODEL_ID = "Qwen/Qwen3-4B"
_REVISION = "1cfa9a7208912126459214e8b04321603b3df60c"


def get_model() -> torch.nn.Module:
    model = AutoModelForCausalLM.from_pretrained(
        _MODEL_ID,
        revision=_REVISION,
        # KvCacheLM passes a cache to every call; this makes the config say so.
        use_cache=True,
        dtype=torch.float16,
    )
    # from_pretrained leaves some buffers in f32; the export has to see a
    # uniformly f16 module or the reference and the compiled module disagree on
    # dtypes.
    model.to(dtype=torch.float16)
    return KvCacheLM(model, max_seq_len=128, block_seq_stride=32)


def get_tokenizer_json() -> Path:
    """The model's tokenizer, for the `generation` check."""
    return Path(hf_hub_download(_MODEL_ID, "tokenizer.json", revision=_REVISION))

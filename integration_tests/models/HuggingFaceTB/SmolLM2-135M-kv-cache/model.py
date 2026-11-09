# Copyright 2026 RooflineAI GmbH
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import torch
from transformers import AutoModelForCausalLM

from integration_tests.models.kv_cache import KvCacheLM

_MODEL_ID = "HuggingFaceTB/SmolLM2-135M"
_REVISION = "93efa2f097d58c2a74874c7e644dbc9b0cee75a2"


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

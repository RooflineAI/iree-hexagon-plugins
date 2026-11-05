# Copyright 2026 RooflineAI GmbH
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import torch
from transformers import AutoModelForCausalLM

_MODEL_ID = "Qwen/Qwen3-4B-Thinking-2507"
_REVISION = "768f209d9ea81521153ed38c47d515654e938aea"


def get_model() -> torch.nn.Module:
    model = AutoModelForCausalLM.from_pretrained(
        _MODEL_ID,
        revision=_REVISION,
        # No KV cache: one static prefill, which is what this suite compiles.
        use_cache=False,
        dtype=torch.float16,
        # sdpa keeps attention in a form the decomposition list handles.
        attn_implementation="sdpa",
    )
    # from_pretrained leaves some buffers in f32; the export has to see a
    # uniformly f16 module or the reference and the compiled module disagree on
    # dtypes.
    model.to(dtype=torch.float16)
    return model

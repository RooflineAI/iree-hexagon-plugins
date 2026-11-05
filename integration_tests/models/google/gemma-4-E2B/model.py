# Copyright 2026 RooflineAI GmbH
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import torch
from transformers import AutoModelForCausalLM

_MODEL_ID = "google/gemma-4-E2B"
_REVISION = "433813a669639fd7eaa83155828e9000ed357d53"


class _TextPrefill(torch.nn.Module):
    # The test calls the module with positional inputs, but the multimodal
    # forward takes pixel_values second; pass the text inputs by keyword.
    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask):
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            # No KV cache: one static prefill.
            use_cache=False,
        ).logits


def get_model() -> torch.nn.Module:
    model = AutoModelForCausalLM.from_pretrained(
        _MODEL_ID,
        revision=_REVISION,
        dtype=torch.float16,
        # sdpa keeps attention in a form the decomposition list handles.
        attn_implementation="sdpa",
    )
    # from_pretrained leaves some buffers in f32; the export has to see a
    # uniformly f16 module or the reference and the compiled module disagree on
    # dtypes.
    model.to(dtype=torch.float16)
    return _TextPrefill(model)

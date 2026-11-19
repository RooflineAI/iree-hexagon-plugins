# Copyright 2026 RooflineAI GmbH
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Gemma 3 4B's language model, one static text prefill.

The residual stream outgrows f16: in f32 it peaks at 101k (|x|, last layer;
measured 2026-11-05 on the suite's input), above f16's 65504, so the f16
forward goes inf at layer 28 and every logit is NaN. Every read of the residual
goes through an RMSNorm, which is scale-invariant, so the residual is carried
divided by _RESIDUAL_SCALE: the embedding scale and the two post-norms that
write into it are divided by it, and nothing else changes. A power of two keeps
that rescale exact in f16.
"""

import torch
from transformers import AutoModelForCausalLM
from transformers.models.gemma3.modeling_gemma3 import Gemma3RMSNorm

_MODEL_ID = "google/gemma-3-4b-it"
_REVISION = "093f9f388b31de276ce2de164bdc2081324b9767"

# The f32 peak / 8 = 12.7k leaves 5x headroom below f16's max.
_RESIDUAL_SCALE = 8.0


class _ResidualRMSNorm(Gemma3RMSNorm):
    # Gemma3RMSNorm.forward, with the output scaled into the residual's range.
    def forward(self, x):
        output = self._norm(x.float()) * (1.0 + self.weight.float())
        return (output / _RESIDUAL_SCALE).type_as(x)


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
    text_model = model.model.language_model
    text_model.embed_tokens.embed_scale /= _RESIDUAL_SCALE
    for layer in text_model.layers:
        layer.post_attention_layernorm.__class__ = _ResidualRMSNorm
        layer.post_feedforward_layernorm.__class__ = _ResidualRMSNorm
    return _TextPrefill(model)

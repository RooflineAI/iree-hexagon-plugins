# Copyright 2026 RooflineAI GmbH
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import types

import torch
from transformers import AutoModelForCausalLM

_MODEL_ID = "microsoft/Phi-4-mini-instruct"
_REVISION = "cfbefacb99257ffa30c83adab238a50856ac3083"


def _sliced_mlp_forward(self, hidden_states):
    # Phi3MLP.forward. Upstream splits the fused gate_up_proj output with
    # .chunk(2, dim=-1), which the decompositions turn into aten.as_strided,
    # and iree-compile fails on it ("failed to legalize operation
    # 'torch.constant.int'", the as_strided strides). Same values, as two
    # slices, which is how Phi3Attention already splits its fused qkv.
    up_states = self.gate_up_proj(hidden_states)
    half = up_states.shape[-1] // 2
    gate, up_states = up_states[..., :half], up_states[..., half:]
    return self.down_proj(up_states * self.activation_fn(gate))


def get_model() -> torch.nn.Module:
    model = AutoModelForCausalLM.from_pretrained(
        _MODEL_ID,
        revision=_REVISION,
        # No KV cache: one static prefill.
        use_cache=False,
        dtype=torch.float16,
        # sdpa keeps attention in a form the decomposition list handles.
        attn_implementation="sdpa",
    )
    # from_pretrained leaves some buffers in f32; the export has to see a
    # uniformly f16 module or the reference and the compiled module disagree on
    # dtypes.
    model.to(dtype=torch.float16)
    # longrope picks its short or long frequencies on every call, from
    # `max(position_ids) + 1 > original_max_position_embeddings` (4096): a
    # data-dependent branch torch.export cannot trace. A 128-token prefill
    # always takes the short ones, which the module already holds from init,
    # so skip the update; cos/sin are bit-identical (checked 2026-11-05).
    model.model.rotary_emb.rope_type = "default"
    for layer in model.model.layers:
        layer.mlp.forward = types.MethodType(_sliced_mlp_forward, layer.mlp)
    return model

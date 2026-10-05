# Copyright 2026 RooflineAI GmbH
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Gemma 4 E2B multi-token-prediction drafter, one draft step.

The drafter is not a standalone LM: it consumes the backbone's hidden state
(concatenated with the last token's embedding, 2 * 1536 wide) and the
backbone's last-layer KV for each attention type. model.yaml feeds those as
seeded normals: q_len 1 over kv_len 128. Output is the drafter's logits.

The masked embedder's forward is replaced, on this instance, by an equivalent that export
and iree-compile can handle (bit-identical in eager torch):
  * the mask value stays a 0-d tensor instead of a Python float from `.item()`;
  * the final `scatter` becomes an `index_put` - `scatter.src` segfaults
    iree-compile in torch-mlir's TorchToTMTensor
    (convertTorchScatterIndexAndSrcToTMScatterIndexAndSrc), on every target.
"""

import types

import torch
from transformers import AutoModelForCausalLM

_MODEL_ID = "google/gemma-4-E2B-it-assistant"
_REVISION = "2d874ef7d29f9a30599a1e4b3c1cbc9595f005df"


def _masked_embedder_forward(self, hidden_states, lm_head_weight):
    batch, seq_len = hidden_states.shape[:2]
    centroid_logits = self.centroids(hidden_states)
    _, top_k_indices = torch.topk(
        centroid_logits, k=self.centroid_intermediate_top_k, dim=-1
    )
    per_cluster = self.token_ordering.long().view(
        self.num_centroids, self.vocab_size_per_centroid
    )
    selected_canonical = per_cluster[top_k_indices]
    selected_embeddings = lm_head_weight[selected_canonical.reshape(-1)].view(
        batch,
        seq_len,
        self.centroid_intermediate_top_k * self.vocab_size_per_centroid,
        self.hidden_size,
    )
    selected_logits = (
        hidden_states.unsqueeze(-2) @ selected_embeddings.transpose(-1, -2)
    ).squeeze(-2)
    mask_value = selected_logits.min() - 1.0
    output = mask_value.expand(batch, seq_len, self.vocab_size).clone()
    idx = selected_canonical.view(batch, seq_len, -1)
    b = torch.arange(batch).view(batch, 1, 1).expand_as(idx)
    s = torch.arange(seq_len).view(1, seq_len, 1).expand_as(idx)
    return output.index_put((b, s, idx), selected_logits)


class _Drafter(torch.nn.Module):
    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, inputs_embeds, full_k, full_v, swa_k, swa_v, position_ids):
        return self.model(
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            shared_kv_states={
                "full_attention": (full_k, full_v),
                "sliding_attention": (swa_k, swa_v),
            },
        ).logits


def get_model() -> torch.nn.Module:
    model = AutoModelForCausalLM.from_pretrained(
        _MODEL_ID,
        revision=_REVISION,
        dtype=torch.float16,
        attn_implementation="sdpa",
    )
    model.to(dtype=torch.float16)
    model.masked_embedding.forward = types.MethodType(
        _masked_embedder_forward, model.masked_embedding
    )
    return _Drafter(model)

# Copyright 2026 RooflineAI GmbH
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Deterministic Gemma-4 E2B text components, without downloading weights.

Keep E2B hidden, head, MLP and per-layer-input dimensions. Reduce the vocabulary
and sliding window so small tests exercise lookup and window boundaries. These
are operator/composition checks, not pretrained-model or generation validation.
"""

import torch
from transformers import Gemma4TextConfig
from transformers.masking_utils import (
    create_causal_mask,
    create_sliding_window_causal_mask,
)
from transformers.models.gemma4.modeling_gemma4 import (
    Gemma4RMSNorm,
    Gemma4TextAttention,
    Gemma4TextDecoderLayer,
    Gemma4TextMLP,
    Gemma4TextModel,
    Gemma4TextRotaryEmbedding,
    Gemma4TextScaledWordEmbedding,
)


def config() -> Gemma4TextConfig:
    cfg = Gemma4TextConfig(
        vocab_size=1024,
        vocab_size_per_layer_input=1024,
        hidden_size=1536,
        intermediate_size=6144,
        num_hidden_layers=35,
        num_attention_heads=8,
        num_key_value_heads=1,
        head_dim=256,
        global_head_dim=512,
        hidden_size_per_layer_input=256,
        num_kv_shared_layers=20,
        use_double_wide_mlp=True,
        hidden_activation="gelu_pytorch_tanh",
        layer_types=(["sliding_attention"] * 4 + ["full_attention"]) * 7,
        sliding_window=64,
        max_position_embeddings=131072,
        rope_parameters={
            "sliding_attention": {"rope_type": "default", "rope_theta": 10000.0},
            "full_attention": {
                "rope_type": "proportional",
                "rope_theta": 1000000.0,
                "partial_rotary_factor": 0.25,
            },
        },
        final_logit_softcapping=30.0,
        pad_token_id=0,
        use_cache=False,
    )
    cfg._attn_implementation = "sdpa"
    return cfg


class Embeddings(torch.nn.Module):
    """Token embeddings and the two sources of per-layer input, all 35 slots."""

    def __init__(self, cfg):
        super().__init__()
        self.config = cfg
        self.hidden_size_per_layer_input = cfg.hidden_size_per_layer_input
        width = cfg.num_hidden_layers * cfg.hidden_size_per_layer_input
        self.embed_tokens = Gemma4TextScaledWordEmbedding(
            cfg.vocab_size,
            cfg.hidden_size,
            cfg.pad_token_id,
            embed_scale=cfg.hidden_size**0.5,
        )
        self.embed_tokens_per_layer = Gemma4TextScaledWordEmbedding(
            cfg.vocab_size_per_layer_input,
            width,
            cfg.pad_token_id,
            embed_scale=cfg.hidden_size_per_layer_input**0.5,
        )
        self.per_layer_model_projection = torch.nn.Linear(
            cfg.hidden_size, width, bias=False
        )
        self.per_layer_model_projection_scale = cfg.hidden_size**-0.5
        self.per_layer_projection_norm = Gemma4RMSNorm(
            cfg.hidden_size_per_layer_input, cfg.rms_norm_eps
        )
        self.per_layer_input_scale = 2.0**-0.5

    def forward(self, ids):
        hidden = self.embed_tokens(ids)
        identity = Gemma4TextModel.get_per_layer_inputs(self, ids, hidden)
        ple = Gemma4TextModel.project_per_layer_inputs(self, hidden, identity)
        return hidden, ple


class AttentionInputs(torch.nn.Module):
    def __init__(self, cfg, layer_idx):
        super().__init__()
        self.config = cfg
        self.kind = cfg.layer_types[layer_idx]
        self.rotary = Gemma4TextRotaryEmbedding(cfg)

    def forward(self, hidden, padding):
        positions = torch.arange(hidden.shape[1], device=hidden.device).unsqueeze(0)
        make_mask = (
            create_sliding_window_causal_mask
            if self.kind == "sliding_attention"
            else create_causal_mask
        )
        mask = make_mask(
            config=self.config,
            inputs_embeds=hidden,
            attention_mask=padding,
            past_key_values=None,
            position_ids=positions,
        )
        return self.rotary(hidden, positions, self.kind), mask


class Attention(torch.nn.Module):
    """Q/K/V projection, RMSNorm, RoPE, dynamic mask, SDPA and output projection."""

    def __init__(self, cfg, layer_idx):
        super().__init__()
        self.inputs = AttentionInputs(cfg, layer_idx)
        self.norm = Gemma4RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.attention = Gemma4TextAttention(cfg, layer_idx)
        # Layers 13 and 14 write KV for all following layers of their type.
        assert self.attention.store_full_length_kv

    def forward(self, hidden, padding):
        positions, mask = self.inputs(hidden, padding)
        shared = {}
        result = self.attention(self.norm(hidden), positions, mask, shared)[0]
        k, v = shared[self.attention.layer_type]
        return result, k, v


class MLPs(torch.nn.Module):
    """Regular and double-width gated GELU MLPs on identical normalized input."""

    def __init__(self, cfg):
        super().__init__()
        self.norm = Gemma4RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.regular = Gemma4TextMLP(cfg, 0)
        self.wide = Gemma4TextMLP(cfg, 15)

    def forward(self, hidden):
        normalized = self.norm(hidden)
        return self.regular(normalized), self.wide(normalized)


class Decoder(torch.nn.Module):
    """Attention, MLP, residuals and per-layer-input gating in one fused graph."""

    def __init__(self, cfg, layer_idx):
        super().__init__()
        self.inputs = AttentionInputs(cfg, layer_idx)
        self.layer = Gemma4TextDecoderLayer(cfg, layer_idx)

    def forward(self, hidden, ple, padding, *kv):
        positions, mask = self.inputs(hidden, padding)
        shared = {self.layer.self_attn.layer_type: (kv[0], kv[1])} if kv else {}
        return self.layer(
            hidden,
            per_layer_input=ple,
            shared_kv_states=shared,
            position_embeddings=positions,
            attention_mask=mask,
        )


class Head(torch.nn.Module):
    """Final normalization, vocabulary projection and logit soft cap."""

    def __init__(self, cfg):
        super().__init__()
        self.norm = Gemma4RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.projection = torch.nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
        self.cap = cfg.final_logit_softcapping

    def forward(self, hidden):
        logits = self.projection(self.norm(hidden))
        return torch.tanh(logits / self.cap) * self.cap


def get_component(name: str) -> torch.nn.Module:
    # Avoid affecting the suite's other deterministic models.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        cfg = config()
        factories = {
            "embeddings": lambda: Embeddings(cfg),
            "attention-sliding": lambda: Attention(cfg, 13),
            "attention-full": lambda: Attention(cfg, 14),
            "mlps": lambda: MLPs(cfg),
            "decoder": lambda: Decoder(cfg, 0),
            "decoder-full": lambda: Decoder(cfg, 4),
            "decoder-shared-kv": lambda: Decoder(cfg, 15),
            "decoder-shared-full": lambda: Decoder(cfg, 19),
            "head": lambda: Head(cfg),
        }
        module = factories[name]()
        # Use the pretrained architecture's initialization scale. RMSNorm
        # retains its unit scale; avoid a degenerate all-zero network.
        for child in module.modules():
            if isinstance(child, (torch.nn.Linear, torch.nn.Embedding)):
                torch.nn.init.normal_(child.weight, mean=0.0, std=cfg.initializer_range)
                if (
                    isinstance(child, torch.nn.Embedding)
                    and child.padding_idx is not None
                ):
                    with torch.no_grad():
                        child.weight[child.padding_idx].zero_()
        return module.eval().requires_grad_(False).to(torch.float16)

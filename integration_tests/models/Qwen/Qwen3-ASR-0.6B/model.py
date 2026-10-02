# Copyright 2026 RooflineAI GmbH
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Qwen3-ASR-0.6B: audio encoder + LM, one prefill over a fixed 4 s clip.

Uses Qwen/Qwen3-ASR-0.6B-hf, the transformers-native conversion of
Qwen/Qwen3-ASR-0.6B (the original is in `qwen-asr` package format).

The prompt (a transcription request, language forced to English, with one
audio placeholder per encoder output frame) is built once by the model's own
processor and baked in, because the suite's input generators cannot place
audio tokens. The test inputs are the log-mel features [1, 128, 400] and their
all-valid mask [1, 400].

Instance-level patches, all bit-identical in eager torch on this clip:
  * the encoder's window bookkeeping (cu_seqlens, valid post-CNN positions) is
    computed once from an all-valid 400-frame mask and frozen - torch.export
    cannot trace it, it depends on mask values;
  * the attention skips the `split` into windows when there is one window: the
    export of that split produces a torch.aten.unflatten.int that segfaults
    torch-mlir's ConvertAtenUnflattenIntOp on every target;
  * the audio-placeholder count check is dropped (data-dependent, and the
    processor guarantees it).
"""

import types

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoProcessor, Qwen3ASRForConditionalGeneration
from transformers.models.qwen3_asr import modeling_qwen3_asr as mq

_MODEL_ID = "Qwen/Qwen3-ASR-0.6B-hf"
_REVISION = "7f1569a48a89f3e3f4dc3a5c9d28bddd903bc76c"
_SAMPLE_RATE = 16000
_SECONDS = 4
_FRAMES = 400  # mel frames for _SECONDS of audio; must match model.yaml


def _static_encoder(encoder: torch.nn.Module) -> None:
    mask = torch.ones(1, _FRAMES, dtype=torch.long)
    chunk_len = encoder.n_window * 2
    num_chunks = _FRAMES // chunk_len
    chunk_lengths = mask.view(1, num_chunks, chunk_len).sum(-1).reshape(-1)
    cu = mq.get_audio_cu_seqlens(
        chunk_lengths, mask.sum(-1), encoder.n_window_infer, encoder.n_window
    )
    split_lengths = (cu[1:] - cu[:-1]).tolist()

    def attention_forward(self, hidden_states, cu_seqlens, max_seqlen=None, **kwargs):
        seq_length, _ = hidden_states.size()
        q, k, v = (
            proj(hidden_states)
            .reshape(seq_length, self.num_heads, -1)
            .transpose(0, 1)
            .unsqueeze(0)
            for proj in (self.q_proj, self.k_proj, self.v_proj)
        )
        if len(split_lengths) == 1:
            out = F.scaled_dot_product_attention(q, k, v, scale=self.scaling)
        else:
            out = torch.cat(
                [
                    F.scaled_dot_product_attention(qq, kk, vv, scale=self.scaling)
                    for qq, kk, vv in zip(
                        *(t.split(split_lengths, 2) for t in (q, k, v))
                    )
                ],
                dim=2,
            )
        return self.out_proj(out.transpose(1, 2).reshape(seq_length, -1).contiguous())

    def encoder_forward(self, input_features, input_features_mask, **kwargs):
        bsz, mels, _ = input_features.shape
        x = (
            input_features.view(bsz, mels, num_chunks, chunk_len)
            .permute(0, 2, 1, 3)
            .reshape(bsz * num_chunks, 1, mels, chunk_len)
        )
        x = F.gelu(self.conv2d1(x))
        x = F.gelu(self.conv2d2(x))
        x = F.gelu(self.conv2d3(x))
        chunks, channels, freq, steps = x.size()
        x = self.conv_out(
            x.permute(0, 3, 1, 2).contiguous().view(chunks, steps, channels * freq)
        )
        x = x + self.positional_embedding.positional_embedding[:steps].to(x.dtype)
        # A full mask keeps every post-CNN position, in order.
        h = x.reshape(-1, x.shape[-1])
        for layer in self.layers:
            h = layer(h, None, max_seqlen=None)[0]
        return mq.BaseModelOutputWithPooling(last_hidden_state=self.ln_post(h))

    encoder.forward = types.MethodType(encoder_forward, encoder)
    for layer in encoder.layers:
        layer.self_attn.forward = types.MethodType(attention_forward, layer.self_attn)


class _Transcribe(torch.nn.Module):
    def __init__(self, model: torch.nn.Module, input_ids: torch.Tensor) -> None:
        super().__init__()
        self.model = model
        self.register_buffer("input_ids", input_ids, persistent=False)

    def forward(self, input_features: torch.Tensor, input_features_mask: torch.Tensor):
        return self.model(
            input_ids=self.input_ids,
            attention_mask=torch.ones_like(self.input_ids),
            input_features=input_features,
            input_features_mask=input_features_mask,
            use_cache=False,
        ).logits


def get_model() -> torch.nn.Module:
    processor = AutoProcessor.from_pretrained(_MODEL_ID, revision=_REVISION)
    prompt = processor.apply_transcription_request(
        audio=np.zeros(_SECONDS * _SAMPLE_RATE, dtype=np.float32), language="English"
    )
    if prompt["input_features"].shape[-1] != _FRAMES:
        raise ValueError(
            f"expected {_FRAMES} mel frames, got {prompt['input_features'].shape}"
        )

    model = Qwen3ASRForConditionalGeneration.from_pretrained(
        _MODEL_ID, revision=_REVISION, dtype=torch.float16, attn_implementation="sdpa"
    )
    model.to(dtype=torch.float16)
    _static_encoder(model.model.audio_tower)
    model.model.get_placeholder_mask = types.MethodType(
        lambda self, input_ids, inputs_embeds, audio_features: (
            (input_ids == self.config.audio_token_id)
            .unsqueeze(-1)
            .expand_as(inputs_embeds)
        ),
        model.model,
    )
    return _Transcribe(model, prompt["input_ids"].to(torch.int64))

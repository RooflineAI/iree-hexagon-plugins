# Copyright 2026 RooflineAI GmbH
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Qwen3-VL-2B-Instruct's language model on a prompt with one image in it.

Unlike Qwen/Qwen3-1.7B (text-only prefill), this exercises what is specific to
the VL language model: the image embeddings scattered into the prompt's
<|image_pad|> slots, the deepstack features added to the hidden states after the
first three decoder layers, and real 3-D M-RoPE positions (the 64 image tokens
sit on an 8x8 grid).

The inputs are the four tensors the vision encoder produces
(Qwen/Qwen3-VL-2B-Instruct-vision-encoder): the merged image embeddings and the
three deepstack features. The token ids and their positions are baked in: random
token ids would contain no image slots, and get_rope_index is a Python loop over
the token stream.
"""

import types

import torch
from transformers import AutoTokenizer, Qwen3VLForConditionalGeneration

_MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"
_REVISION = "89644892e4d85e24eaac8bacfd4f463576704203"
# One 256x256 image: a (1, 16, 16) patch grid, 64 tokens after the 2x2 merge.
_GRID = (1, 16, 16)
_PROMPT = "Describe this image in one sentence."


def _static_deepstack(self, hidden_states, visual_pos_masks, visual_embeds):
    # Qwen3VLTextModel._deepstack_process does h[mask, :] += embeds. Its
    # data-dependent shape exports, but iree-turbine's importer rejects the
    # size assertion it carries ("Unimplemented call_function: target=ge").
    # Same values, static shape: gather the i-th embedding for the i-th masked
    # position (batch 1).
    visual_embeds = visual_embeds.to(hidden_states.dtype)
    index = (visual_pos_masks.long().cumsum(-1) - 1).clamp(min=0)
    gathered = visual_embeds.index_select(0, index[0]).unsqueeze(0)
    mask = visual_pos_masks.unsqueeze(-1)
    return hidden_states + torch.where(mask, gathered, torch.zeros_like(gathered))


class _ImagePrefill(torch.nn.Module):
    def __init__(self, model, input_ids, position_ids) -> None:
        super().__init__()
        self.model = model
        self.register_buffer("input_ids", input_ids, persistent=False)
        self.register_buffer(
            "attention_mask", torch.ones_like(input_ids), persistent=False
        )
        self.register_buffer("position_ids", position_ids, persistent=False)
        self.register_buffer(
            "image_mask", input_ids == model.config.image_token_id, persistent=False
        )

    def forward(self, image_embeds, deepstack0, deepstack1, deepstack2):
        inner = self.model.model
        embeds = inner.get_input_embeddings()(self.input_ids)
        mask = self.image_mask.unsqueeze(-1).expand_as(embeds)
        embeds = embeds.masked_scatter(mask, image_embeds.to(embeds.dtype))
        out = inner.language_model(
            input_ids=None,
            inputs_embeds=embeds,
            attention_mask=self.attention_mask,
            position_ids=self.position_ids,
            visual_pos_masks=self.image_mask,
            deepstack_visual_embeds=[deepstack0, deepstack1, deepstack2],
            use_cache=False,
        )
        return self.model.lm_head(out.last_hidden_state)


def _prompt(model) -> tuple[torch.Tensor, torch.Tensor]:
    """The chat-template prompt with its image slots, and its M-RoPE positions."""
    tokenizer = AutoTokenizer.from_pretrained(_MODEL_ID, revision=_REVISION)
    messages = [
        {
            "role": "user",
            "content": [{"type": "image"}, {"type": "text", "text": _PROMPT}],
        }
    ]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    n_image = (
        _GRID[0]
        * _GRID[1]
        * _GRID[2]
        // model.config.vision_config.spatial_merge_size**2
    )
    text = text.replace("<|image_pad|>", "<|image_pad|>" * n_image)
    input_ids = tokenizer(text, return_tensors="pt")["input_ids"]
    mm_token_type_ids = (input_ids == model.config.image_token_id).long()
    with torch.no_grad():
        position_ids, _ = model.model.get_rope_index(
            input_ids,
            mm_token_type_ids,
            image_grid_thw=torch.tensor([_GRID]),
            attention_mask=torch.ones_like(input_ids),
        )
    return input_ids, position_ids


def get_model() -> torch.nn.Module:
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        _MODEL_ID,
        revision=_REVISION,
        dtype=torch.float16,
        attn_implementation="sdpa",
    )
    model.to(dtype=torch.float16)
    text_model = model.model.language_model
    text_model._deepstack_process = types.MethodType(_static_deepstack, text_model)
    input_ids, position_ids = _prompt(model)
    return _ImagePrefill(model, input_ids, position_ids)

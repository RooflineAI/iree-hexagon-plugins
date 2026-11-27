# Copyright 2026 RooflineAI GmbH
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""What a VLM needs before iree-run-llm: an image and a prompt in, the
vision encoder's input and the prompt out.

This is model knowledge (image size and layout, chat template, how many
placeholder tokens one image takes), so it lives here, next to the model
code, and not in iree-run-llm. The suite's import and
integration_tests/tools/preprocess_vlm.py use the same processor, so the
reference and the device see the same pixels and prompt.
"""

from __future__ import annotations

from typing import Literal, Protocol, get_args

import numpy as np
from PIL import Image

# How a non-square image becomes the encoder's square: `stretch` resizes it
# to the square, distorting its aspect ratio; `pad` resizes it to fit and
# pads the short side, centred, with mid-grey (0 after the encoders' 0.5/0.5
# normalization).
Fit = Literal["stretch", "pad"]
FITS: tuple[Fit, ...] = get_args(Fit)
_PAD_COLOR = (128, 128, 128)


class VlmProcessor(Protocol):
    def chat_prompt(self, prompt: str) -> str:
        """The chat template for one image and the user's `prompt`, with the
        image placeholder as the template writes it."""
        ...

    def process(self, image: Image.Image, prompt: str) -> tuple[str, np.ndarray]:
        """`prompt` with its image placeholder expanded as the language model
        takes it, and `image` as the vision encoder takes it."""
        ...


def fit_square(
    image: Image.Image,
    side: int,
    fit: Fit,
    resample: Image.Resampling = Image.Resampling.BICUBIC,
) -> Image.Image:
    """`image` as RGB, `side` x `side`, resized with `resample` (the model's
    HF image processor's); an image of that size already is unchanged."""
    image = image.convert("RGB")
    if image.size == (side, side):
        return image
    if fit == "stretch":
        return image.resize((side, side), resample)
    if fit != "pad":
        raise ValueError(f"unknown fit {fit!r}, expected one of {FITS}")
    width, height = image.size
    scale = side / max(width, height)
    size = (max(1, round(width * scale)), max(1, round(height * scale)))
    canvas = Image.new("RGB", (side, side), _PAD_COLOR)
    canvas.paste(
        image.resize(size, resample),
        ((side - size[0]) // 2, (side - size[1]) // 2),
    )
    return canvas


def to_nchw(image: Image.Image, dtype: np.dtype) -> np.ndarray:
    """An RGB image as [1, 3, H, W] in 0..1."""
    array = np.asarray(image, dtype=np.float32) / 255.0
    return np.ascontiguousarray(array.transpose(2, 0, 1)[None]).astype(dtype)


class Qwen3VLProcessor:
    """Qwen3-VL with a fixed-size encoder (see qwen3_vl.VisionEncoder).

    The image is fitted to `side` x `side`, unlike HF's processor, which keeps
    roughly the image's own size and aspect ratio. Mean/std normalization is
    in the encoder graph, so the encoder takes plain RGB in 0..1, f16.
    """

    IMAGE_PAD = "<|image_pad|>"
    # preprocessor_config.json: patch_size 16, merge_size 2. One image token
    # per IMAGE_FACTOR x IMAGE_FACTOR pixels.
    IMAGE_FACTOR = 32

    def __init__(self, side: int, fit: Fit = "stretch") -> None:
        if side % self.IMAGE_FACTOR:
            raise ValueError(f"side {side} is not a multiple of {self.IMAGE_FACTOR}")
        self.side = side
        self.fit = fit

    @property
    def num_image_tokens(self) -> int:
        return (self.side // self.IMAGE_FACTOR) ** 2

    def chat_prompt(self, prompt: str) -> str:
        """Qwen3-VL's chat template (ChatML with Qwen-VL's vision markers) for
        one user turn: the image, then `prompt`, no system message, and the
        generation prompt that starts the assistant's answer.

        Checked against (chat_template.json, apply_chat_template with
        add_generation_prompt=True)
        """
        return (
            "<|im_start|>user\n<|vision_start|>"
            f"{self.IMAGE_PAD}<|vision_end|>{prompt}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )

    def process(self, image: Image.Image, prompt: str) -> tuple[str, np.ndarray]:
        if prompt.count(self.IMAGE_PAD) != 1:
            raise ValueError(
                f"the prompt needs exactly one {self.IMAGE_PAD}: {prompt!r}"
            )
        expanded = prompt.replace(
            self.IMAGE_PAD, self.IMAGE_PAD * self.num_image_tokens
        )
        pixels = to_nchw(fit_square(image, self.side, self.fit), np.float16)
        return expanded, pixels


class Gemma3Processor:
    """Gemma 3: one `side` x `side` image (896, fixed by the encoder) is 256
    image tokens. As HF's processor without pan and scan, the image is
    resized to the square, bilinear; `pad` keeps its aspect ratio instead.
    Normalization is in the encoder graph, so it takes RGB in 0..1, f16.
    """

    BOI = "<start_of_image>"

    def __init__(self, side: int, num_image_tokens: int, fit: Fit = "stretch") -> None:
        self.side = side
        self.num_image_tokens = num_image_tokens
        self.fit = fit

    def chat_prompt(self, prompt: str) -> str:
        """Gemma 3's chat template (chat_template.json) for one user turn: the
        image, then `prompt`, and the generation prompt."""
        return (
            f"<start_of_turn>user\n{self.BOI}{prompt}<end_of_turn>\n"
            "<start_of_turn>model\n"
        )

    def process(self, image: Image.Image, prompt: str) -> tuple[str, np.ndarray]:
        if prompt.count(self.BOI) != 1:
            raise ValueError(f"the prompt needs exactly one {self.BOI}: {prompt!r}")
        # Gemma3Processor.full_image_sequence.
        expanded = prompt.replace(
            self.BOI,
            f"\n\n{self.BOI}"
            + "<image_soft_token>" * self.num_image_tokens
            + "<end_of_image>\n\n",
        )
        square = fit_square(image, self.side, self.fit, Image.Resampling.BILINEAR)
        return expanded, to_nchw(square, np.float16)

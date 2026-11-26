# Copyright 2026 RooflineAI GmbH
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Prepare an image and a prompt for iree-run-llm with a VLM.

Writes the two files a VLM run needs, made by the model entry's processor
(get_vlm_preprocessor() in its model.py, see integration_tests/models/vlm_preprocessor.py),
the same one the suite's reference uses:

  * image.npy: the image as the vision encoder takes it
    (iree-run-llm --vision_input=@image.npy);
  * prompt.txt: the chat prompt with the image's tokens expanded
    (iree-run-llm --prompt_file=prompt.txt).

Example:

  .venv/bin/python -m integration_tests.tools.preprocess_vlm \\
      --model=Qwen/Qwen3-VL-2B-Instruct-kv-cache --image=photo.jpg \\
      --prompt="Describe this image in one short sentence." -o out/
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from integration_tests.models.vlm_preprocessor import FITS, Fit
from integration_tests.modeltree.discover import select
from integration_tests.modeltree.spec import ModelSpec, load_model_source


@dataclass(frozen=True)
class VlmInputs:
    """What `preprocess` wrote, and what it holds."""

    image_file: Path
    prompt_file: Path
    pixels: np.ndarray
    prompt: str


def preprocess(
    spec: ModelSpec, image: Path, prompt: str, fit: Fit, output_dir: Path
) -> VlmInputs:
    """Write image.npy and prompt.txt for `spec`'s VLM to `output_dir`.
    Parameters are documented below in argparse
    """
    module = load_model_source(spec)
    if not hasattr(module, "get_vlm_preprocessor"):
        raise ImportError(
            f"{spec.model_source} does not define get_vlm_preprocessor(): "
            "not a VLM entry"
        )
    if not image.exists():
        raise FileNotFoundError(f"{spec.name}: no such image {image}")
    processor = module.get_vlm_preprocessor(fit)
    with Image.open(image) as opened:
        expanded, pixels = processor.process(opened, processor.chat_prompt(prompt))

    output_dir.mkdir(parents=True, exist_ok=True)
    image_file = output_dir / "image.npy"
    prompt_file = output_dir / "prompt.txt"
    np.save(image_file, pixels)
    prompt_file.write_text(expanded)
    return VlmInputs(image_file, prompt_file, pixels, expanded)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--model",
        required=True,
        help="model entry under integration_tests/models, e.g. "
        "Qwen/Qwen3-VL-2B-Instruct-kv-cache",
    )
    parser.add_argument("--image", required=True, type=Path, help="any PIL format")
    parser.add_argument(
        "--prompt",
        required=True,
        help="the user prompt, put into the model's chat template for one image",
    )
    parser.add_argument(
        "--fit",
        choices=FITS,
        default="stretch",
        help="how a non-square image becomes the encoder's square size: "
        "stretch it, or keep its aspect ratio and pad (default: %(default)s)",
    )
    parser.add_argument("-o", "--output_dir", required=True, type=Path)
    args = parser.parse_args(argv)

    (spec,) = select([args.model])
    inputs = preprocess(spec, args.image, args.prompt, args.fit, args.output_dir)
    print(
        f"wrote {inputs.image_file} {list(inputs.pixels.shape)} {inputs.pixels.dtype}"
        f" and {inputs.prompt_file}\n"
        "iree-run-llm ... --vision_input=@image.npy --prompt_file=prompt.txt"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

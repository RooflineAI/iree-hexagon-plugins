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
from pathlib import Path

import numpy as np
from PIL import Image

from integration_tests.models.vlm_preprocessor import FITS
from integration_tests.modeltree.discover import select


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

    # The importer loads model.py; it pulls in torch, but builds no model.
    from integration_tests.stages.importing import _load_model_source

    (spec,) = select([args.model])
    module = _load_model_source(spec)
    if not hasattr(module, "get_vlm_preprocessor"):
        parser.error(f"{spec.name} has no get_vlm_preprocessor(): not a VLM entry")
    processor = module.get_vlm_preprocessor(args.fit)

    with Image.open(args.image) as image:
        expanded, pixels = processor.process(image, processor.chat_prompt(args.prompt))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.save(args.output_dir / "image.npy", pixels)
    (args.output_dir / "prompt.txt").write_text(expanded)
    print(
        f"wrote {args.output_dir / 'image.npy'} {list(pixels.shape)} {pixels.dtype}"
        f" and {args.output_dir / 'prompt.txt'}\n"
        "iree-run-llm ... --vision_input=@image.npy --prompt_file=prompt.txt"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

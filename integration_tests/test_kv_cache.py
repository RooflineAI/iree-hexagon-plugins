# Copyright 2026 RooflineAI GmbH
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""A host test of `KvCacheLM` (`models/kv_cache.py`), needing no device.

The device tests check iree-run-llm against `KvCacheLM.generate_greedy`, which
runs the same `prefill` and `decode` code eagerly, mask included. This checks
that code against the model itself: a forward of the whole history, without a
cache, has to give the logits and tokens the turns give with the cache, each
prompt after the first prefilled at the position the turn before left off at.
"""

from __future__ import annotations

from pathlib import Path

import torch
from tokenizers import Tokenizer

from integration_tests.modeltree.spec import load_model_spec
from integration_tests.stages.importing import _load_model_source, load_model

_MODELS = Path(__file__).parent / "models"
# Small, and with more than one prompt in its generation block.
_MODEL = "HuggingFaceTB/SmolLM2-135M-kv-cache"


def test_continued_prefill_matches_full_forward() -> None:
    spec = load_model_spec(_MODELS / _MODEL, _MODELS)
    assert len(spec.generation) > 1
    # float32, so that the cache and the full forward differ by rounding only.
    model = load_model(spec).to(torch.float32)
    tokenizer = Tokenizer.from_file(str(_load_model_source(spec).get_tokenizer_json()))
    turns = [
        (
            tokenizer.encode(turn.prompt, add_special_tokens=index == 0).ids,
            turn.max_new_tokens,
        )
        for index, turn in enumerate(spec.generation)
    ]

    results = model.generate_greedy(turns)

    history: list[int] = []
    # Per turn: where its prefill starts in the history, and its length.
    prefills: list[tuple[int, int]] = []
    for index, ((prompt, _), (generated, _)) in enumerate(zip(turns, results)):
        # A turn after the first starts with the previous turn's last token.
        start = len(history) - (index > 0)
        prefills.append((start, len(history) + len(prompt) - start))
        history += prompt + [info.selected_id for info in generated]
    with torch.no_grad():
        full = model.model(torch.tensor([history]), use_cache=False).logits[0]

    for index, ((start, length), (generated, prefill_logits)) in enumerate(
        zip(prefills, results)
    ):
        torch.testing.assert_close(
            prefill_logits[0, :length],
            full[start : start + length],
            atol=1e-3,
            rtol=1e-3,
            msg=lambda message, turn=index + 1: f"turn {turn} prefill: {message}",
        )
        # Each token is the argmax at the position before it.
        first = start + length - 1
        expected = full[first : first + len(generated)].argmax(dim=-1).tolist()
        assert [info.selected_id for info in generated] == expected, f"turn {index + 1}"

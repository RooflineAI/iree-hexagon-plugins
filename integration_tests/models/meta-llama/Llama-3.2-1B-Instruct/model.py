# Copyright 2026 RooflineAI GmbH
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import torch
from transformers import AutoModelForCausalLM

_MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"
_REVISION = "9213176726f574b556790deb65791e0c5aa438b6"


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
    return model

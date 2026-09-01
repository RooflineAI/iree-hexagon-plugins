# Copyright 2026 RooflineAI GmbH
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import torch
from transformers import AutoModelForImageClassification

_MODEL_ID = "microsoft/resnet-50"
_REVISION = "34c2154c194f829b11125337b98c8f5f9965ff19"


def get_model() -> torch.nn.Module:
    return AutoModelForImageClassification.from_pretrained(
        _MODEL_ID, revision=_REVISION, dtype=torch.float32
    )

# Copyright 2026 RooflineAI GmbH
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Byte predicates selecting f32 lanes, as in Gemma-4's attention mask.

Short byte vectors can legalize through word-sized HVX lanes. Each logical
predicate must control all four bytes of the selected float. Both all-true
and mixed predicates are runtime inputs, preventing constant folding.
"""

import torch


class PredicateSelect(torch.nn.Module):
    def forward(self, all_true, mixed, yes, no):
        return torch.where(all_true, yes, no), torch.where(mixed, yes, no)


def get_model() -> torch.nn.Module:
    return PredicateSelect()

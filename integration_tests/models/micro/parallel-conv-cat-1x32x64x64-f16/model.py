# Copyright 2026 RooflineAI GmbH
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Two independent 3x3 convs on the same input, concatenated and reduced by
a max over the channels, in f16.

The two convs do not depend on each other, so the command buffer records them
in one concurrent group (no barrier in between) and the DSP runtime can run
them at the same time on two worker threads (convs do not use HMX). They
dominate the run time, so async dispatch execution is visible end to end,
compared to running with --hexagon_max_concurrent_dispatches=1.

The branches are concatenated rather than summed: an add gets fused into the
second conv's dispatch, which then reads the first conv's result and has to
wait for it. With a concat each conv writes its own slice of the result.

The max over the channels of the concatenation is a reduction, which does not
fuse into the convs. It becomes a third, short dispatch after the convs that
reads both of their results.
"""

import torch


class ParallelConvCat(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv_a = torch.nn.Conv2d(32, 32, 3, padding=1)
        self.conv_b = torch.nn.Conv2d(32, 32, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = torch.cat([self.conv_a(x), self.conv_b(x)], dim=1)
        return y.amax(dim=1, keepdim=True)


def get_model() -> torch.nn.Module:
    torch.manual_seed(0)
    return ParallelConvCat().to(torch.float16)

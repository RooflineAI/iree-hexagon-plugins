// Copyright 2026 RooflineAI GmbH
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include <cstdint>

extern "C" {
#include "cmd_barrier_serialize.h"
#include "hexagon/arm_dsp/cmd_buf.h"
}

namespace {

TEST(CmdBarrierSerializeTest, Serialize) {
  iree_host_size_t cmd_size = 0;
  IREE_EXPECT_OK(iree_hal_hexagon_cmd_barrier_serialize_prep(&cmd_size));
  EXPECT_EQ(sizeof(hexagon_rt_arm_dsp_cmd_barrier_t), cmd_size);

  std::vector<uint8_t> buffer(cmd_size);
  IREE_EXPECT_OK(iree_hal_hexagon_cmd_barrier_serialize_exec(buffer.data(),
                                                             buffer.size()));

  const auto *cmd_barrier =
      reinterpret_cast<const hexagon_rt_arm_dsp_cmd_barrier_t *>(buffer.data());
  EXPECT_EQ(HEXAGON_RT_ARM_DSP_CMD_BARRIER, cmd_barrier->base.cmd_type);
}

} // namespace

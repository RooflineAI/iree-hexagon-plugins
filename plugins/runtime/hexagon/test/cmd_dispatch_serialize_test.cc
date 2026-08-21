// Copyright 2026 RooflineAI GmbH
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <array>
#include <cstdint>
#include <vector>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

extern "C" {
#include "cmd_dispatch_serialize.h"
#include "hexagon/arm_dsp/cmd_buf.h"
#include "hexagon/serialize/rpc_types.h"
}
#include "serialize_test_utils.h"

namespace {

TEST(CmdDispatchSerializeTest, Serialize) {
  // Dispatch with two indirect bindings, one direct binding, and constants.
  rpc_executable_handle_t rpc_executable_handle = 0x123456;

  iree_hal_dispatch_config_t config =
      iree_hal_make_static_dispatch_config(/*x=*/3, /*y=*/2, /*z=*/1);
  config.workgroup_size[0] = 8;
  config.workgroup_size[1] = 4;
  config.workgroup_size[2] = 2;

  const std::array<uint32_t, 3> constant_values = {0xABCD1234, 0x77778888,
                                                   0x0000FFFF};
  const size_t constant_count = constant_values.size();
  const iree_const_byte_span_t constants = iree_make_const_byte_span(
      reinterpret_cast<const uint8_t *>(constant_values.data()),
      constant_values.size() * sizeof(uint32_t));

  std::array<iree_hal_buffer_ref_t, 4> binding_values = {
      iree_hal_make_indirect_buffer_ref(/*buffer_slot=*/1, /*offset=*/32,
                                        /*length=*/64),
      iree_hal_make_indirect_buffer_ref(/*buffer_slot=*/2, /*offset=*/96,
                                        /*length=*/128),
      iree_hal_make_buffer_ref(&hexagon_test_direct_subspan_buffer0,
                               /*offset=*/160,
                               /*length=*/64),
      iree_hal_make_buffer_ref(&hexagon_test_direct_buffer1, /*offset=*/180,
                               /*length=*/32),
  };
  iree_hal_buffer_ref_list_t bindings = {binding_values.size(),
                                         binding_values.data()};

  // Size computation.
  const iree_host_size_t expected_size =
      sizeof(hexagon_rt_arm_dsp_cmd_dispatch_t) +
      constant_count * sizeof(hexagon_rt_arm_dsp_con_t) +
      binding_values.size() * sizeof(hexagon_rt_arm_dsp_buf_ref_t);
  iree_host_size_t cmd_size = 0;
  IREE_EXPECT_OK(iree_hal_hexagon_cmd_dispatch_serialize_prep(
      &constants, &bindings, &cmd_size));
  EXPECT_EQ(expected_size, cmd_size);

  // Serialization.
  std::vector<uint8_t> buffer(cmd_size);
  IREE_EXPECT_OK(iree_hal_hexagon_cmd_dispatch_serialize_exec(
      rpc_executable_handle, /*export_ordinal=*/7, &config, &constants,
      &bindings, hexagon_test_get_buffer_fd, buffer.data(), buffer.size()));

  const auto *cmd_dispatch =
      reinterpret_cast<const hexagon_rt_arm_dsp_cmd_dispatch_t *>(
          buffer.data());
  EXPECT_EQ(HEXAGON_RT_ARM_DSP_CMD_DISPATCH, cmd_dispatch->base.cmd_type);
  EXPECT_EQ(rpc_executable_handle, cmd_dispatch->executable_handle);
  EXPECT_EQ(7u, cmd_dispatch->export_ordinal);
  EXPECT_EQ(8u, cmd_dispatch->workgroup_size_x);
  EXPECT_EQ(4u, cmd_dispatch->workgroup_size_y);
  EXPECT_EQ(2u, cmd_dispatch->workgroup_size_z);
  EXPECT_EQ(3u, cmd_dispatch->workgroup_count_x);
  EXPECT_EQ(2u, cmd_dispatch->workgroup_count_y);
  EXPECT_EQ(1u, cmd_dispatch->workgroup_count_z);
  EXPECT_EQ(constant_count, cmd_dispatch->constant_count);
  EXPECT_EQ(binding_values.size(), cmd_dispatch->num_bindings);

  const uint8_t *cursor = buffer.data() + sizeof(*cmd_dispatch);
  const auto *constants_data =
      reinterpret_cast<const hexagon_rt_arm_dsp_con_t *>(cursor);
  cursor += constant_count * sizeof(*constants_data);
  for (size_t i = 0; i < constant_count; ++i) {
    EXPECT_EQ(constant_values[i], constants_data[i].value);
  }

  for (size_t i = 0; i < binding_values.size(); ++i) {
    const auto *binding =
        reinterpret_cast<const hexagon_rt_arm_dsp_buf_ref_t *>(cursor);
    cursor += sizeof(*binding);
    EXPECT_EQ(binding_values[i].buffer_slot, binding->slot);
    EXPECT_EQ(i == 2   ? HEXAGON_TEST_DIRECT_SUBSPAN_BUFFER0_FD
              : i == 3 ? HEXAGON_TEST_DIRECT_BUFFER1_FD
                       : -1,
              binding->fd);
    EXPECT_EQ((i == 2 ? HEXAGON_TEST_DIRECT_SUBSPAN_BUFFER0_OFFSET : 0) +
                  binding_values[i].offset,
              binding->offset);
    EXPECT_EQ(binding_values[i].length, binding->length);
  }

  // Ensure we consumed exactly the serialized buffer.
  EXPECT_EQ(buffer.data() + buffer.size(), cursor);
}

} // namespace

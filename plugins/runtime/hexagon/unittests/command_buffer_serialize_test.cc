
#include <array>
#include <cstdint>
#include <vector>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

extern "C" {
#include "command_buffer_serialize.h"
#include "command_buffer_types.h"
#include "hexagon/arm_dsp/cmd_buf.h"
}

namespace {

TEST(CommandBufferSerializeTest, EmptyCommandBuffer) {
  iree_hal_hexagon_command_buffer_t command_buffer = {};

  iree_host_size_t cmd_buf_size = 0;
  uint32_t num_entries = 0;
  IREE_EXPECT_OK(iree_hal_hexagon_command_buffer_serialize_prep(
      &command_buffer, &cmd_buf_size, &num_entries));
  EXPECT_EQ(sizeof(hexagon_rt_arm_dsp_cmd_buf_t), cmd_buf_size);
  EXPECT_EQ(0u, num_entries);

  std::vector<uint8_t> buffer(cmd_buf_size);
  IREE_EXPECT_OK(iree_hal_hexagon_command_buffer_serialize_exec(
      &command_buffer, num_entries, buffer.data(), buffer.size()));

  const auto *header =
      reinterpret_cast<const hexagon_rt_arm_dsp_cmd_buf_t *>(buffer.data());
  EXPECT_EQ(0u, header->num_entries);
}

TEST(CommandBufferSerializeTest, SingleDispatch) {
  iree_hal_hexagon_command_dispatch_t dispatch = {};
  dispatch.base.cmd_type = IREE_HAL_HEXAGON_COMMAND_DISPATCH;
  dispatch.rpc_executable_handle = 0x123456;
  dispatch.entry_point = 7;

  iree_hal_buffer_t direct_buffer = {};

  std::array<iree_hal_buffer_ref_t, 3> binding_values = {
      iree_hal_make_indirect_buffer_ref(/*buffer_slot=*/1, /*offset=*/32,
                                        /*length=*/64),
      iree_hal_make_indirect_buffer_ref(/*buffer_slot=*/2, /*offset=*/96,
                                        /*length=*/128),
      iree_hal_make_buffer_ref(&direct_buffer, /*offset=*/160, /*length=*/64),
  };
  dispatch.bindings.count = binding_values.size();
  dispatch.bindings.values = binding_values.data();

  iree_hal_hexagon_command_buffer_t command_buffer = {};
  command_buffer.first_entry = &dispatch.base;
  command_buffer.last_entry = &dispatch.base;

  const iree_host_size_t expected_size =
      sizeof(hexagon_rt_arm_dsp_cmd_buf_t) +
      sizeof(hexagon_rt_arm_dsp_cmd_dispatch_t) +
      binding_values.size() * sizeof(hexagon_rt_arm_dsp_binding_t);

  iree_host_size_t cmd_buf_size = 0;
  uint32_t num_entries = 0;
  IREE_EXPECT_OK(iree_hal_hexagon_command_buffer_serialize_prep(
      &command_buffer, &cmd_buf_size, &num_entries));
  EXPECT_EQ(expected_size, cmd_buf_size);
  EXPECT_EQ(1u, num_entries);

  std::vector<uint8_t> buffer(cmd_buf_size);
  IREE_EXPECT_OK(iree_hal_hexagon_command_buffer_serialize_exec(
      &command_buffer, num_entries, buffer.data(), buffer.size()));

  const auto *header =
      reinterpret_cast<const hexagon_rt_arm_dsp_cmd_buf_t *>(buffer.data());
  EXPECT_EQ(1u, header->num_entries);

  const uint8_t *cursor = buffer.data() + sizeof(*header);
  const auto *cmd_dispatch =
      reinterpret_cast<const hexagon_rt_arm_dsp_cmd_dispatch_t *>(cursor);
  cursor += sizeof(*cmd_dispatch);

  EXPECT_EQ(HEXAGON_RT_ARM_DSP_CMD_DISPATCH, cmd_dispatch->base.cmd_type);
  EXPECT_EQ(dispatch.rpc_executable_handle, cmd_dispatch->executable_handle);
  EXPECT_EQ(dispatch.entry_point, cmd_dispatch->entry_point);
  EXPECT_EQ(binding_values.size(), cmd_dispatch->num_bindings);

  for (size_t i = 0; i < binding_values.size(); ++i) {
    const auto *binding =
        reinterpret_cast<const hexagon_rt_arm_dsp_binding_t *>(cursor);
    cursor += sizeof(*binding);
    EXPECT_EQ(binding_values[i].buffer_slot, binding->slot);
    EXPECT_EQ(0, binding->buffer_handle);
    EXPECT_EQ(binding_values[i].offset, binding->offset);
    EXPECT_EQ(binding_values[i].length, binding->length);
  }
  EXPECT_EQ(buffer.data() + buffer.size(), cursor);
}

TEST(CommandBufferSerializeTest, SingleExecutionBarrier) {
  iree_hal_hexagon_command_barrier_t barrier = {};
  barrier.base.cmd_type = IREE_HAL_HEXAGON_COMMAND_BARRIER;

  iree_hal_hexagon_command_buffer_t command_buffer = {};
  command_buffer.first_entry = &barrier.base;
  command_buffer.last_entry = &barrier.base;

  const iree_host_size_t expected_size =
      sizeof(hexagon_rt_arm_dsp_cmd_buf_t) +
      sizeof(hexagon_rt_arm_dsp_cmd_barrier_t);

  iree_host_size_t cmd_buf_size = 0;
  uint32_t num_entries = 0;
  IREE_EXPECT_OK(iree_hal_hexagon_command_buffer_serialize_prep(
      &command_buffer, &cmd_buf_size, &num_entries));
  EXPECT_EQ(expected_size, cmd_buf_size);
  EXPECT_EQ(1u, num_entries);

  std::vector<uint8_t> buffer(cmd_buf_size);
  IREE_EXPECT_OK(iree_hal_hexagon_command_buffer_serialize_exec(
      &command_buffer, num_entries, buffer.data(), buffer.size()));

  const auto *header =
      reinterpret_cast<const hexagon_rt_arm_dsp_cmd_buf_t *>(buffer.data());
  EXPECT_EQ(1u, header->num_entries);

  const uint8_t *cursor = buffer.data() + sizeof(*header);
  const auto *cmd_barrier =
      reinterpret_cast<const hexagon_rt_arm_dsp_cmd_barrier_t *>(cursor);
  cursor += sizeof(*cmd_barrier);

  EXPECT_EQ(HEXAGON_RT_ARM_DSP_CMD_BARRIER, cmd_barrier->base.cmd_type);
  EXPECT_EQ(buffer.data() + buffer.size(), cursor);
}

TEST(CommandBufferSerializeTest, DispatchBarrierDispatch) {
  rpc_executable_handle_t first_rpc_executable_handle = 0xAAAA;
  rpc_executable_handle_t second_rpc_executable_handle = 0xBBBB;

  iree_hal_hexagon_command_dispatch_t first_dispatch = {};
  first_dispatch.base.cmd_type = IREE_HAL_HEXAGON_COMMAND_DISPATCH;
  first_dispatch.rpc_executable_handle = first_rpc_executable_handle;
  first_dispatch.entry_point = 2;

  iree_hal_hexagon_command_barrier_t barrier = {};
  barrier.base.cmd_type = IREE_HAL_HEXAGON_COMMAND_BARRIER;

  iree_hal_hexagon_command_dispatch_t second_dispatch = {};
  second_dispatch.base.cmd_type = IREE_HAL_HEXAGON_COMMAND_DISPATCH;
  second_dispatch.rpc_executable_handle = second_rpc_executable_handle;
  second_dispatch.entry_point = 5;

  std::array<iree_hal_buffer_ref_t, 1> first_bindings = {
      iree_hal_make_indirect_buffer_ref(/*buffer_slot=*/3, /*offset=*/0,
                                        /*length=*/256),
  };
  first_dispatch.bindings.count = first_bindings.size();
  first_dispatch.bindings.values = first_bindings.data();

  iree_hal_buffer_t second_direct_buffer = {};

  std::array<iree_hal_buffer_ref_t, 4> second_bindings = {
      iree_hal_make_indirect_buffer_ref(/*buffer_slot=*/4, /*offset=*/32,
                                        /*length=*/64),
      iree_hal_make_indirect_buffer_ref(/*buffer_slot=*/7, /*offset=*/96,
                                        /*length=*/128),
      iree_hal_make_indirect_buffer_ref(/*buffer_slot=*/8, /*offset=*/224,
                                        /*length=*/48),
      iree_hal_make_buffer_ref(&second_direct_buffer, /*offset=*/0,
                               /*length=*/192),
  };
  second_dispatch.bindings.count = second_bindings.size();
  second_dispatch.bindings.values = second_bindings.data();

  first_dispatch.base.next = &barrier.base;
  barrier.base.prev = &first_dispatch.base;
  barrier.base.next = &second_dispatch.base;
  second_dispatch.base.prev = &barrier.base;

  iree_hal_hexagon_command_buffer_t command_buffer = {};
  command_buffer.first_entry = &first_dispatch.base;
  command_buffer.last_entry = &second_dispatch.base;

  const iree_host_size_t expected_size =
      sizeof(hexagon_rt_arm_dsp_cmd_buf_t) +
      sizeof(hexagon_rt_arm_dsp_cmd_dispatch_t) +
      first_bindings.size() * sizeof(hexagon_rt_arm_dsp_binding_t) +
      sizeof(hexagon_rt_arm_dsp_cmd_barrier_t) +
      sizeof(hexagon_rt_arm_dsp_cmd_dispatch_t) +
      second_bindings.size() * sizeof(hexagon_rt_arm_dsp_binding_t);

  iree_host_size_t cmd_buf_size = 0;
  uint32_t num_entries = 0;
  IREE_EXPECT_OK(iree_hal_hexagon_command_buffer_serialize_prep(
      &command_buffer, &cmd_buf_size, &num_entries));
  EXPECT_EQ(expected_size, cmd_buf_size);
  EXPECT_EQ(3u, num_entries);

  std::vector<uint8_t> buffer(cmd_buf_size);
  IREE_EXPECT_OK(iree_hal_hexagon_command_buffer_serialize_exec(
      &command_buffer, num_entries, buffer.data(), buffer.size()));

  const auto *header =
      reinterpret_cast<const hexagon_rt_arm_dsp_cmd_buf_t *>(buffer.data());
  EXPECT_EQ(3u, header->num_entries);

  const uint8_t *cursor = buffer.data() + sizeof(*header);
  auto verify_dispatch =
      [&](const rpc_executable_handle_t &rpc_executable_handle,
          int32_t entry_point, const iree_hal_buffer_ref_t *refs,
          size_t count) {
        const auto *cmd_dispatch =
            reinterpret_cast<const hexagon_rt_arm_dsp_cmd_dispatch_t *>(cursor);
        cursor += sizeof(*cmd_dispatch);

        EXPECT_EQ(HEXAGON_RT_ARM_DSP_CMD_DISPATCH, cmd_dispatch->base.cmd_type);
        EXPECT_EQ(rpc_executable_handle, cmd_dispatch->executable_handle);
        EXPECT_EQ(entry_point, cmd_dispatch->entry_point);
        EXPECT_EQ(count, cmd_dispatch->num_bindings);

        for (size_t i = 0; i < count; ++i) {
          const auto *binding =
              reinterpret_cast<const hexagon_rt_arm_dsp_binding_t *>(cursor);
          cursor += sizeof(*binding);
          EXPECT_EQ(refs[i].buffer_slot, binding->slot);
          EXPECT_EQ(0, binding->buffer_handle);
          EXPECT_EQ(refs[i].offset, binding->offset);
          EXPECT_EQ(refs[i].length, binding->length);
        }
      };

  verify_dispatch(first_rpc_executable_handle, first_dispatch.entry_point,
                  first_bindings.data(), first_bindings.size());

  const auto *cmd_barrier =
      reinterpret_cast<const hexagon_rt_arm_dsp_cmd_barrier_t *>(cursor);
  cursor += sizeof(*cmd_barrier);
  EXPECT_EQ(HEXAGON_RT_ARM_DSP_CMD_BARRIER, cmd_barrier->base.cmd_type);

  verify_dispatch(second_rpc_executable_handle, second_dispatch.entry_point,
                  second_bindings.data(), second_bindings.size());

  EXPECT_EQ(buffer.data() + buffer.size(), cursor);
}

} // namespace

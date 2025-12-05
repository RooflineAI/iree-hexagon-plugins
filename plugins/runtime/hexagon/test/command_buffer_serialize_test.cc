
#include <array>
#include <cstdint>
#include <vector>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

extern "C" {
#include "command_buffer_serialize.h"
#include "command_buffer_types.h"
#include "hexagon/arm_dsp/cmd_buf.h"
#include "hexagon/serialize/rpc_types.h"
}

namespace {

iree_hal_buffer_t direct_buffers[2] = {};

// Mock function for getting DSP virtual address from buffer.
// Return some fake DSP virtual address that has the number of the direct buffer
// at the end.
iree_status_t buffer_to_dsp_vaddr(iree_hal_buffer_t *buffer,
                                  rpc_dsp_vaddr_t *out_dsp_vaddr) {
  for (size_t i = 0; i < sizeof(direct_buffers) / sizeof(direct_buffers[0]);
       ++i) {
    if (buffer == &direct_buffers[i]) {
      *out_dsp_vaddr = static_cast<rpc_dsp_vaddr_t>(0x420000 + i);
      return iree_ok_status();
    }
  }
  return iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "unknown buffer");
}

TEST(CommandBufferSerializeTest, EmptyCommandBuffer) {

  // Test input data structure: empty command buffer.

  iree_hal_hexagon_command_buffer_t command_buffer = {};

  iree_host_size_t cmd_buf_size = 0;
  uint32_t num_entries = 0;
  IREE_EXPECT_OK(iree_hal_hexagon_command_buffer_serialize_prep(
      &command_buffer, &cmd_buf_size, &num_entries));
  EXPECT_EQ(sizeof(hexagon_rt_arm_dsp_cmd_buf_t), cmd_buf_size);
  EXPECT_EQ(0u, num_entries);

  // Test generating serialized data.

  std::vector<uint8_t> buffer(cmd_buf_size);
  IREE_EXPECT_OK(iree_hal_hexagon_command_buffer_serialize_exec(
      &command_buffer, buffer_to_dsp_vaddr, num_entries, buffer.data(),
      buffer.size()));

  const auto *header =
      reinterpret_cast<const hexagon_rt_arm_dsp_cmd_buf_t *>(buffer.data());
  EXPECT_EQ(0u, header->num_entries);
}

TEST(CommandBufferSerializeTest, SingleDispatch) {
  // Test input data structure: command buffer with single dispatch.
  // The dispatch has three buffers references, the first two dynamic/indirect,
  // the last one fixed/direct.

  iree_hal_hexagon_command_dispatch_t dispatch = {};
  dispatch.base.cmd_type = IREE_HAL_HEXAGON_COMMAND_DISPATCH;
  dispatch.rpc_executable_handle = 0x123456;
  dispatch.export_ordinal = 7;
  dispatch.workgroup_size_x = 8;
  dispatch.workgroup_size_y = 4;
  dispatch.workgroup_size_z = 2;
  dispatch.workgroup_count_x = 3;
  dispatch.workgroup_count_y = 2;
  dispatch.workgroup_count_z = 1;

  std::array<iree_hal_buffer_ref_t, 3> binding_values = {
      iree_hal_make_indirect_buffer_ref(/*buffer_slot=*/1, /*offset=*/32,
                                        /*length=*/64),
      iree_hal_make_indirect_buffer_ref(/*buffer_slot=*/2, /*offset=*/96,
                                        /*length=*/128),
      iree_hal_make_buffer_ref(&direct_buffers[0], /*offset=*/160,
                               /*length=*/64),
  };
  dispatch.bindings.count = binding_values.size();
  dispatch.bindings.values = binding_values.data();

  iree_hal_hexagon_command_buffer_t command_buffer = {};
  command_buffer.first_entry = &dispatch.base;
  command_buffer.last_entry = &dispatch.base;

  // Test computing size of serialized data and number of entries.

  const iree_host_size_t expected_size =
      sizeof(hexagon_rt_arm_dsp_cmd_buf_t) +
      sizeof(hexagon_rt_arm_dsp_cmd_dispatch_t) +
      binding_values.size() * sizeof(hexagon_rt_arm_dsp_buf_ref_t);

  iree_host_size_t cmd_buf_size = 0;
  uint32_t num_entries = 0;
  IREE_EXPECT_OK(iree_hal_hexagon_command_buffer_serialize_prep(
      &command_buffer, &cmd_buf_size, &num_entries));
  EXPECT_EQ(expected_size, cmd_buf_size);
  EXPECT_EQ(1u, num_entries);

  // Test generating serialized data.

  std::vector<uint8_t> buffer(cmd_buf_size);
  IREE_EXPECT_OK(iree_hal_hexagon_command_buffer_serialize_exec(
      &command_buffer, buffer_to_dsp_vaddr, num_entries, buffer.data(),
      buffer.size()));

  const auto *header =
      reinterpret_cast<const hexagon_rt_arm_dsp_cmd_buf_t *>(buffer.data());
  EXPECT_EQ(1u, header->num_entries);

  const uint8_t *cursor = buffer.data() + sizeof(*header);
  const auto *cmd_dispatch =
      reinterpret_cast<const hexagon_rt_arm_dsp_cmd_dispatch_t *>(cursor);
  cursor += sizeof(*cmd_dispatch);

  EXPECT_EQ(HEXAGON_RT_ARM_DSP_CMD_DISPATCH, cmd_dispatch->base.cmd_type);
  EXPECT_EQ(dispatch.rpc_executable_handle, cmd_dispatch->executable_handle);
  EXPECT_EQ(dispatch.export_ordinal, cmd_dispatch->export_ordinal);
  EXPECT_EQ(dispatch.workgroup_size_x, cmd_dispatch->workgroup_size_x);
  EXPECT_EQ(dispatch.workgroup_size_y, cmd_dispatch->workgroup_size_y);
  EXPECT_EQ(dispatch.workgroup_size_z, cmd_dispatch->workgroup_size_z);
  EXPECT_EQ(dispatch.workgroup_count_x, cmd_dispatch->workgroup_count_x);
  EXPECT_EQ(dispatch.workgroup_count_y, cmd_dispatch->workgroup_count_y);
  EXPECT_EQ(dispatch.workgroup_count_z, cmd_dispatch->workgroup_count_z);
  EXPECT_EQ(binding_values.size(), cmd_dispatch->num_bindings);

  for (size_t i = 0; i < binding_values.size(); ++i) {
    const auto *binding =
        reinterpret_cast<const hexagon_rt_arm_dsp_buf_ref_t *>(cursor);
    cursor += sizeof(*binding);
    EXPECT_EQ(binding_values[i].buffer_slot, binding->slot);
    // buffer ref at index 2 uses direct buffer 0 -> expect DSP virtual address
    EXPECT_EQ(i == 2 ? 0x420000 : 0, binding->buffer_dsp_vaddr);
    EXPECT_EQ(binding_values[i].offset, binding->offset);
    EXPECT_EQ(binding_values[i].length, binding->length);
  }

  // Test the test: Generated data needs to end at end of buffer.

  EXPECT_EQ(buffer.data() + buffer.size(), cursor);
}

TEST(CommandBufferSerializeTest, SingleExecutionBarrier) {

  // Test input data structure: command buffer with single execution barrier
  // entry.

  iree_hal_hexagon_command_barrier_t barrier = {};
  barrier.base.cmd_type = IREE_HAL_HEXAGON_COMMAND_BARRIER;

  iree_hal_hexagon_command_buffer_t command_buffer = {};
  command_buffer.first_entry = &barrier.base;
  command_buffer.last_entry = &barrier.base;

  // Test computing size of serialized data and number of entries.

  const iree_host_size_t expected_size =
      sizeof(hexagon_rt_arm_dsp_cmd_buf_t) +
      sizeof(hexagon_rt_arm_dsp_cmd_barrier_t);

  iree_host_size_t cmd_buf_size = 0;
  uint32_t num_entries = 0;
  IREE_EXPECT_OK(iree_hal_hexagon_command_buffer_serialize_prep(
      &command_buffer, &cmd_buf_size, &num_entries));
  EXPECT_EQ(expected_size, cmd_buf_size);
  EXPECT_EQ(1u, num_entries);

  // Test generating serialized data.

  std::vector<uint8_t> buffer(cmd_buf_size);
  IREE_EXPECT_OK(iree_hal_hexagon_command_buffer_serialize_exec(
      &command_buffer, buffer_to_dsp_vaddr, num_entries, buffer.data(),
      buffer.size()));

  const auto *header =
      reinterpret_cast<const hexagon_rt_arm_dsp_cmd_buf_t *>(buffer.data());
  EXPECT_EQ(1u, header->num_entries);

  const uint8_t *cursor = buffer.data() + sizeof(*header);
  const auto *cmd_barrier =
      reinterpret_cast<const hexagon_rt_arm_dsp_cmd_barrier_t *>(cursor);
  cursor += sizeof(*cmd_barrier);

  EXPECT_EQ(HEXAGON_RT_ARM_DSP_CMD_BARRIER, cmd_barrier->base.cmd_type);

  // Test the test: Generated data needs to end at end of buffer.

  EXPECT_EQ(buffer.data() + buffer.size(), cursor);
}

TEST(CommandBufferSerializeTest, SingleCopy) {

  // Test input data structure: command buffer with single buffer copy.

  iree_hal_hexagon_command_copy_t copy = {};
  copy.base.cmd_type = IREE_HAL_HEXAGON_COMMAND_COPY;
  copy.src = iree_hal_make_buffer_ref(&direct_buffers[0], /*offset=*/8,
                                      /*length=*/64);
  copy.dest = iree_hal_make_indirect_buffer_ref(/*buffer_slot=*/9,
                                                /*offset=*/24, /*length=*/64);

  iree_hal_hexagon_command_buffer_t command_buffer = {};
  command_buffer.first_entry = &copy.base;
  command_buffer.last_entry = &copy.base;

  // Test computing size of serialized data and number of entries.

  const iree_host_size_t expected_size = sizeof(hexagon_rt_arm_dsp_cmd_buf_t) +
                                         sizeof(hexagon_rt_arm_dsp_cmd_copy_t);

  iree_host_size_t cmd_buf_size = 0;
  uint32_t num_entries = 0;
  IREE_EXPECT_OK(iree_hal_hexagon_command_buffer_serialize_prep(
      &command_buffer, &cmd_buf_size, &num_entries));
  EXPECT_EQ(expected_size, cmd_buf_size);
  EXPECT_EQ(1u, num_entries);

  // Test generating serialized data.

  std::vector<uint8_t> buffer(cmd_buf_size);
  IREE_EXPECT_OK(iree_hal_hexagon_command_buffer_serialize_exec(
      &command_buffer, buffer_to_dsp_vaddr, num_entries, buffer.data(),
      buffer.size()));

  const auto *header =
      reinterpret_cast<const hexagon_rt_arm_dsp_cmd_buf_t *>(buffer.data());
  EXPECT_EQ(1u, header->num_entries);

  const uint8_t *cursor = buffer.data() + sizeof(*header);
  const auto *cmd_copy =
      reinterpret_cast<const hexagon_rt_arm_dsp_cmd_copy_t *>(cursor);
  cursor += sizeof(*cmd_copy);

  EXPECT_EQ(HEXAGON_RT_ARM_DSP_CMD_COPY, cmd_copy->base.cmd_type);

  EXPECT_EQ(copy.src.buffer_slot, cmd_copy->src.slot);
  EXPECT_EQ(0x420000, cmd_copy->src.buffer_dsp_vaddr);
  EXPECT_EQ(copy.src.offset, cmd_copy->src.offset);
  EXPECT_EQ(copy.src.length, cmd_copy->src.length);

  EXPECT_EQ(copy.dest.buffer_slot, cmd_copy->dest.slot);
  EXPECT_EQ(0, cmd_copy->dest.buffer_dsp_vaddr);
  EXPECT_EQ(copy.dest.offset, cmd_copy->dest.offset);
  EXPECT_EQ(copy.dest.length, cmd_copy->dest.length);

  // Test the test: Generated data needs to end at end of buffer.

  EXPECT_EQ(buffer.data() + buffer.size(), cursor);
}

TEST(CommandBufferSerializeTest, DispatchBarrierDispatch) {
  // Test input data structure: command buffer with the following entries:
  // first dispatch, barrier second dispatch.
  // The first dispatch has one dynamic/indirect buffer reference.
  // The second dispatch has four buffers references, the first three
  // dynamic/indirect, the last one fixed/direct.

  rpc_executable_handle_t first_rpc_executable_handle = 0xAAAA;
  rpc_executable_handle_t second_rpc_executable_handle = 0xBBBB;

  iree_hal_hexagon_command_dispatch_t first_dispatch = {};
  first_dispatch.base.cmd_type = IREE_HAL_HEXAGON_COMMAND_DISPATCH;
  first_dispatch.rpc_executable_handle = first_rpc_executable_handle;
  first_dispatch.export_ordinal = 2;
  first_dispatch.workgroup_size_x = 1;
  first_dispatch.workgroup_size_y = 2;
  first_dispatch.workgroup_size_z = 3;
  first_dispatch.workgroup_count_x = 4;
  first_dispatch.workgroup_count_y = 5;
  first_dispatch.workgroup_count_z = 6;

  iree_hal_hexagon_command_barrier_t barrier = {};
  barrier.base.cmd_type = IREE_HAL_HEXAGON_COMMAND_BARRIER;

  iree_hal_hexagon_command_dispatch_t second_dispatch = {};
  second_dispatch.base.cmd_type = IREE_HAL_HEXAGON_COMMAND_DISPATCH;
  second_dispatch.rpc_executable_handle = second_rpc_executable_handle;
  second_dispatch.export_ordinal = 5;
  second_dispatch.workgroup_size_x = 7;
  second_dispatch.workgroup_size_y = 8;
  second_dispatch.workgroup_size_z = 9;
  second_dispatch.workgroup_count_x = 10;
  second_dispatch.workgroup_count_y = 11;
  second_dispatch.workgroup_count_z = 12;

  std::array<iree_hal_buffer_ref_t, 1> first_bindings = {
      iree_hal_make_indirect_buffer_ref(/*buffer_slot=*/3, /*offset=*/0,
                                        /*length=*/256),
  };
  first_dispatch.bindings.count = first_bindings.size();
  first_dispatch.bindings.values = first_bindings.data();

  std::array<iree_hal_buffer_ref_t, 4> second_bindings = {
      iree_hal_make_indirect_buffer_ref(/*buffer_slot=*/4, /*offset=*/32,
                                        /*length=*/64),
      iree_hal_make_indirect_buffer_ref(/*buffer_slot=*/7, /*offset=*/96,
                                        /*length=*/128),
      iree_hal_make_indirect_buffer_ref(/*buffer_slot=*/8, /*offset=*/224,
                                        /*length=*/48),
      iree_hal_make_buffer_ref(&direct_buffers[1], /*offset=*/0,
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

  // Test computing size of serialized data and number of entries.

  const iree_host_size_t expected_size =
      sizeof(hexagon_rt_arm_dsp_cmd_buf_t) +
      sizeof(hexagon_rt_arm_dsp_cmd_dispatch_t) +
      first_bindings.size() * sizeof(hexagon_rt_arm_dsp_buf_ref_t) +
      sizeof(hexagon_rt_arm_dsp_cmd_barrier_t) +
      sizeof(hexagon_rt_arm_dsp_cmd_dispatch_t) +
      second_bindings.size() * sizeof(hexagon_rt_arm_dsp_buf_ref_t);

  iree_host_size_t cmd_buf_size = 0;
  uint32_t num_entries = 0;
  IREE_EXPECT_OK(iree_hal_hexagon_command_buffer_serialize_prep(
      &command_buffer, &cmd_buf_size, &num_entries));
  EXPECT_EQ(expected_size, cmd_buf_size);
  EXPECT_EQ(3u, num_entries);

  // Test generating serialized data.

  std::vector<uint8_t> buffer(cmd_buf_size);
  IREE_EXPECT_OK(iree_hal_hexagon_command_buffer_serialize_exec(
      &command_buffer, buffer_to_dsp_vaddr, num_entries, buffer.data(),
      buffer.size()));

  const auto *header =
      reinterpret_cast<const hexagon_rt_arm_dsp_cmd_buf_t *>(buffer.data());
  EXPECT_EQ(3u, header->num_entries);

  const uint8_t *cursor = buffer.data() + sizeof(*header);
  auto verify_dispatch =
      [&](const iree_hal_hexagon_command_dispatch_t &dispatch,
          const iree_hal_buffer_ref_t *refs, size_t count,
          size_t direct_buffer_at_idx, int64_t dsp_virt_addr) {
        const auto *cmd_dispatch =
            reinterpret_cast<const hexagon_rt_arm_dsp_cmd_dispatch_t *>(cursor);
        cursor += sizeof(*cmd_dispatch);

        EXPECT_EQ(HEXAGON_RT_ARM_DSP_CMD_DISPATCH, cmd_dispatch->base.cmd_type);
        EXPECT_EQ(dispatch.rpc_executable_handle,
                  cmd_dispatch->executable_handle);
        EXPECT_EQ(dispatch.export_ordinal, cmd_dispatch->export_ordinal);
        EXPECT_EQ(dispatch.workgroup_size_x, cmd_dispatch->workgroup_size_x);
        EXPECT_EQ(dispatch.workgroup_size_y, cmd_dispatch->workgroup_size_y);
        EXPECT_EQ(dispatch.workgroup_size_z, cmd_dispatch->workgroup_size_z);
        EXPECT_EQ(dispatch.workgroup_count_x, cmd_dispatch->workgroup_count_x);
        EXPECT_EQ(dispatch.workgroup_count_y, cmd_dispatch->workgroup_count_y);
        EXPECT_EQ(dispatch.workgroup_count_z, cmd_dispatch->workgroup_count_z);
        EXPECT_EQ(count, cmd_dispatch->num_bindings);

        for (size_t i = 0; i < count; ++i) {
          const auto *binding =
              reinterpret_cast<const hexagon_rt_arm_dsp_buf_ref_t *>(cursor);
          cursor += sizeof(*binding);
          EXPECT_EQ(refs[i].buffer_slot, binding->slot);
          // expect DSP virtual address for direct buffer reference
          EXPECT_EQ(i == direct_buffer_at_idx ? dsp_virt_addr : 0,
                    binding->buffer_dsp_vaddr);
          EXPECT_EQ(refs[i].offset, binding->offset);
          EXPECT_EQ(refs[i].length, binding->length);
        }
      };

  verify_dispatch(first_dispatch, first_bindings.data(), first_bindings.size(),
                  first_bindings.size() /* no direct buffer ref*/,
                  -1 /* unused */);

  const auto *cmd_barrier =
      reinterpret_cast<const hexagon_rt_arm_dsp_cmd_barrier_t *>(cursor);
  cursor += sizeof(*cmd_barrier);
  EXPECT_EQ(HEXAGON_RT_ARM_DSP_CMD_BARRIER, cmd_barrier->base.cmd_type);

  verify_dispatch(second_dispatch, second_bindings.data(),
                  second_bindings.size(), 3, 0x420001);

  // Test the test: Generated data needs to end at end of buffer.

  EXPECT_EQ(buffer.data() + buffer.size(), cursor);
}

TEST(CommandBufferSerializeTest, DispatchBarrierCopy) {
  // Test input data structure: command buffer with the following entries:
  // dispatch, barrier, buffer copy.
  // The dispatch has two buffers references, the first dynamic/indirect, the
  // second fixed/direct.

  rpc_executable_handle_t rpc_executable_handle = 0xCAFE;

  iree_hal_hexagon_command_dispatch_t dispatch = {};
  dispatch.base.cmd_type = IREE_HAL_HEXAGON_COMMAND_DISPATCH;
  dispatch.rpc_executable_handle = rpc_executable_handle;
  dispatch.export_ordinal = 4;
  dispatch.workgroup_size_x = 8;
  dispatch.workgroup_size_y = 6;
  dispatch.workgroup_size_z = 4;
  dispatch.workgroup_count_x = 2;
  dispatch.workgroup_count_y = 3;
  dispatch.workgroup_count_z = 5;

  std::array<iree_hal_buffer_ref_t, 2> binding_values = {
      iree_hal_make_indirect_buffer_ref(/*buffer_slot=*/6, /*offset=*/0,
                                        /*length=*/128),
      iree_hal_make_buffer_ref(&direct_buffers[0], /*offset=*/64,
                               /*length=*/32),
  };
  dispatch.bindings.count = binding_values.size();
  dispatch.bindings.values = binding_values.data();

  iree_hal_hexagon_command_barrier_t barrier = {};
  barrier.base.cmd_type = IREE_HAL_HEXAGON_COMMAND_BARRIER;

  iree_hal_hexagon_command_copy_t copy = {};
  copy.base.cmd_type = IREE_HAL_HEXAGON_COMMAND_COPY;
  copy.src = iree_hal_make_indirect_buffer_ref(/*buffer_slot=*/11,
                                               /*offset=*/32, /*length=*/96);
  copy.dest = iree_hal_make_buffer_ref(&direct_buffers[1], /*offset=*/48,
                                       /*length=*/96);

  dispatch.base.next = &barrier.base;
  barrier.base.prev = &dispatch.base;
  barrier.base.next = &copy.base;
  copy.base.prev = &barrier.base;

  iree_hal_hexagon_command_buffer_t command_buffer = {};
  command_buffer.first_entry = &dispatch.base;
  command_buffer.last_entry = &copy.base;

  // Test computing size of serialized data and number of entries.

  const iree_host_size_t expected_size =
      sizeof(hexagon_rt_arm_dsp_cmd_buf_t) +
      sizeof(hexagon_rt_arm_dsp_cmd_dispatch_t) +
      binding_values.size() * sizeof(hexagon_rt_arm_dsp_buf_ref_t) +
      sizeof(hexagon_rt_arm_dsp_cmd_barrier_t) +
      sizeof(hexagon_rt_arm_dsp_cmd_copy_t);

  iree_host_size_t cmd_buf_size = 0;
  uint32_t num_entries = 0;
  IREE_EXPECT_OK(iree_hal_hexagon_command_buffer_serialize_prep(
      &command_buffer, &cmd_buf_size, &num_entries));
  EXPECT_EQ(expected_size, cmd_buf_size);
  EXPECT_EQ(3u, num_entries);

  // Test generating serialized data.

  std::vector<uint8_t> buffer(cmd_buf_size);
  IREE_EXPECT_OK(iree_hal_hexagon_command_buffer_serialize_exec(
      &command_buffer, buffer_to_dsp_vaddr, num_entries, buffer.data(),
      buffer.size()));

  const auto *header =
      reinterpret_cast<const hexagon_rt_arm_dsp_cmd_buf_t *>(buffer.data());
  EXPECT_EQ(3u, header->num_entries);

  const uint8_t *cursor = buffer.data() + sizeof(*header);
  const auto *cmd_dispatch =
      reinterpret_cast<const hexagon_rt_arm_dsp_cmd_dispatch_t *>(cursor);
  cursor += sizeof(*cmd_dispatch);

  EXPECT_EQ(HEXAGON_RT_ARM_DSP_CMD_DISPATCH, cmd_dispatch->base.cmd_type);
  EXPECT_EQ(dispatch.rpc_executable_handle, cmd_dispatch->executable_handle);
  EXPECT_EQ(dispatch.export_ordinal, cmd_dispatch->export_ordinal);
  EXPECT_EQ(dispatch.workgroup_size_x, cmd_dispatch->workgroup_size_x);
  EXPECT_EQ(dispatch.workgroup_size_y, cmd_dispatch->workgroup_size_y);
  EXPECT_EQ(dispatch.workgroup_size_z, cmd_dispatch->workgroup_size_z);
  EXPECT_EQ(dispatch.workgroup_count_x, cmd_dispatch->workgroup_count_x);
  EXPECT_EQ(dispatch.workgroup_count_y, cmd_dispatch->workgroup_count_y);
  EXPECT_EQ(dispatch.workgroup_count_z, cmd_dispatch->workgroup_count_z);
  EXPECT_EQ(binding_values.size(), cmd_dispatch->num_bindings);

  for (size_t i = 0; i < binding_values.size(); ++i) {
    const auto *binding =
        reinterpret_cast<const hexagon_rt_arm_dsp_buf_ref_t *>(cursor);
    cursor += sizeof(*binding);
    EXPECT_EQ(binding_values[i].buffer_slot, binding->slot);
    EXPECT_EQ(i == 1 ? 0x420000 : 0, binding->buffer_dsp_vaddr);
    EXPECT_EQ(binding_values[i].offset, binding->offset);
    EXPECT_EQ(binding_values[i].length, binding->length);
  }

  const auto *cmd_barrier =
      reinterpret_cast<const hexagon_rt_arm_dsp_cmd_barrier_t *>(cursor);
  cursor += sizeof(*cmd_barrier);
  EXPECT_EQ(HEXAGON_RT_ARM_DSP_CMD_BARRIER, cmd_barrier->base.cmd_type);

  const auto *cmd_copy =
      reinterpret_cast<const hexagon_rt_arm_dsp_cmd_copy_t *>(cursor);
  cursor += sizeof(*cmd_copy);

  EXPECT_EQ(HEXAGON_RT_ARM_DSP_CMD_COPY, cmd_copy->base.cmd_type);

  EXPECT_EQ(copy.src.buffer_slot, cmd_copy->src.slot);
  EXPECT_EQ(0, cmd_copy->src.buffer_dsp_vaddr);
  EXPECT_EQ(copy.src.offset, cmd_copy->src.offset);
  EXPECT_EQ(copy.src.length, cmd_copy->src.length);

  EXPECT_EQ(copy.dest.buffer_slot, cmd_copy->dest.slot);
  EXPECT_EQ(0x420001, cmd_copy->dest.buffer_dsp_vaddr);
  EXPECT_EQ(copy.dest.offset, cmd_copy->dest.offset);
  EXPECT_EQ(copy.dest.length, cmd_copy->dest.length);

  // Test the test: Generated data needs to end at end of buffer.

  EXPECT_EQ(buffer.data() + buffer.size(), cursor);
}

} // namespace

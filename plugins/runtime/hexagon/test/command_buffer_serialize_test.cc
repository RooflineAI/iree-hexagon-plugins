
#include <array>
#include <cstdint>
#include <cstring>
#include <vector>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

extern "C" {
#include "cmd_barrier_serialize.h"
#include "cmd_copy_serialize.h"
#include "cmd_dispatch_serialize.h"
#include "cmd_fill_serialize.h"
#include "command_buffer_serialize.h"
#include "command_buffer_types.h"
#include "hexagon/arm_dsp/cmd_buf.h"
#include "hexagon/serialize/rpc_types.h"
}
#include "serialize_test_utils.h"

namespace {

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
      &command_buffer, num_entries, buffer.data(), buffer.size()));

  const auto *header =
      reinterpret_cast<const hexagon_rt_arm_dsp_cmd_buf_t *>(buffer.data());
  EXPECT_EQ(0u, header->num_entries);
}

TEST(CommandBufferSerializeTest, ConcatenatesSerializedEntries) {
  // Pre-serialize a fill, a dispatch, a barrier, and a copy and check they are
  // concatenated into the final command buffer.
  struct SerializedEntry {
    std::vector<uint8_t> storage;
    iree_hal_hexagon_command_buffer_entry_t *entry = nullptr;
    uint8_t *cmd_data = nullptr;
  };
  auto make_entry = [](iree_host_size_t cmd_size) {
    SerializedEntry result;
    result.storage.resize(sizeof(iree_hal_hexagon_command_buffer_entry_t) +
                          cmd_size);
    result.entry = reinterpret_cast<iree_hal_hexagon_command_buffer_entry_t *>(
        result.storage.data());
    *result.entry = {};
    result.entry->size = cmd_size;
    result.cmd_data =
        result.storage.data() + sizeof(iree_hal_hexagon_command_buffer_entry_t);
    return result;
  };

  // Fill entry.
  const std::array<uint8_t, 4> fill_pattern = {0x10, 0x20, 0x30, 0x40};
  iree_hal_buffer_ref_t fill_dest = iree_hal_make_buffer_ref(
      &hexagon_test_direct_buffers[0], /*offset=*/16, /*length=*/48);
  iree_host_size_t fill_cmd_size = 0;
  IREE_EXPECT_OK(iree_hal_hexagon_cmd_fill_serialize_prep(&fill_cmd_size));
  auto fill_entry = make_entry(fill_cmd_size);
  IREE_EXPECT_OK(iree_hal_hexagon_cmd_fill_serialize_exec(
      fill_pattern.size(), fill_pattern.data(), &fill_dest,
      hexagon_test_get_buffer_fd, fill_entry.cmd_data, fill_entry.entry->size));

  // Dispatch entry.
  const iree_const_byte_span_t empty_constants =
      iree_make_const_byte_span(nullptr, 0);
  iree_hal_dispatch_config_t dispatch_config =
      iree_hal_make_static_dispatch_config(/*x=*/2, /*y=*/3, /*z=*/5);
  dispatch_config.workgroup_size[0] = 8;
  dispatch_config.workgroup_size[1] = 6;
  dispatch_config.workgroup_size[2] = 4;
  std::array<iree_hal_buffer_ref_t, 2> dispatch_bindings = {
      iree_hal_make_indirect_buffer_ref(/*buffer_slot=*/6, /*offset=*/0,
                                        /*length=*/128),
      iree_hal_make_buffer_ref(&hexagon_test_direct_buffers[0], /*offset=*/64,
                               /*length=*/32),
  };
  iree_hal_buffer_ref_list_t dispatch_binding_list = {dispatch_bindings.size(),
                                                      dispatch_bindings.data()};
  iree_host_size_t dispatch_cmd_size = 0;
  IREE_EXPECT_OK(iree_hal_hexagon_cmd_dispatch_serialize_prep(
      &empty_constants, &dispatch_binding_list, &dispatch_cmd_size));
  auto dispatch_entry = make_entry(dispatch_cmd_size);
  IREE_EXPECT_OK(iree_hal_hexagon_cmd_dispatch_serialize_exec(
      /*executable_handle=*/0xCAFE, /*export_ordinal=*/4, &dispatch_config,
      &empty_constants, &dispatch_binding_list, hexagon_test_get_buffer_fd,
      dispatch_entry.cmd_data, dispatch_entry.entry->size));

  // Barrier entry.
  iree_host_size_t barrier_cmd_size = 0;
  IREE_EXPECT_OK(
      iree_hal_hexagon_cmd_barrier_serialize_prep(&barrier_cmd_size));
  auto barrier_entry = make_entry(barrier_cmd_size);
  IREE_EXPECT_OK(iree_hal_hexagon_cmd_barrier_serialize_exec(
      barrier_entry.cmd_data, barrier_entry.entry->size));

  // Copy entry.
  iree_hal_buffer_ref_t copy_src = iree_hal_make_indirect_buffer_ref(
      /*buffer_slot=*/11, /*offset=*/32, /*length=*/96);
  iree_hal_buffer_ref_t copy_dest = iree_hal_make_buffer_ref(
      &hexagon_test_direct_buffers[1], /*offset=*/48, /*length=*/96);
  iree_host_size_t copy_cmd_size = 0;
  IREE_EXPECT_OK(iree_hal_hexagon_cmd_copy_serialize_prep(&copy_cmd_size));
  auto copy_entry = make_entry(copy_cmd_size);
  IREE_EXPECT_OK(iree_hal_hexagon_cmd_copy_serialize_exec(
      &copy_src, &copy_dest, hexagon_test_get_buffer_fd, copy_entry.cmd_data,
      copy_entry.entry->size));

  // Chain entries.
  fill_entry.entry->next = dispatch_entry.entry;
  dispatch_entry.entry->next = barrier_entry.entry;
  barrier_entry.entry->next = copy_entry.entry;

  iree_hal_hexagon_command_buffer_t command_buffer = {};
  command_buffer.first_entry = fill_entry.entry;
  command_buffer.last_entry = copy_entry.entry;

  // Serialize command buffer.
  const iree_host_size_t expected_cmd_buf_size =
      sizeof(hexagon_rt_arm_dsp_cmd_buf_t) + fill_entry.entry->size +
      dispatch_entry.entry->size + barrier_entry.entry->size +
      copy_entry.entry->size;
  iree_host_size_t cmd_buf_size = 0;
  uint32_t num_entries = 0;
  IREE_EXPECT_OK(iree_hal_hexagon_command_buffer_serialize_prep(
      &command_buffer, &cmd_buf_size, &num_entries));
  EXPECT_EQ(expected_cmd_buf_size, cmd_buf_size);
  EXPECT_EQ(4u, num_entries);

  std::vector<uint8_t> buffer(cmd_buf_size);
  IREE_EXPECT_OK(iree_hal_hexagon_command_buffer_serialize_exec(
      &command_buffer, num_entries, buffer.data(), buffer.size()));

  // Check that the serialization of the command buffer set up a header and
  // concatenated the data of the pre-serialized entries.

  const auto *header =
      reinterpret_cast<const hexagon_rt_arm_dsp_cmd_buf_t *>(buffer.data());
  EXPECT_EQ(4u, header->num_entries);

  const uint8_t *cursor = buffer.data() + sizeof(*header);
  EXPECT_EQ(0, memcmp(cursor, fill_entry.cmd_data, fill_entry.entry->size));
  cursor += fill_entry.entry->size;
  EXPECT_EQ(
      0, memcmp(cursor, dispatch_entry.cmd_data, dispatch_entry.entry->size));
  cursor += dispatch_entry.entry->size;
  EXPECT_EQ(0,
            memcmp(cursor, barrier_entry.cmd_data, barrier_entry.entry->size));
  cursor += barrier_entry.entry->size;
  EXPECT_EQ(0, memcmp(cursor, copy_entry.cmd_data, copy_entry.entry->size));
  cursor += copy_entry.entry->size;

  EXPECT_EQ(buffer.data() + buffer.size(), cursor);
}

} // namespace

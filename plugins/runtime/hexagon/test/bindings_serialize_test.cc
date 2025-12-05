#include <array>
#include <cstdint>
#include <vector>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

extern "C" {
#include "bindings_serialize.h"
#include "hexagon/arm_dsp/bindings.h"
#include "hexagon/serialize/rpc_types.h"
}

namespace {

iree_hal_buffer_t direct_buffers[3] = {};

// Returns deterministic fake DSP virtual addresses for known buffers.
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

TEST(BindingsSerializeTest, EmptyBindingTable) {

  // Test input data: empty binding table.

  iree_hal_buffer_binding_table_t binding_table = {};

  // Test computing size of serialized data and number of entries.

  iree_host_size_t bind_tab_size = 0;
  IREE_EXPECT_OK(
      iree_hal_hexagon_bindings_serialize_prep(&binding_table, &bind_tab_size));
  EXPECT_EQ(sizeof(hexagon_rt_arm_dsp_binding_tab_t), bind_tab_size);

  // Test generating serialized data.

  std::vector<uint8_t> buffer(bind_tab_size);
  IREE_EXPECT_OK(iree_hal_hexagon_bindings_serialize_exec(
      &binding_table, buffer_to_dsp_vaddr, buffer.data(), buffer.size()));

  const auto *header =
      reinterpret_cast<const hexagon_rt_arm_dsp_binding_tab_t *>(buffer.data());
  EXPECT_EQ(0u, header->num_entries);
}

TEST(BindingsSerializeTest, MultipleBindings) {

  // Test input data: binding table with three different entries.

  std::array<iree_hal_buffer_binding_t, 3> bindings = {
      {{&direct_buffers[0], 0, 32},
       {&direct_buffers[1], 64, 96},
       {&direct_buffers[2], 256, 16}}};

  iree_hal_buffer_binding_table_t binding_table = {};
  binding_table.count = bindings.size();
  binding_table.bindings = bindings.data();

  // Test computing size of serialized data and number of entries.

  const iree_host_size_t expected_size =
      sizeof(hexagon_rt_arm_dsp_binding_tab_t) +
      bindings.size() * sizeof(hexagon_rt_arm_dsp_binding_t);

  iree_host_size_t bind_tab_size = 0;
  IREE_EXPECT_OK(
      iree_hal_hexagon_bindings_serialize_prep(&binding_table, &bind_tab_size));
  EXPECT_EQ(expected_size, bind_tab_size);

  // Test generating serialized data.

  std::vector<uint8_t> buffer(bind_tab_size);
  IREE_EXPECT_OK(iree_hal_hexagon_bindings_serialize_exec(
      &binding_table, buffer_to_dsp_vaddr, buffer.data(), buffer.size()));

  const auto *header =
      reinterpret_cast<const hexagon_rt_arm_dsp_binding_tab_t *>(buffer.data());
  EXPECT_EQ(bindings.size(), header->num_entries);

  const uint8_t *cursor = buffer.data() + sizeof(*header);
  for (size_t i = 0; i < bindings.size(); ++i) {
    const auto *binding =
        reinterpret_cast<const hexagon_rt_arm_dsp_binding_t *>(cursor);
    cursor += sizeof(*binding);

    EXPECT_EQ(static_cast<rpc_dsp_vaddr_t>(0x420000 + i),
              binding->buffer_dsp_vaddr);
    EXPECT_EQ(bindings[i].offset, binding->offset);
    EXPECT_EQ(bindings[i].length, binding->length);
  }

  // Test the test: Generated data needs to end at end of buffer.

  EXPECT_EQ(buffer.data() + buffer.size(), cursor);
}

} // namespace

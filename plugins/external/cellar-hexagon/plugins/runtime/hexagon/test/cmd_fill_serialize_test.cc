#include <array>
#include <cstdint>
#include <vector>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

extern "C" {
#include "cmd_fill_serialize.h"
#include "hexagon/arm_dsp/cmd_buf.h"
#include "hexagon/serialize/rpc_types.h"
}
#include "serialize_test_utils.h"

namespace {

TEST(CmdFillSerializeTest, SerializeDirectBufferWithFourBytePattern) {
  const std::array<uint8_t, 4> pattern = {0xAA, 0xBB, 0xCC, 0xDD};
  iree_hal_buffer_ref_t dest = iree_hal_make_buffer_ref(
      &hexagon_test_direct_buffers[1], /*offset=*/24, /*length=*/80);

  iree_host_size_t cmd_size = 0;
  IREE_EXPECT_OK(iree_hal_hexagon_cmd_fill_serialize_prep(&cmd_size));
  EXPECT_EQ(sizeof(hexagon_rt_arm_dsp_cmd_fill_t), cmd_size);

  std::vector<uint8_t> buffer(cmd_size);
  IREE_EXPECT_OK(iree_hal_hexagon_cmd_fill_serialize_exec(
      pattern.size(), pattern.data(), &dest, hexagon_test_get_buffer_fd,
      buffer.data(), buffer.size()));

  const auto *cmd_fill =
      reinterpret_cast<const hexagon_rt_arm_dsp_cmd_fill_t *>(buffer.data());
  EXPECT_EQ(HEXAGON_RT_ARM_DSP_CMD_FILL, cmd_fill->base.cmd_type);
  EXPECT_EQ(pattern.size(), cmd_fill->pattern_length);
  for (size_t i = 0; i < pattern.size(); ++i) {
    EXPECT_EQ(pattern[i], cmd_fill->pattern[i]);
  }

  EXPECT_EQ(dest.buffer_slot, cmd_fill->trgt.slot);
  EXPECT_EQ(HEXAGON_TEST_FAKE_FD(1), cmd_fill->trgt.fd);
  EXPECT_EQ(dest.offset, cmd_fill->trgt.offset);
  EXPECT_EQ(dest.length, cmd_fill->trgt.length);
}

TEST(CmdFillSerializeTest, SerializeIndirectBufferWithSingleBytePattern) {
  const std::array<uint8_t, 1> pattern = {0x11};
  iree_hal_buffer_ref_t dest = iree_hal_make_indirect_buffer_ref(
      /*buffer_slot=*/5, /*offset=*/12, /*length=*/48);

  iree_host_size_t cmd_size = 0;
  IREE_EXPECT_OK(iree_hal_hexagon_cmd_fill_serialize_prep(&cmd_size));
  EXPECT_EQ(sizeof(hexagon_rt_arm_dsp_cmd_fill_t), cmd_size);

  std::vector<uint8_t> buffer(cmd_size);
  IREE_EXPECT_OK(iree_hal_hexagon_cmd_fill_serialize_exec(
      pattern.size(), pattern.data(), &dest, hexagon_test_get_buffer_fd,
      buffer.data(), buffer.size()));

  const auto *cmd_fill =
      reinterpret_cast<const hexagon_rt_arm_dsp_cmd_fill_t *>(buffer.data());
  EXPECT_EQ(HEXAGON_RT_ARM_DSP_CMD_FILL, cmd_fill->base.cmd_type);
  EXPECT_EQ(pattern.size(), cmd_fill->pattern_length);
  EXPECT_EQ(pattern[0], cmd_fill->pattern[0]);

  EXPECT_EQ(dest.buffer_slot, cmd_fill->trgt.slot);
  EXPECT_EQ(-1, cmd_fill->trgt.fd);
  EXPECT_EQ(dest.offset, cmd_fill->trgt.offset);
  EXPECT_EQ(dest.length, cmd_fill->trgt.length);
}

} // namespace

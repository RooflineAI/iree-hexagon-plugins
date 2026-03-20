#include <cstdint>
#include <vector>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

extern "C" {
#include "cmd_copy_serialize.h"
#include "hexagon/arm_dsp/cmd_buf.h"
#include "hexagon/serialize/rpc_types.h"
}
#include "serialize_test_utils.h"

namespace {

TEST(CmdCopySerializeTest, Serialize) {
  iree_hal_buffer_ref_t src =
      iree_hal_make_buffer_ref(&hexagon_test_direct_buffers[0], /*offset=*/8,
                               /*length=*/64);
  iree_hal_buffer_ref_t dest = iree_hal_make_indirect_buffer_ref(
      /*buffer_slot=*/9, /*offset=*/24, /*length=*/64);

  iree_host_size_t cmd_size = 0;
  IREE_EXPECT_OK(iree_hal_hexagon_cmd_copy_serialize_prep(&cmd_size));
  EXPECT_EQ(sizeof(hexagon_rt_arm_dsp_cmd_copy_t), cmd_size);

  std::vector<uint8_t> buffer(cmd_size);
  IREE_EXPECT_OK(iree_hal_hexagon_cmd_copy_serialize_exec(
      &src, &dest, hexagon_test_get_buffer_fd, buffer.data(), buffer.size()));

  const auto *cmd_copy =
      reinterpret_cast<const hexagon_rt_arm_dsp_cmd_copy_t *>(buffer.data());
  EXPECT_EQ(HEXAGON_RT_ARM_DSP_CMD_COPY, cmd_copy->base.cmd_type);

  EXPECT_EQ(src.buffer_slot, cmd_copy->src.slot);
  EXPECT_EQ(HEXAGON_TEST_FAKE_FD(0), cmd_copy->src.fd);
  EXPECT_EQ(src.offset, cmd_copy->src.offset);
  EXPECT_EQ(src.length, cmd_copy->src.length);

  EXPECT_EQ(dest.buffer_slot, cmd_copy->trgt.slot);
  EXPECT_EQ(-1, cmd_copy->trgt.fd);
  EXPECT_EQ(dest.offset, cmd_copy->trgt.offset);
  EXPECT_EQ(dest.length, cmd_copy->trgt.length);
}

} // namespace

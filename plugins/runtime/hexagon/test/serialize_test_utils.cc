#include "serialize_test_utils.h"

#include <cstddef>

iree_hal_buffer_t hexagon_test_direct_buffers[3] = {};

iree_status_t hexagon_test_buffer_to_dsp_vaddr(iree_hal_buffer_t *buffer,
                                               rpc_dsp_vaddr_t *out_dsp_vaddr) {
  for (size_t i = 0; i < sizeof(hexagon_test_direct_buffers) /
                             sizeof(hexagon_test_direct_buffers[0]);
       ++i) {
    if (buffer == &hexagon_test_direct_buffers[i]) {
      *out_dsp_vaddr = static_cast<rpc_dsp_vaddr_t>(0x420000 + i);
      return iree_ok_status();
    }
  }
  return iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "unknown buffer");
}

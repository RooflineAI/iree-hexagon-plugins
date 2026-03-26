#include "serialize_test_utils.h"

#include <cstddef>

iree_hal_buffer_t hexagon_test_direct_buffers[3] = {};

iree_status_t hexagon_test_get_buffer_fd(iree_hal_buffer_t *buffer,
                                         int *out_fd) {
  for (size_t i = 0; i < sizeof(hexagon_test_direct_buffers) /
                             sizeof(hexagon_test_direct_buffers[0]);
       ++i) {
    if (buffer == &hexagon_test_direct_buffers[i]) {
      *out_fd = HEXAGON_TEST_FAKE_FD(i);
      return iree_ok_status();
    }
  }
  return iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "unknown buffer");
}

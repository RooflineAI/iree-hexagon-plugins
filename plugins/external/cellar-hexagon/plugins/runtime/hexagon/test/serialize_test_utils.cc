#include "serialize_test_utils.h"

#include <cstdint>

iree_hal_buffer_t hexagon_test_direct_buffer0;
iree_hal_buffer_t hexagon_test_direct_buffer1;
iree_hal_buffer_t hexagon_test_direct_buffer2;

iree_hal_buffer_t hexagon_test_direct_subspan_buffer0;

static iree_device_size_t
hexagon_test_get_buffer_subspan_offset(iree_hal_buffer_t *buffer) {
  if (buffer == &hexagon_test_direct_subspan_buffer0) {
    return HEXAGON_TEST_DIRECT_SUBSPAN_BUFFER0_OFFSET;
  }
  return 0;
}

iree_status_t iree_hal_buffer_ref_normalize(iree_hal_buffer_ref_t *buffer_ref) {
  if (buffer_ref) {
    buffer_ref->offset +=
        hexagon_test_get_buffer_subspan_offset(buffer_ref->buffer);
  }
  return iree_ok_status();
}

iree_status_t
iree_hal_buffer_binding_normalize(iree_hal_buffer_binding_t *binding) {
  if (binding) {
    binding->offset += hexagon_test_get_buffer_subspan_offset(binding->buffer);
  }
  return iree_ok_status();
}

iree_status_t hexagon_test_get_buffer_fd(iree_hal_buffer_t *buffer,
                                         int *out_fd) {
  if (buffer == &hexagon_test_direct_buffer0) {
    *out_fd = HEXAGON_TEST_DIRECT_BUFFER0_FD;
    return iree_ok_status();
  }
  if (buffer == &hexagon_test_direct_buffer1) {
    *out_fd = HEXAGON_TEST_DIRECT_BUFFER1_FD;
    return iree_ok_status();
  }
  if (buffer == &hexagon_test_direct_buffer2) {
    *out_fd = HEXAGON_TEST_DIRECT_BUFFER2_FD;
    return iree_ok_status();
  }
  if (buffer == &hexagon_test_direct_subspan_buffer0) {
    *out_fd = HEXAGON_TEST_DIRECT_SUBSPAN_BUFFER0_FD;
    return iree_ok_status();
  }
  return iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "unknown buffer");
}

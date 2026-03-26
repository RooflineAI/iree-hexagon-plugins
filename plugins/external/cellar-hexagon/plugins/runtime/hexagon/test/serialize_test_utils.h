#ifndef IREE_HAL_DRIVERS_HEXAGON_TEST_SERIALIZE_TEST_UTILS_H_
#define IREE_HAL_DRIVERS_HEXAGON_TEST_SERIALIZE_TEST_UTILS_H_

#include "iree/base/api.h"
#include "iree/hal/api.h"

// Shared helpers for serialization unit tests.

// Fake file descriptor for buffer with index idx
#define HEXAGON_TEST_FAKE_FD(idx) (0x42 + (idx))

// Global direct buffers used by tests. The array is sized to cover all current
// usages; additional entries can be added if needed.
extern iree_hal_buffer_t hexagon_test_direct_buffers[3];

// Returns deterministic fake DSP file descriptor for known buffers.
iree_status_t hexagon_test_get_buffer_fd(iree_hal_buffer_t *buffer,
                                         int *out_fd);

#endif // IREE_HAL_DRIVERS_HEXAGON_TEST_SERIALIZE_TEST_UTILS_H_

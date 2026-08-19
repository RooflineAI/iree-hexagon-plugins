#ifndef IREE_HAL_DRIVERS_HEXAGON_TEST_SERIALIZE_TEST_UTILS_H_
#define IREE_HAL_DRIVERS_HEXAGON_TEST_SERIALIZE_TEST_UTILS_H_

#include <cstdint>

#include "iree/base/api.h"
#include "iree/hal/api.h"

// Shared helpers for serialization unit tests.

/// Fake direct buffers to use in the tests
extern iree_hal_buffer_t hexagon_test_direct_buffer0;
#define HEXAGON_TEST_DIRECT_BUFFER0_FD (42)
extern iree_hal_buffer_t hexagon_test_direct_buffer1;
#define HEXAGON_TEST_DIRECT_BUFFER1_FD (43)
extern iree_hal_buffer_t hexagon_test_direct_buffer2;
#define HEXAGON_TEST_DIRECT_BUFFER2_FD (44)

/// Fake direct subspan buffers to use in the tests
extern iree_hal_buffer_t hexagon_test_direct_subspan_buffer0;
#define HEXAGON_TEST_DIRECT_SUBSPAN_BUFFER0_FD (45)
#define HEXAGON_TEST_DIRECT_SUBSPAN_BUFFER0_OFFSET (1000)

// Mock function for normalizing buffer references.
iree_status_t iree_hal_buffer_ref_normalize(iree_hal_buffer_ref_t *buffer_ref);

// Mock function for normalizing buffer bindings.
iree_status_t
iree_hal_buffer_binding_normalize(iree_hal_buffer_binding_t *binding);

// Returns deterministic fake DSP file descriptor and fake subspan buffer
// offsets for known buffers.
iree_status_t hexagon_test_get_buffer_fd(iree_hal_buffer_t *buffer,
                                         int *out_fd);

#endif // IREE_HAL_DRIVERS_HEXAGON_TEST_SERIALIZE_TEST_UTILS_H_

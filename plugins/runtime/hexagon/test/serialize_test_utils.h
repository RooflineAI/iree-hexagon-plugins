#ifndef IREE_HAL_DRIVERS_HEXAGON_TEST_SERIALIZE_TEST_UTILS_H_
#define IREE_HAL_DRIVERS_HEXAGON_TEST_SERIALIZE_TEST_UTILS_H_

#include <cstddef>

#include "hexagon/serialize/rpc_types.h"
#include "iree/base/api.h"
#include "iree/hal/api.h"

// Shared helpers for serialization unit tests.

// Global direct buffers used by tests. The array is sized to cover all current
// usages; additional entries can be added if needed.
extern iree_hal_buffer_t hexagon_test_direct_buffers[3];

// Returns deterministic fake DSP virtual addresses for known buffers.
iree_status_t hexagon_test_buffer_to_dsp_vaddr(iree_hal_buffer_t *buffer,
                                               rpc_dsp_vaddr_t *out_dsp_vaddr);

#endif // IREE_HAL_DRIVERS_HEXAGON_TEST_SERIALIZE_TEST_UTILS_H_

// Copyright 2025 RooflineAI GmbH

#ifndef IREE_HAL_DRIVERS_HEXAGON_PROFILING_H_
#define IREE_HAL_DRIVERS_HEXAGON_PROFILING_H_

#include <stdint.h>

#include "hexagon/api.h"
#include "iree/base/api.h"
#include "iree/hal/api.h"

#if defined(IREE_HAL_HEXAGON_ENABLE_PROFILING)
#include "iree/base/tracing.h"
#else
typedef struct iree_tracing_context_t iree_tracing_context_t;
#endif // IREE_HAL_HEXAGON_ENABLE_PROFILING

// Tracy GPU query ids are uint16_t; we use 2 ids per zone (begin/end).
#define IREE_HAL_HEXAGON_TRACY_MAX_ZONES 0x7FFFu
// Sentinel for an uninitialized tracy GPU context id.
#define IREE_HAL_HEXAGON_TRACY_CONTEXT_INVALID 0xFFu

iree_status_t iree_hal_hexagon_alloc_and_init_profiling_data(
    iree_hal_command_buffer_t *command_buffer,
    const iree_hal_hexagon_device_options_t *device_options,
    uint8_t **profiling_data, iree_host_size_t *profiling_data_size);

iree_status_t iree_hal_hexagon_export_profiling_data(
    iree_allocator_t host_allocator, uint8_t *profiling_data,
    uint8_t *tracy_context_id, iree_tracing_context_t **tracy_plot_context);

#endif // IREE_HAL_DRIVERS_HEXAGON_PROFILING_H_

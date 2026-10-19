// Copyright 2025 RooflineAI GmbH
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_HEXAGON_PROFILER_H_
#define IREE_HAL_DRIVERS_HEXAGON_PROFILER_H_

#include <stdint.h>

#include "hexagon/api.h"
#include "iree/base/api.h"
#include "iree/hal/api.h"

#if defined(IREE_HAL_HEXAGON_ENABLE_PROFILER)
#include "iree/base/tracing.h"
#else
typedef struct iree_tracing_context_t iree_tracing_context_t;
#endif // IREE_HAL_HEXAGON_ENABLE_PROFILER

// Tracy GPU query ids are uint16_t; we use 2 ids per zone (begin/end).
#define IREE_HAL_HEXAGON_TRACY_MAX_ZONES 0x7FFFu
// Sentinel for an uninitialized tracy GPU context id.
#define IREE_HAL_HEXAGON_TRACY_CONTEXT_INVALID 0xFFu
// Max. number of DSP threads that get their own tracy track.
#define IREE_HAL_HEXAGON_TRACY_MAX_LANES 16u

// Tracy GPU contexts, one per DSP thread, so that zones running in parallel on
// different DSP threads show up side by side. Cached in the device, because
// the number of GPU contexts per process is limited. Lanes are numbered in the
// order their threads are first seen, so lane 0 is the thread walking the
// command buffer, which records the first zone (DSP_EXECUTION). All zero means
// no lanes yet.
typedef struct iree_hal_hexagon_tracy_lanes_t {
  uint32_t thread_ids[IREE_HAL_HEXAGON_TRACY_MAX_LANES];
  uint8_t context_ids[IREE_HAL_HEXAGON_TRACY_MAX_LANES];
  uint32_t count;
} iree_hal_hexagon_tracy_lanes_t;

iree_status_t iree_hal_hexagon_alloc_and_init_profiler_data(
    iree_hal_command_buffer_t *command_buffer,
    const iree_hal_hexagon_device_options_t *device_options,
    uint8_t **profiler_data, iree_host_size_t *profiler_data_size);

iree_status_t iree_hal_hexagon_export_profiler_data(
    iree_allocator_t host_allocator, uint8_t *profiler_data,
    iree_hal_hexagon_tracy_lanes_t *tracy_lanes,
    iree_tracing_context_t **tracy_plot_context);

#endif // IREE_HAL_DRIVERS_HEXAGON_PROFILER_H_

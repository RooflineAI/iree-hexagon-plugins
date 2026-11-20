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
//
// Tracy keys GPU zone annotations (the PMU counters) by context and query id,
// so query ids keep counting up per lane across command buffers instead of
// restarting at 0: a zone would otherwise show the counters of a later zone.
typedef struct iree_hal_hexagon_tracy_lanes_t {
  uint32_t thread_ids[IREE_HAL_HEXAGON_TRACY_MAX_LANES];
  uint8_t context_ids[IREE_HAL_HEXAGON_TRACY_MAX_LANES];
  // Next free query id of each lane; wraps around after 0x7FFF zones.
  uint16_t next_query_ids[IREE_HAL_HEXAGON_TRACY_MAX_LANES];
  // Whether the query ids of a lane wrapped around.
  bool query_ids_wrapped[IREE_HAL_HEXAGON_TRACY_MAX_LANES];
  uint32_t annotated_pmu_event_ids[IREE_HAL_HEXAGON_PMU_COUNTERS];
  uint32_t count;
} iree_hal_hexagon_tracy_lanes_t;

iree_status_t iree_hal_hexagon_alloc_and_init_profiler_data(
    iree_hal_command_buffer_t *command_buffer,
    const iree_hal_hexagon_device_options_t *device_options,
    uint8_t **profiler_data, iree_host_size_t *profiler_data_size);

iree_status_t iree_hal_hexagon_export_profiler_data(
    iree_allocator_t host_allocator, uint8_t *profiler_data,
    iree_hal_hexagon_tracy_lanes_t *tracy_lanes);

#endif // IREE_HAL_DRIVERS_HEXAGON_PROFILER_H_

// Copyright 2025 RooflineAI GmbH
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "hexagon/profiler.h"

#include "iree/base/status.h"

#if !defined(IREE_HAL_HEXAGON_ENABLE_PROFILER)

iree_status_t iree_hal_hexagon_alloc_and_init_profiler_data(
    iree_hal_command_buffer_t *command_buffer,
    const iree_hal_hexagon_device_options_t *device_options,
    uint8_t **profiler_data, iree_host_size_t *profiler_data_size) {
  (void)command_buffer;
  (void)device_options;
  (void)profiler_data;
  (void)profiler_data_size;
  return iree_make_status(IREE_STATUS_UNAVAILABLE,
                          "profiler is not enabled in this build");
}

iree_status_t iree_hal_hexagon_export_profiler_data(
    iree_allocator_t host_allocator, uint8_t *profiler_data,
    iree_hal_hexagon_tracy_lanes_t *tracy_lanes) {
  (void)profiler_data;
  (void)tracy_lanes;
  return iree_make_status(IREE_STATUS_UNAVAILABLE,
                          "profiler is not enabled in this build");
}

#else // IREE_HAL_HEXAGON_ENABLE_PROFILER

#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "hexagon/arm_dsp/profiler.h"
#include "hexagon/serialize/command_buffer_types.h"
#include "iree/base/tracing.h"

#include "rpcmem.h"

// This is done to avoid weird dependencies between the files declaring these
_Static_assert(IREE_HAL_HEXAGON_PMU_COUNTERS == HEXAGON_PMU_COUNTERS,
               "PMU counter count mismatch");

typedef struct hexagon_rt_prof_record_view_t {
  uint32_t record_index;
  int64_t start_us;
  int64_t end_us;
  uint32_t zone_type;
  uint16_t query_begin;
  uint16_t query_end;
  // tracy GPU context of the DSP thread that recorded the zone, and its lane
  uint8_t context_id;
  uint8_t lane;
  const char *extra_info;
} hexagon_rt_prof_record_view_t;

static int iree_hal_hexagon_compare_profiler_records_by_lane_start_and_end(
    const void *lhs_ptr, const void *rhs_ptr) {
  const hexagon_rt_prof_record_view_t *lhs =
      (const hexagon_rt_prof_record_view_t *)lhs_ptr;
  const hexagon_rt_prof_record_view_t *rhs =
      (const hexagon_rt_prof_record_view_t *)rhs_ptr;
  if (lhs->context_id < rhs->context_id)
    return -1;
  if (lhs->context_id > rhs->context_id)
    return 1;
  if (lhs->start_us < rhs->start_us)
    return -1;
  if (lhs->start_us > rhs->start_us)
    return 1;
  if (lhs->end_us < rhs->end_us)
    return -1;
  if (lhs->end_us > rhs->end_us)
    return 1;
  return 0;
}

// This helper is meant to be used to choose the colors in the tracy trace plot
// I chose them manually, but feel free to change them or do something a bit
// more sophisticated
static uint32_t iree_hal_hexagon_zone_color_xbgr(uint32_t zone_type) {
  switch (zone_type) {
  case DSP_EXECUTION:
    return 0xa46e5cu;
  case SCHEDULE:
    return 0xb68b7cu;
  case KERNEL:
    return 0x808cb1u;
  case BARRIER:
    return 0x93a9c8u;
  case COPY:
    return 0xdde2ddu;
  case FILL:
    return 0xdde2ddu;
  case MEMORY_MANAGEMENT:
    return 0xdde2ddu;
  case MARKER:
    return 0x9ac7b7u;
  case UNKNOWN:
  default:
    return 0xA97E90u;
  }
}

// Names the PMU annotations of a tracy GPU context after the PMU events. The
// annotation id is the PMU counter index, so that the same event selected
// twice still gets two distinct annotations.
static void
iree_hal_hexagon_name_pmu_annotations(uint8_t context_id,
                                      const uint32_t *pmu_event_ids) {
  for (uint32_t j = 0; j < HEXAGON_PMU_COUNTERS; ++j) {
    const char *pmu_name = hexagon_pmu_event_name(pmu_event_ids[j]);
    if (!pmu_name) {
      pmu_name = "hexagon_pmu_unknown";
    }
    iree_tracing_gpu_context_annotation_name(context_id, (int64_t)j, pmu_name,
                                             strlen(pmu_name));
  }
}

// Makes the PMU annotations of all lanes match the events of this command
// buffer. Lanes created later are named on creation.
static void iree_hal_hexagon_update_pmu_annotations(
    iree_hal_hexagon_tracy_lanes_t *lanes,
    const hexagon_pmu_counters_ids_t *pmu_event_ids) {
  if (memcmp(lanes->annotated_pmu_event_ids, pmu_event_ids->ids,
             sizeof(lanes->annotated_pmu_event_ids)) == 0) {
    return;
  }
  memcpy(lanes->annotated_pmu_event_ids, pmu_event_ids->ids,
         sizeof(lanes->annotated_pmu_event_ids));
  for (uint32_t i = 0; i < lanes->count; ++i) {
    iree_hal_hexagon_name_pmu_annotations(lanes->context_ids[i],
                                          lanes->annotated_pmu_event_ids);
  }
}

// Returns the lane of a DSP thread, allocating its tracy GPU context on first
// use. Returns IREE_HAL_HEXAGON_TRACY_CONTEXT_INVALID if there are more
// threads than lanes.
static uint8_t
iree_hal_hexagon_tracy_lane(iree_hal_hexagon_tracy_lanes_t *lanes,
                            uint32_t thread_id, uint64_t cpu_anchor) {
  // search for previous context
  for (uint32_t i = 0; i < lanes->count; ++i) {
    if (lanes->thread_ids[i] == thread_id)
      return (uint8_t)i;
  }
  if (lanes->count >= IREE_HAL_HEXAGON_TRACY_MAX_LANES)
    return IREE_HAL_HEXAGON_TRACY_CONTEXT_INVALID;
  // create new context
  char name[32];
  snprintf(name, sizeof(name), "hexagon-dsp thread %u",
           (unsigned int)lanes->count);
  // Timestamps passed to it are already rebased to cpu time
  // (see iree_hal_hexagon_rebase_prof_timestamps)
  // so it is calibrated with cpu and gpu timestamps both set to the anchor.
  uint8_t context_id = iree_tracing_gpu_context_allocate(
      IREE_TRACING_GPU_CONTEXT_TYPE_VULKAN, name, strlen(name),
      /*is_calibrated=*/true,
      /*cpu_timestamp=*/cpu_anchor,
      /*gpu_timestamp=*/cpu_anchor,
      /*timestamp_period=*/1);
  iree_hal_hexagon_name_pmu_annotations(context_id,
                                        lanes->annotated_pmu_event_ids);
  lanes->thread_ids[lanes->count] = thread_id;
  lanes->context_ids[lanes->count] = context_id;
  lanes->next_query_ids[lanes->count] = 0;
  lanes->query_ids_wrapped[lanes->count] = false;
  return (uint8_t)lanes->count++;
}

static iree_status_t iree_hal_hexagon_collect_and_sort_prof_views(
    iree_allocator_t host_allocator, const hexagon_rt_prof_record_t *records,
    uint32_t count, iree_hal_hexagon_tracy_lanes_t *lanes, uint64_t cpu_anchor,
    hexagon_rt_prof_record_view_t **out_views, uint32_t *out_view_count) {
  *out_views = NULL;
  *out_view_count = 0;

  uint32_t max_zones = count;
  if (max_zones > IREE_HAL_HEXAGON_TRACY_MAX_ZONES) {
    fprintf(stderr,
            "WARNING: Too many profiler records for tracy query ids (%u). "
            "Truncating to %u.\n",
            max_zones, (unsigned int)IREE_HAL_HEXAGON_TRACY_MAX_ZONES);
    max_zones = IREE_HAL_HEXAGON_TRACY_MAX_ZONES;
  }

  hexagon_rt_prof_record_view_t *views;
  iree_allocator_malloc(host_allocator, sizeof(*views) * max_zones,
                        (void **)&views);
  if (!views) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "out of host memory for profiler views");
  }

  uint32_t dropped = 0;
  uint32_t view_count = 0;
  for (uint32_t i = 0; i < count && view_count < max_zones; ++i) {
    const hexagon_rt_prof_record_t *record = &records[i];
    if (!record->record_completed)
      continue;
    const int64_t start_us = (int64_t)record->start_timer_ticks_timestamp;
    const int64_t end_us = (int64_t)record->stop_timer_ticks_timestamp;
    if (end_us <= start_us)
      continue;
    // count the number of records dropped due to having too many threads
    // to print a warning below
    const uint8_t lane =
        iree_hal_hexagon_tracy_lane(lanes, record->thread_id, cpu_anchor);
    if (lane == IREE_HAL_HEXAGON_TRACY_CONTEXT_INVALID) {
      ++dropped;
      continue;
    }
    views[view_count].context_id = lanes->context_ids[lane];
    views[view_count].lane = lane;
    views[view_count].record_index = i;
    views[view_count].start_us = start_us;
    views[view_count].end_us = end_us;
    views[view_count].zone_type = record->zone_type;
    views[view_count].extra_info = record->extra_info;
    ++view_count;
  }

  if (dropped > 0) {
    fprintf(stderr,
            "WARNING: More than %u DSP threads; dropped %u profiler "
            "records of the additional threads.\n",
            (unsigned int)IREE_HAL_HEXAGON_TRACY_MAX_LANES,
            (unsigned int)dropped);
  }

  if (view_count > 0) {
    qsort(views, view_count, sizeof(*views),
          iree_hal_hexagon_compare_profiler_records_by_lane_start_and_end);

    for (uint32_t i = 0; i < view_count; ++i) {
      const uint8_t lane = views[i].lane;
      views[i].query_begin = lanes->next_query_ids[lane];
      views[i].query_end = (uint16_t)(lanes->next_query_ids[lane] + 1);
      lanes->next_query_ids[lane] += 2;
      if (lanes->next_query_ids[lane] == 0 && !lanes->query_ids_wrapped[lane]) {
        lanes->query_ids_wrapped[lane] = true;
        fprintf(stderr,
                "WARNING: Tracy query ids of DSP thread lane %u wrapped "
                "around after %u zones; PMU counters shown on earlier zones "
                "of this lane may belong to later zones.\n",
                (unsigned int)lane, 0x8000u);
      }
    }
  }

  *out_views = views;
  *out_view_count = view_count;
  return iree_ok_status();
}

// Only kernels and the markers inside them get PMU annotations: the 32 bit
// counters can wrap around more than once over the long outer zones.
static bool iree_hal_hexagon_zone_has_pmu_annotations(uint32_t zone_type) {
  return zone_type == KERNEL || zone_type == MARKER;
}

// Emits tracy zones in the correct order to generate a nested trace. Views are
// sorted by lane (context) first, a lane at a time is nested on the stack.
static iree_status_t iree_hal_hexagon_emit_tracy_zones(
    iree_allocator_t host_allocator, const hexagon_rt_prof_record_t *records,
    hexagon_rt_prof_record_view_t *views, uint32_t view_count) {
  uint32_t *stack;
  iree_allocator_malloc(host_allocator, sizeof(*stack) * view_count,
                        (void **)&stack);
  if (!stack) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "out of host memory for profiler stack");
  }

  uint32_t stack_size = 0;
  for (uint32_t i = 0; i < view_count; ++i) {
    hexagon_rt_prof_record_view_t *view = &views[i];
    // End all zones that are still on the stack and ended before the start of
    // the current zone, or belong to the previous lane
    while (stack_size > 0) {
      hexagon_rt_prof_record_view_t *top = &views[stack[stack_size - 1]];
      if (top->context_id == view->context_id && top->end_us > view->start_us)
        break;
      iree_tracing_gpu_zone_end(top->context_id, top->query_end);
      iree_tracing_gpu_zone_notify(top->context_id, top->query_end,
                                   top->end_us);
      --stack_size;
    }

    // Begin the current zone and put it on the stack, in order to keep track
    // that it is still "active" (i.e. not ended yet)

    // Note that in tracy, query ids are supposed to represent timestamps,
    // but this profiler currently combines two timestamps into a single
    // record describing a zone. Additionally, the zones are defined after
    // execution so note that we are passing the timestamps stored in the
    // records during notification. Do not get confused with the begin and
    // end as markers of the zone in cpu time, their order only affects the
    // construction of the trace plot.
    const char *zone_name = view->extra_info[0]
                                ? view->extra_info
                                : zone_to_string(view->zone_type);
    const uint32_t zone_color =
        iree_hal_hexagon_zone_color_xbgr(view->zone_type);
    iree_tracing_gpu_zone_begin_external_colored(
        view->context_id, view->query_begin, __FILE__, strlen(__FILE__),
        __LINE__, __func__, strlen(__func__), zone_name, strlen(zone_name),
        zone_color);
    iree_tracing_gpu_zone_notify(view->context_id, view->query_begin,
                                 view->start_us);
    // Attach the PMU counter deltas of the zone. The counters are 32 bit
    // and wrap around, hence the unsigned 32 bit difference.
    // FIXME: PMU counters are not per thread, so the deltas of kernels running
    // in parallel include each other's work.
    if (iree_hal_hexagon_zone_has_pmu_annotations(view->zone_type)) {
      const hexagon_rt_prof_record_t *record = &records[view->record_index];
      for (uint32_t j = 0; j < HEXAGON_PMU_COUNTERS; ++j) {
        const uint32_t pmu_delta = record->stop_pmu_registers_stamp.cts[j] -
                                   record->start_pmu_registers_stamp.cts[j];
        iree_tracing_gpu_zone_annotation(view->context_id, view->query_begin,
                                         (int64_t)j, (double)pmu_delta);
      }
    }
    stack[stack_size++] = i;
  }

  // All zones processed, now end all zones that are still on the stack
  while (stack_size > 0) {
    hexagon_rt_prof_record_view_t *top = &views[stack[stack_size - 1]];
    iree_tracing_gpu_zone_end(top->context_id, top->query_end);
    iree_tracing_gpu_zone_notify(top->context_id, top->query_end, top->end_us);
    --stack_size;
  }

  iree_allocator_free(host_allocator, stack);
  return iree_ok_status();
}

static iree_status_t
iree_hal_hexagon_rebase_prof_timestamps(hexagon_rt_prof_header_t *header,
                                        hexagon_rt_prof_record_t *records,
                                        uint32_t count) {
  // Vulkan periodically calls the iree_tracing_gpu_context_calibrate function
  // to rectify calibration. This is currently not supported for Hexagon.
  // Instead, we calibrate dsp and cpu times by using the start and end of the
  // command buffer execution exclusively.

  // We want to get the tracing time as quickly as possible
  const uint64_t cpu_duration_ns =
      iree_tracing_time() - header->start_cmd_buffer_exec_cpu_time;

  if (header->num_records < 1 || header->started_records < 1 ||
      !records[0].record_completed || records[0].zone_type != DSP_EXECUTION) {
    fprintf(stderr,
            "WARNING: Profiler records missing entry indicating total "
            "execution time of command buffer execution from dsp timer.\n "
            "Records expected: %d, Records started: %d, Record completed %d, "
            "Record type: %d\n",
            header->num_records, header->started_records,
            records[0].record_completed, records[0].zone_type);
  }

  const double dsp_ticks_to_ns = 1000. / tick_timer_freq_MHz;

  const int64_t dsp_duration_ticks = records[0].stop_timer_ticks_timestamp -
                                     records[0].start_timer_ticks_timestamp;
  if (dsp_duration_ticks == 0) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "DSP records during profiler were not correctly "
                            "updated");
  }

  // Center the DSP timeline in the expected zone cpu one, note that this is
  // an estimation and not necessarily realistic!
  const uint64_t dsp_duration_ns = (double)dsp_duration_ticks * dsp_ticks_to_ns;
  if (dsp_duration_ns >= cpu_duration_ns) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "Unexpected durations obtained during profiler");
  }

  uint64_t rpc_offset_estimation = (cpu_duration_ns - dsp_duration_ns) / 2;

  const uint64_t cpu_timestamp_anchor =
      header->start_cmd_buffer_exec_cpu_time + rpc_offset_estimation;

  // This value is updated because it will be used as anchor when creating the
  // gpu zone
  header->start_cmd_buffer_exec_cpu_time = cpu_timestamp_anchor;

  // Rebase all timestamps, also count incomplete records
  const int64_t base_ticks = (int64_t)records[0].start_timer_ticks_timestamp;
  uint32_t incomplete = 0;
  for (uint32_t i = 0; i < count; ++i) {
    hexagon_rt_prof_record_t *record = &records[i];
    if (!record->record_completed) {
      ++incomplete;
      continue;
    }
    const int64_t start_ticks = (int64_t)record->start_timer_ticks_timestamp;
    const int64_t end_ticks = (int64_t)record->stop_timer_ticks_timestamp;
    const int64_t start_offset_ns =
        (double)(start_ticks - base_ticks) * dsp_ticks_to_ns;
    const int64_t end_offset_ns =
        (double)(end_ticks - base_ticks) * dsp_ticks_to_ns;
    record->start_timer_ticks_timestamp =
        cpu_timestamp_anchor + start_offset_ns;
    record->stop_timer_ticks_timestamp = cpu_timestamp_anchor + end_offset_ns;
  }

  if (incomplete > 0) {
    fprintf(stderr, "WARNING: Ignoring %u incomplete profiler records.\n",
            (unsigned int)incomplete);
  }

  return iree_ok_status();
}

iree_status_t iree_hal_hexagon_alloc_and_init_profiler_data(
    iree_hal_command_buffer_t *command_buffer,
    const iree_hal_hexagon_device_options_t *device_options,
    uint8_t **profiler_data, iree_host_size_t *profiler_data_size) {
  iree_hal_hexagon_command_buffer_t *hexagon_command_buffer =
      (iree_hal_hexagon_command_buffer_t *)command_buffer;

  *profiler_data_size = sizeof(hexagon_rt_prof_header_t) +
                        hexagon_command_buffer->profiler_record_capacity *
                            sizeof(hexagon_rt_prof_record_t);
  if (*profiler_data_size >
      INT_MAX /* max size supported by rpcmem_alloc() */) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "profiler data structure too big");
  }

  *profiler_data = rpcmem_alloc(RPCMEM_HEAP_ID_SYSTEM, RPCMEM_DEFAULT_FLAGS,
                                *profiler_data_size);
  if (!*profiler_data) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "out of RPC memory");
  }

  // Initialize shared buffer.
  memset(*profiler_data, 0, *profiler_data_size);
  hexagon_rt_prof_header_t *header = (hexagon_rt_prof_header_t *)*profiler_data;

  header->num_records = hexagon_command_buffer->profiler_record_capacity;
  uint32_t count = device_options ? device_options->pmu_event_ids_count : 0;
  for (uint32_t i = 0; i < HEXAGON_PMU_COUNTERS; ++i) {
    header->pmu_event_ids.ids[i] =
        i < count ? device_options->pmu_event_ids[i] : default_ids.ids[i];
  }
  header->start_cmd_buffer_exec_cpu_time = iree_tracing_time();

  return iree_ok_status();
}

iree_status_t iree_hal_hexagon_export_profiler_data(
    iree_allocator_t host_allocator, uint8_t *profiler_data,
    iree_hal_hexagon_tracy_lanes_t *tracy_lanes) {
  hexagon_rt_prof_header_t *header = (hexagon_rt_prof_header_t *)profiler_data;
  if (header->dropped_records > 0) {
    fprintf(stderr,
            "WARNING: Profiler record capacity was exhausted; dropped %u "
            "markers. Increase profiler_extra_records_per_dispatch for more "
            "complete traces.\n",
            (unsigned int)header->dropped_records);
  }
  if (header->started_records == 0) {
    fprintf(stderr, "WARNING: No records were started during profiler, "
                    "skipping data export");
    return iree_ok_status();
  }

  uint32_t count = header->started_records < header->num_records
                       ? header->started_records
                       : header->num_records;

  uint8_t *records_base = (uint8_t *)header + sizeof(hexagon_rt_prof_header_t);
  hexagon_rt_prof_record_t *records = (hexagon_rt_prof_record_t *)records_base;

  iree_status_t rebase_status =
      iree_hal_hexagon_rebase_prof_timestamps(header, records, count);
  if (!iree_status_is_ok(rebase_status)) {
    return rebase_status;
  }

  // Timestamps are rebased to cpu time above, instead of calibrating the tracy
  // GPU contexts with iree_tracing_gpu_context_calibrate.
  const uint64_t cpu_anchor = header->start_cmd_buffer_exec_cpu_time;

  iree_hal_hexagon_update_pmu_annotations(tracy_lanes, &header->pmu_event_ids);

  // Add DSP activity to timeline
  if (count > 0) {
    hexagon_rt_prof_record_view_t *views = NULL;
    uint32_t view_count = 0;
    iree_status_t view_status = iree_hal_hexagon_collect_and_sort_prof_views(
        host_allocator, records, count, tracy_lanes, cpu_anchor, &views,
        &view_count);
    if (!iree_status_is_ok(view_status)) {
      return view_status;
    }

    if (view_count > 0) {
      iree_status_t emit_status = iree_hal_hexagon_emit_tracy_zones(
          host_allocator, records, views, view_count);
      iree_allocator_free(host_allocator, views);
      if (!iree_status_is_ok(emit_status)) {
        return emit_status;
      }
    } else {
      iree_allocator_free(host_allocator, views);
    }
  }

  return iree_ok_status();
}

#endif // IREE_HAL_HEXAGON_ENABLE_PROFILER

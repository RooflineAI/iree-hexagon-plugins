// Copyright 2025 RooflineAI GmbH
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "hexagon/dsp/rt/profiler.h"
#include "hexagon/arm_dsp/profiler.h"
#include "hexagon/dsp/rt/runtime_state.h"
#include "qurt_atomic_ops.h"
#include <string.h>

#if !defined(IREE_HAL_HEXAGON_ENABLE_PROFILER)

void profiler_context_init(hexagon_rt_prof_header_t *header,
                           hexagon_rt_prof_record_t *records,
                           hexagon_rt_prof_context_t *out_prof_context) {
  (void)header;
  (void)records;
  (void)out_prof_context;
}

void profiler_context_deinit(hexagon_rt_prof_context_t *prof_context) {
  (void)prof_context;
}

inline hexagon_rt_prof_record_t *
profiler_measurement_start(hexagon_rt_prof_context_t *prof_context,
                           uint32_t zone_type, const char *extra_info) {
  (void)prof_context;
  (void)zone_type;
  (void)extra_info;
  return NULL;
}

inline void
profiler_measurement_finish_and_record(hexagon_rt_prof_record_t *record) {
  (void)record;
}

hexagon_rt_prof_record_t *
hexagon_runtime_profiler_zone_begin(hexagon_rt_state_t *runtime_state,
                                    uint32_t zone_type,
                                    const char *extra_info) {
  (void)runtime_state;
  (void)zone_type;
  (void)extra_info;
  return NULL;
}

void hexagon_runtime_profiler_zone_end(hexagon_rt_prof_record_t *record) {
  (void)record;
};

#else

#include "HAP_farf.h"
#include "hexagon/dsp/pmu/hexagon_pmu.h"
#include "hexagon/dsp/pmu/hexagon_timer.h"

void profiler_context_init(hexagon_rt_prof_header_t *header,
                           hexagon_rt_prof_record_t *records,
                           hexagon_rt_prof_context_t *out_prof_context) {
  if (!header || !records || !out_prof_context) {
    FARF(RUNTIME_HIGH,
         "HEXAGON-RUNTIME-ERROR: NULL pointer in profiler_context_init");
    return;
  }

  // Keep the pointers to the main data structres.
  out_prof_context->header = header;
  out_prof_context->records = records;

  // Set up atomic counters that get modified in multithreaded environment.
  qurt_atomic_set(&out_prof_context->next_record_idx, 0);
  qurt_atomic_set(&out_prof_context->dropped_records, 0);
}

void profiler_context_deinit(hexagon_rt_prof_context_t *prof_context) {
  if (!prof_context) {
    FARF(RUNTIME_HIGH,
         "HEXAGON-RUNTIME-ERROR: NULL pointer in profiler_context_deinit");
    return;
  }

  // Copy the atomic counter into the header structure shared with ARM host.
  // There is no qurt_atomic_read(), so read directly.
  prof_context->header->started_records = prof_context->next_record_idx;
  prof_context->header->dropped_records = prof_context->dropped_records;

  // Invalidate the context
  prof_context->header = NULL;
  prof_context->records = NULL;
}

/**
 * @brief Obtain next unused profiler record from the context.
 *        Emit an error to the log if out of records.
 * @param prof_context The profiler context data structure to which to add the
 *                     profiler record.
 * @return pointer to profiler record or NULL if out of records
 */
static hexagon_rt_prof_record_t *
profiler_context_get_fresh_record(hexagon_rt_prof_context_t *prof_context) {
  // Do a fast pre-check if we are not already out of records.
  // There is no qurt_atomic_read(), so read directly.
  if (prof_context->next_record_idx < prof_context->header->num_records) {
    // Acquire a record by atomically incrementing the next index.
    unsigned new_next_idx =
        qurt_atomic_inc_return(&prof_context->next_record_idx);
    // The incremented index is returned. This means we allocated the index
    // before.
    unsigned idx = new_next_idx - 1;
    // Our index is still within the range of the records.
    // -> We acquired a record. Return it.
    if (idx < prof_context->header->num_records) {
      return &prof_context->records[idx];
    }
    // The index is beyond the end of the records. This means a race happened
    // and we lost. We did not get a record. There are no more records. Undo
    // the increment we just did. Note that this decrement will never make
    // the atomic next_idx go below num_records, so this is safe.
    qurt_atomic_dec(&prof_context->next_record_idx);
  }

  // We did not get a record. So we need to drop the measurement. Count this,
  // print and error and do not return a record pointer.
  qurt_atomic_inc(&prof_context->dropped_records);
  FARF(RUNTIME_HIGH,
       "HEXAGON-RUNTIME-ERROR: Not enough profiler records allocated for "
       "measurements");
  return NULL;
}

hexagon_rt_prof_record_t *
profiler_measurement_start(hexagon_rt_prof_context_t *prof_context,
                           uint32_t zone_type, const char *extra_info) {
  if (!prof_context) {
    FARF(RUNTIME_HIGH, "HEXAGON-RUNTIME-ERROR: Unexpected null pointer during "
                       "profiler, ignoring profiler marker");
    return NULL;
  }

  hexagon_rt_prof_record_t *record =
      profiler_context_get_fresh_record(prof_context);
  if (!record) {
    return NULL;
  }

  record->zone_type = zone_type;
  record->start_timer_ticks_timestamp = read_timer();
  hexagon_pmu_read(&record->start_pmu_registers_stamp);
  // The null terminators are not needed since everything is initialized to 0
  // already. Keeping them anyway, in case that changes in the future
  if (extra_info) {
    strncpy(record->extra_info, extra_info, sizeof(record->extra_info) - 1);
    record->extra_info[sizeof(record->extra_info) - 1] = '\0';
  } else {
    record->extra_info[0] = '\0';
  }

  return record;
}

void profiler_measurement_finish_and_record(hexagon_rt_prof_record_t *record) {
  if (!record) {
    return;
  }
  if (record->record_completed) {
    FARF(RUNTIME_HIGH, "HEXAGON-RUNTIME-ERROR: profiler_measurement_finish "
                       "called on a finished record");
    return;
  }
  record->stop_timer_ticks_timestamp = read_timer();
  hexagon_pmu_read(&record->stop_pmu_registers_stamp);
  record->record_completed = 1;
}

hexagon_rt_prof_record_t *
hexagon_runtime_profiler_zone_begin(hexagon_rt_state_t *runtime_state,
                                    uint32_t zone_type,
                                    const char *extra_info) {
  if (!runtime_state || !runtime_state->prof_context) {
    return NULL;
  }
  return profiler_measurement_start(runtime_state->prof_context, zone_type,
                                    extra_info);
}

void hexagon_runtime_profiler_zone_end(hexagon_rt_prof_record_t *record) {
  profiler_measurement_finish_and_record(record);
}

#endif

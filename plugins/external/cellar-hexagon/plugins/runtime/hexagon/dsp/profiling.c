// Copyright 2025 RooflineAI GmbH

#include "hexagon/dsp/profiling.h"
#include "hexagon/arm_dsp/profiling.h"
#include <string.h>

#if !defined(IREE_HAL_HEXAGON_ENABLE_PROFILING)

void inline profiler_measurement_start(hexagon_rt_prof_header_t *header,
                                       hexagon_rt_prof_record_t *records,
                                       uint32_t zone_type) {
  (void)header;
  (void)records;
  (void)zone_type;
  return;
}

void inline profiler_measurement_finish_and_record(
    hexagon_rt_prof_header_t *header, hexagon_rt_prof_record_t *records) {
  (void)header;
  (void)records;
  return;
}

void inline profiler_measurement_start_extra_info(
    hexagon_rt_prof_header_t *header, hexagon_rt_prof_record_t *records,
    uint32_t zone_type, const char *extra_info) {
  (void)header;
  (void)records;
  (void)zone_type;
  (void)extra_info;
  return;
}

void profiler_set_active_context(hexagon_rt_prof_header_t *header,
                                 hexagon_rt_prof_record_t *records) {
  (void)header;
  (void)records;
  return;
}

void profiler_clear_active_context(void) { return; }

void hexagon_runtime_profiling_zone_begin(uint32_t zone_type,
                                          const char *extra_info) {
  (void)zone_type;
  (void)extra_info;
  return;
}

void hexagon_runtime_profiling_zone_end(void) { return; }

#else

#include "HAP_farf.h"
#include "hexagon/dsp/pmu/hexagon_pmu.h"
#include "hexagon/dsp/pmu/hexagon_timer.h"

static hexagon_rt_prof_header_t *active_profiling_header = NULL;
static hexagon_rt_prof_record_t *active_profiling_records = NULL;

void profiler_measurement_start_extra_info(hexagon_rt_prof_header_t *header,
                                           hexagon_rt_prof_record_t *records,
                                           uint32_t zone_type,
                                           const char *extra_info) {
  if (!header || !records) {
    FARF(RUNTIME_HIGH, "HEXAGON-RUNTIME-ERROR: Unexpected null pointer during "
                       "profiling, ignoring profiling marker");
    return;
  }

  if (header->started_records >= header->num_records) {
    ++header->dropped_records;
    ++header->dropped_open_records;
    FARF(RUNTIME_HIGH,
         "HEXAGON-RUNTIME-WARNING: Not enough profiling records allocated for "
         "measurements, dropping profiling marker");
    return;
  }

  hexagon_rt_prof_record_t *record = &records[header->started_records];
  record->zone_type = zone_type;
  record->start_timer_ticks_timestamp = read_timer();
  hexagon_pmu_read(&records[header->started_records].start_pmu_registers_stamp);
  // The null terminators are not needed since everything is initialized to 0
  // already. Keeping them anyway, in case that changes in the future
  if (extra_info) {
    strncpy(record->extra_info, extra_info, sizeof(record->extra_info) - 1);
    record->extra_info[sizeof(record->extra_info) - 1] = '\0';
  } else {
    record->extra_info[0] = '\0';
  }
  ++header->started_records;
}

void profiler_measurement_start(hexagon_rt_prof_header_t *header,
                                hexagon_rt_prof_record_t *records,
                                uint32_t zone_type) {
  profiler_measurement_start_extra_info(header, records, zone_type, NULL);
}

void profiler_measurement_finish_and_record(hexagon_rt_prof_header_t *header,
                                            hexagon_rt_prof_record_t *records) {
  if (!header || !records) {
    FARF(RUNTIME_HIGH, "HEXAGON-RUNTIME-ERROR: Unexpected null pointer during "
                       "profiling finish, ignoring profiling marker");
    return;
  }

  if (header->dropped_open_records > 0) {
    --header->dropped_open_records;
    return;
  }

  if (header->completed_records < header->started_records) {
    for (int i = header->started_records - 1; i >= 0; --i) {
      if (records[i].record_completed)
        continue;
      records[i].stop_timer_ticks_timestamp = read_timer();
      hexagon_pmu_read(&records[i].stop_pmu_registers_stamp);
      records[i].record_completed = 1;
      ++header->completed_records;
      return;
    }
  }

  FARF(RUNTIME_HIGH, "HEXAGON-RUNTIME-ERROR: Mismatch between "
                     "profiler_measurement_start and finish");
}

void profiler_set_active_context(hexagon_rt_prof_header_t *header,
                                 hexagon_rt_prof_record_t *records) {
  active_profiling_header = header;
  active_profiling_records = records;
}

void profiler_clear_active_context(void) {
  active_profiling_header = NULL;
  active_profiling_records = NULL;
}

void hexagon_runtime_profiling_zone_begin(uint32_t zone_type,
                                          const char *extra_info) {
  if (!active_profiling_header || !active_profiling_records) {
    return;
  }
  profiler_measurement_start_extra_info(
      active_profiling_header, active_profiling_records, zone_type, extra_info);
}

void hexagon_runtime_profiling_zone_end(void) {
  if (!active_profiling_header || !active_profiling_records) {
    return;
  }
  profiler_measurement_finish_and_record(active_profiling_header,
                                         active_profiling_records);
}

#endif

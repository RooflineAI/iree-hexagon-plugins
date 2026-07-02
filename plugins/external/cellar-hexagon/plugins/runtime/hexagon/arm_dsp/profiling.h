// Copyright 2025 RooflineAI GmbH

#ifndef IREE_HAL_DRIVERS_HEXAGON_ARM_DSP_PROFILING_H_
#define IREE_HAL_DRIVERS_HEXAGON_ARM_DSP_PROFILING_H_

#include "hexagon/arm_dsp/pmu/hexagon_pmu_events_ids.h"
#include <stdint.h>

// Profiling buffer layout shared between ARM host and DSP.
// ARM allocates/initializes; DSP writes records and updates the header.

static const double tick_timer_freq_MHz = 19.2;

typedef struct hexagon_pmu_counters_s {
  uint32_t cts[HEXAGON_PMU_COUNTERS];
} __attribute__((packed)) hexagon_pmu_counters_t;

typedef struct hexagon_pmu_counters_ids_s {
  uint32_t ids[HEXAGON_PMU_COUNTERS];
} __attribute__((packed)) hexagon_pmu_counters_ids_t;

static const hexagon_pmu_counters_ids_t default_ids = {
    HEX_PMU_EVENT_COMMITTED_INSTS,
    HEX_PMU_EVENT_COMMITTED_PKT_ANY,
    HEX_PMU_EVENT_HVX_ACTIVE,
    HEX_PMU_EVENT_HVX_PKT,
    HEX_PMU_EVENT_L2_DU_READ_MISS,
    HEX_PMU_EVENT_L2_DU_STORE_MISS,
    HEX_PMU_EVENT_VTCM_FIFO_FULL_CYCLES,
    HEX_PMU_EVENT_ANY_DU_REPLAY};

#define HEXAGON_PROFILING_ZONES                                                \
  HEXAGON_PROFILING_ZONE(DSP_EXECUTION, "DSP execution")                       \
  HEXAGON_PROFILING_ZONE(DISPATCH, "Dispatch")                                 \
  HEXAGON_PROFILING_ZONE(KERNEL, "Kernel")                                     \
  HEXAGON_PROFILING_ZONE(BARRIER, "Barrier")                                   \
  HEXAGON_PROFILING_ZONE(COPY, "Copy")                                         \
  HEXAGON_PROFILING_ZONE(FILL, "Fill")                                         \
  HEXAGON_PROFILING_ZONE(MEMORY_MANAGEMENT, "Mem Management")                  \
  HEXAGON_PROFILING_ZONE(MARKER, "Marker")                                     \
  HEXAGON_PROFILING_ZONE(UNKNOWN, "Unknown")

typedef enum iree_hal_hexagon_profiling_zone_types_s {
#define HEXAGON_PROFILING_ZONE(id, name) id,
  HEXAGON_PROFILING_ZONES
#undef HEXAGON_PROFILING_ZONE
      ZONES_COUNT
} iree_hal_hexagon_profiling_zone_types_t;

static const char *const zone_names[ZONES_COUNT] = {
#define HEXAGON_PROFILING_ZONE(id, name) name,
    HEXAGON_PROFILING_ZONES
#undef HEXAGON_PROFILING_ZONE
};

static inline const char *
zone_to_string(iree_hal_hexagon_profiling_zone_types_t type) {
  return (type >= 0 && type < ZONES_COUNT) ? zone_names[type]
                                           : "Invalid zone name";
}

typedef struct hexagon_rt_prof_header_s {
  hexagon_pmu_counters_ids_t pmu_event_ids;
  // Capacity of the records array following this header.
  uint32_t num_records;
  uint32_t started_records;
  uint32_t dropped_records;
  uint64_t start_cmd_buffer_exec_cpu_time;
  // DSP-side bookkeeping for overflowed begin/end pairs. Overflowed records are
  // not written to the records array, but matching ends still need to be
  // consumed before closing the next real record.
  uint32_t dropped_open_records;
  // Followed by num_records hexagon_rt_prof_record_t
} __attribute__((packed)) hexagon_rt_prof_header_t;

// This struct is written to only by the dsp side
typedef struct hexagon_rt_prof_record_s {
  uint64_t start_timer_ticks_timestamp;
  uint64_t stop_timer_ticks_timestamp;

  // These must be 4 byte aligned, otherwise syscalls will fail
  hexagon_pmu_counters_t start_pmu_registers_stamp;
  hexagon_pmu_counters_t stop_pmu_registers_stamp;

  // This is extra information dependent on the zone_type.
  // For kernels, it is the name of the function
  char extra_info[64];

  uint16_t zone_type;
  uint8_t record_completed;
  // Needed for the structs above
  uint8_t _pad;
} __attribute__((packed)) hexagon_rt_prof_record_t;

#endif // IREE_HAL_DRIVERS_HEXAGON_ARM_DSP_PROFILING_H_

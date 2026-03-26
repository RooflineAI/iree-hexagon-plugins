// Copyright 2025 RooflineAI GmbH

#ifndef IREE_HAL_DRIVERS_HEXAGON_DSP_PROFILING_H_
#define IREE_HAL_DRIVERS_HEXAGON_DSP_PROFILING_H_

#include <stdint.h>

#include "hexagon/arm_dsp/profiling.h"

/**
 * @brief Starts a profiling measurement for the given zone.
 *
 * Writes the zone type, start timestamp, and PMU counters into the next
 * available record and increments header->started_records. If no record slot
 * is available, increments header->dropped_records and logs an error.
 *
 * @param header Profiling header shared between ARM host and DSP.
 * @param records Profiling records array following the header.
 * @param zone_type Zone identifier to associate with this record.
 */
void profiler_measurement_start(hexagon_rt_prof_header_t *header,
                                hexagon_rt_prof_record_t *records,
                                uint32_t zone_type);

/**
 * @brief Starts a profiling measurement for the given zone.
 *
 * Writes the zone type, start timestamp, and PMU counters into the next
 * available record and increments header->started_records. If no record slot
 * is available, increments header->dropped_records and logs an error.
 *
 * @param header Profiling header shared between ARM host and DSP.
 * @param records Profiling records array following the header.
 * @param zone_type Zone identifier to associate with this record.
 * @param extra_info Additional information to be copied into the record.
 * Supports up to 63 characters and longer strings will be truncated.
 */
void profiler_measurement_start_extra_info(hexagon_rt_prof_header_t *header,
                                           hexagon_rt_prof_record_t *records,
                                           uint32_t zone_type,
                                           const char *extra_info);

/**
 * @brief Finishes the most recent in-flight profiling measurement.
 *
 * Finds the latest record that has been started but not completed, writes the
 * stop timestamp and PMU counters, marks it completed, and increments
 * header->completed_records. Logs an error if the start/finish counts are
 * inconsistent.
 *
 * @param header Profiling header shared between ARM host and DSP.
 * @param records Profiling records array following the header.
 */
void profiler_measurement_finish_and_record(hexagon_rt_prof_header_t *header,
                                            hexagon_rt_prof_record_t *records);

#endif // IREE_HAL_DRIVERS_HEXAGON_DSP_PROFILING_H_

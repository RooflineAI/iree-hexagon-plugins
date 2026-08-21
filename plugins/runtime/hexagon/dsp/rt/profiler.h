// Copyright 2025 RooflineAI GmbH
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef HEXAGON_DSP_RT_PROFILER_H
#define HEXAGON_DSP_RT_PROFILER_H

#include "hexagon/arm_dsp/profiler.h"
#include "hexagon/dsp/rt/runtime_state_fwd_decl.h"
#include <stdint.h>

/**
 * @brief Profiler context that can be used to profile things in multithreaded
 *        environments.
 * This structure serves multiple purposes:
 *   - Separation of run-time state of profiler and raw profiler memory
 *     blocked passed from ARM host (header, records).
 *   - Only have a single pointer to pass to the profiler start function.
 *     So only one pointer needs to be plumbed through to wherever profiler is
 *     needed.
 *   - Decouple data types of counters in the data structures shared with ARM
 *     host and used in QuRT atomic calls. (Technically, it is uint32 and
 *     unsigned int, which is the same underlying type by coincidence. It's
 *     still a cleaner design to not have QuRT atomic calls work on fields of a
 *     data structure shared with ARM host.)
 * How it works / is used:
 *   - The structure contains pointers to the not directly thread-safe
 *     profiler header and the memory for the profiler records. Reading
 *     information that does not change (like number of records in buffer)
 *     is thread-safe. As each profiler entry is accessed by only one thread,
 *     filling it directly in the records array (that is shared with ARM host)
 *     is fine, because ARM host will not access it until DSP side is done.
 *   - There is an atomic copy of the counters for thread-safe usage. It is
 *     initialized at the beginning and copied back to the header at the end.
 *     Both of this happens in non-threaded environment.
 */
typedef struct hexagon_rt_prof_context_s {
// Nothing of this structure gets used if tracing is disabled, so an empty
// structure would be fine, just standard C does not permit it, so make it
// small.
#if !defined(IREE_HAL_HEXAGON_ENABLE_PROFILER)
  char dummy;
#else
  hexagon_rt_prof_header_t *header;  /**< profiler headers passed by ARM host */
  hexagon_rt_prof_record_t *records; /**< records storage passed by ARM host */
  /** index of next record to use, out of records if equal to num_records,
   * QuRT atomic */
  unsigned int next_record_idx;
  /** number of dropped records due to insufficient number of records */
  unsigned int dropped_records;
#endif
} hexagon_rt_prof_context_t;

/**
 * @brief Initialize a profiler context that can be used to profile things
 *        in multithreaded environment.
 * This function needs to be called in non-threaded environment.
 * @param header Profiler header shared between ARM host and DSP.
 * @param records Profiler records array following the header.
 * @param[out] out_prof_context The profiler context data structure to be
 *                              initialized.
 */
void profiler_context_init(hexagon_rt_prof_header_t *header,
                           hexagon_rt_prof_record_t *records,
                           hexagon_rt_prof_context_t *out_prof_context);

/**
 * @brief De-initialize a profiler context and copy back the counters to the
 *        original header.
 * This function needs to be called in non-threaded environment.
 * @param prof_context The profiler context data structure to be de-initialized
 */
void profiler_context_deinit(hexagon_rt_prof_context_t *prof_context);

/**
 * @brief Starts a profiler measurement for the given zone.
 *
 * Writes the zone type, start timestamp, and PMU counters into the next
 * available record. If no record slot is available, it counts the dropped
 * record, logs an error and returns NULL.
 *
 * @param prof_context The profiler context data structure to which to add the
 *                     profiler record.
 * @param zone_type Zone identifier to associate with this record.
 * @param extra_info Additional information (string) to be copied into the
 *                   record. Supports up to 63 characters and longer strings
 *                   will be truncated. May be NULL if no extra information
 *                   is present.
 * @return profiler record being filled, NULL if no record available
 */
hexagon_rt_prof_record_t *
profiler_measurement_start(hexagon_rt_prof_context_t *prof_context,
                           uint32_t zone_type, const char *extra_info);

/**
 * @brief Finishes the most recent in-flight profiler measurement.
 *
 * Writes the stop timestamp and PMU counters to the passed record and marks it
 * completed. Logs an error if the profiler record is already marked completed.
 * If NULL is passed, this becomes a no-op.
 *
 * @param record Profiler record being filled. May be NULL.
 */
void profiler_measurement_finish_and_record(hexagon_rt_prof_record_t *record);

/**
 * @brief API called from inside dispatches.
 * @{
 */

/**
 * @brief Starts a profiler zone using the active dispatch profiler context.
 *
 * This is intended for generated code. It is a no-op when no active profiler
 * context is set.
 *
 * @param runtime_state The runtime state data structure from which to get the
 *                      profiler context data structure. May be NULL.
 * @param zone_type Zone identifier to associate with this record.
 * @param extra_info Additional information (string) to be copied into the
 *                   record. Supports up to 63 characters and longer strings
 *                   will be truncated. May be NULL if no extra information
 *                   is present.
 * @return profiler record being filled, NULL if no runtime_state or no record
 *         storage space available
 */
hexagon_rt_prof_record_t *
hexagon_runtime_profiler_zone_begin(hexagon_rt_state_t *runtime_state,
                                    uint32_t zone_type, const char *extra_info);

/**
 * @brief Finishes the passed profiler zone in the active dispatch
 * profiler context.
 *
 * This is intended for generated code. It is a no-op when no active profiler
 * context is set.
 *
 * @param record Profiler record being filled. May be NULL.
 */
void hexagon_runtime_profiler_zone_end(hexagon_rt_prof_record_t *record);

/**
 * @}
 */

#endif // HEXAGON_DSP_RT_PROFILER_H

// Copyright 2025 RooflineAI GmbH

#ifndef IREE_HAL_DRIVERS_HEXAGON_DSP_PROFILING_H_
#define IREE_HAL_DRIVERS_HEXAGON_DSP_PROFILING_H_

#include "hexagon/arm_dsp/profiling.h"
#include <stdint.h>

/**
 * @brief Profiling context that can be used to profile things in multithreaded
 *        environments.
 * This structure serves multiple purposes:
 *   - Separation of run-time state of profiling and raw profiling memory
 *     blocked passed from ARM host (header, records).
 *   - Only have a single pointer to pass to the profiling start function.
 *     So only one pointer needs to be plumbed through to wherever profiling is
 *     needed.
 *   - Decouple data types of counters in the data structures shared with ARM
 *     host and used in QuRT atomic calls. (Technically, it is uint32 and
 *     unsigned int, which is the same underlying type by coincidence. It's
 *     still a cleaner design to not have QuRT atomic calls work on fields of a
 *     data structure shared with ARM host.)
 * How it works / is used:
 *   - The structure contains pointers to the not directly thread-safe
 *     profiling header and the memory for the profiling records. Reading
 *     information that does not change (like number of records in buffer)
 *     is thread-safe. As each profiling entry is accessed by only one thread,
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
#if !defined(IREE_HAL_HEXAGON_ENABLE_PROFILING)
  char dummy;
#else
  hexagon_rt_prof_header_t *header; /**< profiling headers passed by ARM host */
  hexagon_rt_prof_record_t *records; /**< records storage passed by ARM host */
  /** index of next record to use, out of records if equal to num_records,
   * QuRT atomic */
  unsigned int next_record_idx;
  /** number of dropped records due to insufficient number of records */
  unsigned int dropped_records;
#endif
} hexagon_rt_prof_context_t;

/**
 * @brief Initialize a profiling context that can be used to profile things
 *        in multithreaded environment.
 * This function needs to be called in non-threaded environment.
 * @param header Profiling header shared between ARM host and DSP.
 * @param records Profiling records array following the header.
 * @param[out] out_prof_context The profiling context data structure to be
 *                              initialized.
 */
void profiler_context_init(hexagon_rt_prof_header_t *header,
                           hexagon_rt_prof_record_t *records,
                           hexagon_rt_prof_context_t *out_prof_context);

/**
 * @brief De-initialize a profiling context and copy back the counters to the
 *        original header.
 * This function needs to be called in non-threaded environment.
 * @param prof_context The profiling context data structure to be de-initialized
 */
void profiler_context_deinit(hexagon_rt_prof_context_t *prof_context);

/**
 * @brief Starts a profiling measurement for the given zone.
 *
 * Writes the zone type, start timestamp, and PMU counters into the next
 * available record. If no record slot is available, it counts the dropped
 * record, logs an error and returns NULL.
 *
 * @param prof_context The profiling context data structure to which to add the
 *                     profiling record.
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
 * @brief Finishes the most recent in-flight profiling measurement.
 *
 * Writes the stop timestamp and PMU counters to the passed record, marks it
 * completed. Then increments header->completed_records.
 * Logs an error if the profiling record is already marked completed.
 * If NULL is passed, this becomes a no-op.
 *
 * @param record Profiling record being filled. May be NULL.
 */
void profiler_measurement_finish_and_record(hexagon_rt_prof_record_t *record);

/**
 * @brief API using a globally active context.
 * FIXME: Phase out usage of this API and then delete it. (ROO-1669)
 * @{
 */

/**
 * @brief Sets the active profiling context used by runtime symbols called from
 * dispatch code.
 */
void profiler_set_active_context(hexagon_rt_prof_context_t *prof_ctx);

/**
 * @brief Clears the active dispatch profiling context.
 */
void profiler_clear_active_context(void);

/**
 * @brief Starts a profiling zone using the active dispatch profiling context.
 *
 * This is intended for generated code. It is a no-op when no active profiling
 * context is set.
 *
 * @param zone_type Zone identifier to associate with this record.
 * @param extra_info Additional information (string) to be copied into the
 *                   record. Supports up to 63 characters and longer strings
 *                   will be truncated. May be NULL if no extra information
 *                   is present.
 * @return profiler record being filled, NULL if no record available or no
 *         context active
 */
hexagon_rt_prof_record_t *
hexagon_runtime_profiling_zone_begin(uint32_t zone_type,
                                     const char *extra_info);

/**
 * @brief Finishes the passed profiling zone in the active dispatch
 * profiling context.
 *
 * This is intended for generated code. It is a no-op when no active profiling
 * context is set.
 *
 * @param record Profiling record being filled. May be NULL.
 */
void hexagon_runtime_profiling_zone_end(hexagon_rt_prof_record_t *record);

/**
 * @}
 */

#endif // IREE_HAL_DRIVERS_HEXAGON_DSP_PROFILING_H_

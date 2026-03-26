// Copyright 2025 RooflineAI GmbH

#include "hexagon/semaphore.h"

#include "iree/hal/utils/semaphore_base.h"

//===----------------------------------------------------------------------===//
// iree_hal_hexagon_timepoint_t
//===----------------------------------------------------------------------===//

// Represents a point in the timeline that someone is waiting to be reached.
// When the semaphore is signaled to at least the specified value then the
// given event will be signaled and the timepoint discarded.
//
// Instances are owned and retained by the caller that requested them.
typedef struct iree_hal_hexagon_timepoint_t {
  iree_hal_semaphore_timepoint_t base;
  iree_hal_semaphore_t *semaphore;
  iree_event_t event;
} iree_hal_hexagon_timepoint_t;

// Handles timepoint callbacks when either the timepoint is reached or it fails.
// We set the event in either case and let the waiters deal with the fallout.
static iree_status_t iree_hal_hexagon_semaphore_timepoint_callback(
    void *user_data, iree_hal_semaphore_t *semaphore, uint64_t value,
    iree_status_code_t status_code) {
  iree_hal_hexagon_timepoint_t *timepoint =
      (iree_hal_hexagon_timepoint_t *)user_data;
  iree_event_set(&timepoint->event);
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// iree_hal_hexagon_semaphore_t
//
// The current implementation works at the host side only. It is largely
// inspired by the semaphore implementation of the "task" driver.
//===----------------------------------------------------------------------===//

typedef struct iree_hal_hexagon_semaphore_t {
  iree_hal_semaphore_t base;
  iree_allocator_t host_allocator;
  iree_event_pool_t *event_pool;

  /// mutex to guard the mutable fields
  iree_slim_mutex_t mutex;

  /// mutable fieleds
  //@{

  // Current signaled value. May be IREE_HAL_SEMAPHORE_FAILURE_VALUE to
  // indicate that the semaphore has been signaled for failure and
  // |failure_status| contains the error.
  uint64_t current_value;

  // OK or the status passed to iree_hal_semaphore_fail. Owned by the semaphore.
  iree_status_t failure_status;

  //@}
} iree_hal_hexagon_semaphore_t;

static const iree_hal_semaphore_vtable_t iree_hal_hexagon_semaphore_vtable;

static iree_hal_hexagon_semaphore_t *
iree_hal_hexagon_semaphore_cast(iree_hal_semaphore_t *base_value) {
  IREE_HAL_ASSERT_TYPE(base_value, &iree_hal_hexagon_semaphore_vtable);
  return (iree_hal_hexagon_semaphore_t *)base_value;
}

iree_status_t iree_hal_hexagon_semaphore_create(
    iree_hal_queue_affinity_t queue_affinity, uint64_t initial_value,
    iree_hal_semaphore_flags_t flags, iree_allocator_t host_allocator,
    iree_event_pool_t *event_pool, iree_hal_semaphore_t **out_semaphore) {
  IREE_ASSERT_ARGUMENT(out_semaphore);
  IREE_TRACE_ZONE_BEGIN(z0);
  *out_semaphore = NULL;

  iree_hal_hexagon_semaphore_t *semaphore = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_allocator_malloc(host_allocator, sizeof(*semaphore),
                                (void **)&semaphore));
  iree_hal_semaphore_initialize(&iree_hal_hexagon_semaphore_vtable,
                                &semaphore->base);
  semaphore->host_allocator = host_allocator;
  semaphore->event_pool = event_pool;

  // TODO(hexagon): implement semaphores. Note that there is some basic support
  // provided for timepoints as part of iree/hal/utils/semaphore_base.h but the
  // actual synchronization aspects are handled by the implementation.
  //
  // If the DEVICE_LOCAL flag and a |queue_affinity| is assigned (and not just
  // IREE_HAL_QUEUE_AFFINITY_ANY) then the implementation can assume that it is
  // only used on that set of queues (never waited/signaled from anywhere else).
  // If DEVICE_LOCAL is not set then other devices may signal or wait.
  //
  // If the IREE_HAL_SEMAPHORE_FLAG_HOST_INTERRUPT flag is not set then waits
  // from the host are allowed to spin instead of performing optimized platform
  // blocking (via interrupt mechanisms).

  iree_slim_mutex_initialize(&semaphore->mutex);
  semaphore->current_value = initial_value;
  semaphore->failure_status = iree_ok_status();

  *out_semaphore = &semaphore->base;
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static void
iree_hal_hexagon_semaphore_destroy(iree_hal_semaphore_t *base_semaphore) {
  iree_hal_hexagon_semaphore_t *semaphore =
      iree_hal_hexagon_semaphore_cast(base_semaphore);
  iree_allocator_t host_allocator = semaphore->host_allocator;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_slim_mutex_deinitialize(&semaphore->mutex);
  iree_status_ignore(semaphore->failure_status);

  iree_hal_semaphore_deinitialize(&semaphore->base);
  iree_allocator_free(host_allocator, semaphore);

  IREE_TRACE_ZONE_END(z0);
}

static iree_status_t
iree_hal_hexagon_semaphore_query(iree_hal_semaphore_t *base_semaphore,
                                 uint64_t *out_value) {
  *out_value = 0;
  iree_hal_hexagon_semaphore_t *semaphore =
      iree_hal_hexagon_semaphore_cast(base_semaphore);

  // TODO(hexagon): return the current value of the semaphore by (depending on
  // the implementation) making a syscall to get it. It's expected that the
  // value may immediately change after being queried here.

  // TODO(hexagon): if the value is IREE_HAL_SEMAPHORE_FAILURE_VALUE then return
  // the failure status cached from the fail call by cloning it (like `return
  // iree_status_clone(semaphore->failure_status)`).

  iree_slim_mutex_lock(&semaphore->mutex);

  *out_value = semaphore->current_value;

  iree_status_t status = iree_ok_status();
  if (*out_value >= IREE_HAL_SEMAPHORE_FAILURE_VALUE) {
    status = iree_status_clone(semaphore->failure_status);
  }

  iree_slim_mutex_unlock(&semaphore->mutex);

  return status;
}

static iree_status_t
iree_hal_hexagon_semaphore_signal(iree_hal_semaphore_t *base_semaphore,
                                  uint64_t new_value) {
  iree_hal_hexagon_semaphore_t *semaphore =
      iree_hal_hexagon_semaphore_cast(base_semaphore);

  // TODO(hexagon): validation is optional but encouraged if cheap: semaphores
  // must always be signaled to a value that is greater than the previous value
  // (not less-than-or-equal).

  // TODO(hexagon): signals when the semaphore have failed should also fail and
  // because failed semaphores have their value set to
  // IREE_HAL_SEMAPHORE_FAILURE_VALUE that should happen naturally during
  // validation. If not then an IREE_STATUS_DATA_LOSS or IREE_STATUS_ABORTED
  // depending on how fatal such an occurrence is in the implementation.
  // Data-loss usually indicates an abort()-worthy situation where graceful
  // handling is not possible while Aborted indicates that an individual work
  // stream may be invalid but unrelated work streams may still progress.

  iree_slim_mutex_lock(&semaphore->mutex);

  if (new_value <= semaphore->current_value) {
    uint64_t current_value IREE_ATTRIBUTE_UNUSED = semaphore->current_value;
    iree_slim_mutex_unlock(&semaphore->mutex);
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "semaphore values must be monotonically "
                            "increasing; current_value=%" PRIu64
                            ", new_value=%" PRIu64,
                            current_value, new_value);
  }

  semaphore->current_value = new_value;

  iree_slim_mutex_unlock(&semaphore->mutex);

  // Notify timepoints - note that this must happen outside the lock.
  iree_hal_semaphore_notify(&semaphore->base, new_value, IREE_STATUS_OK);

  return iree_ok_status();
}

static void
iree_hal_hexagon_semaphore_fail(iree_hal_semaphore_t *base_semaphore,
                                iree_status_t status) {
  iree_hal_hexagon_semaphore_t *semaphore =
      iree_hal_hexagon_semaphore_cast(base_semaphore);
  const iree_status_code_t status_code = iree_status_code(status);

  // TODO(hexagon): if the semaphore has already failed and has a status set
  // then `IREE_IGNORE_ERROR(status)` and return without modifying anything.
  // Note that it's possible for fail to be called concurrently from multiple
  // threads.

  // TODO(hexagon): set the value to `IREE_HAL_SEMAPHORE_FAILURE_VALUE` as
  // expected by the API.

  // TODO(hexagon): take ownership of the status (no need to clone, the caller
  // is giving it to us) and keep it until the semaphore is destroyed.

  iree_slim_mutex_lock(&semaphore->mutex);

  // Try to set our local status - we only preserve the first failure so only
  // do this if we are going from a valid semaphore to a failed one.
  if (!iree_status_is_ok(semaphore->failure_status)) {
    // Previous status was not OK; drop our new status.
    IREE_IGNORE_ERROR(status);
    iree_slim_mutex_unlock(&semaphore->mutex);
    return;
  }

  // Signal to our failure sentinel value.
  semaphore->current_value = IREE_HAL_SEMAPHORE_FAILURE_VALUE;
  semaphore->failure_status = status;

  iree_slim_mutex_unlock(&semaphore->mutex);

  // Notify timepoints - note that this must happen outside the lock.
  iree_hal_semaphore_notify(&semaphore->base, IREE_HAL_SEMAPHORE_FAILURE_VALUE,
                            status_code);
}

// Acquires a timepoint waiting for the given value.
// |out_timepoint| is owned by the caller and must be kept live until the
// timepoint has been reached (or it is cancelled by the caller).
static iree_status_t iree_hal_hexagon_semaphore_acquire_timepoint(
    iree_hal_hexagon_semaphore_t *semaphore, uint64_t minimum_value,
    iree_timeout_t timeout, iree_hal_hexagon_timepoint_t *out_timepoint) {
  IREE_RETURN_IF_ERROR(
      iree_event_pool_acquire(semaphore->event_pool, 1, &out_timepoint->event));
  out_timepoint->semaphore = &semaphore->base;
  iree_hal_semaphore_acquire_timepoint(
      &semaphore->base, minimum_value, timeout,
      (iree_hal_semaphore_callback_t){
          .fn = iree_hal_hexagon_semaphore_timepoint_callback,
          .user_data = out_timepoint,
      },
      &out_timepoint->base);
  return iree_ok_status();
}

static iree_status_t
iree_hal_hexagon_semaphore_wait(iree_hal_semaphore_t *base_semaphore,
                                uint64_t value, iree_timeout_t timeout,
                                iree_hal_wait_flags_t flags) {
  iree_hal_hexagon_semaphore_t *semaphore =
      iree_hal_hexagon_semaphore_cast(base_semaphore);

  // TODO(hexagon): if a failure status is set return
  // `iree_status_from_code(IREE_STATUS_ABORTED)`. Avoid a full status as it may
  // capture a backtrace and allocate and callers are expected to follow up a
  // failed wait with a query to get the status.

  // TODO(hexagon): prefer having a fast-path for if the semaphore is
  // known-signaled in user-mode. This can usually avoid syscalls/ioctls and
  // potential context switches in polling cases.

  // TODO(hexagon): check for `iree_timeout_is_immediate(timeout)` and return
  // immediately if the condition is not satisfied before waiting with
  // `iree_status_from_code(IREE_STATUS_DEADLINE_EXCEEDED)`. Prefer the raw code
  // status instead of a full status object as immediate timeouts are used when
  // polling and a full status may capture a backtrace and allocate.

  iree_slim_mutex_lock(&semaphore->mutex);

  if (semaphore->current_value == IREE_HAL_SEMAPHORE_FAILURE_VALUE ||
      !iree_status_is_ok(semaphore->failure_status)) {
    // Fastest path: failed; return an error to tell callers to query for it.
    iree_slim_mutex_unlock(&semaphore->mutex);
    return iree_status_from_code(IREE_STATUS_ABORTED);
  } else if (semaphore->current_value >= value) {
    // Fast path: already satisfied.
    iree_slim_mutex_unlock(&semaphore->mutex);
    return iree_ok_status();
  } else if (iree_timeout_is_immediate(timeout)) {
    // Not satisfied but a poll, so can avoid the expensive wait handle work.
    iree_slim_mutex_unlock(&semaphore->mutex);
    return iree_status_from_code(IREE_STATUS_DEADLINE_EXCEEDED);
  }

  iree_time_t deadline_ns = iree_timeout_as_deadline_ns(timeout);

  // Slow path: acquire a timepoint while we hold the lock.
  iree_hal_hexagon_timepoint_t timepoint;
  iree_status_t status = iree_hal_hexagon_semaphore_acquire_timepoint(
      semaphore, value, timeout, &timepoint);

  iree_slim_mutex_unlock(&semaphore->mutex);
  if (IREE_UNLIKELY(!iree_status_is_ok(status)))
    return status;

  // Wait until the timepoint resolves.
  // If satisfied the timepoint is automatically cleaned up and we are done. If
  // the deadline is reached before satisfied then we have to clean it up.
  status = iree_wait_one(&timepoint.event, deadline_ns);
  if (!iree_status_is_ok(status)) {
    iree_hal_semaphore_cancel_timepoint(&semaphore->base, &timepoint.base);
  }
  iree_event_pool_release(semaphore->event_pool, 1, &timepoint.event);

  // Recheck conditions.
  if (iree_status_is_ok(status)) {
    iree_slim_mutex_lock(&semaphore->mutex);
    if (semaphore->current_value == IREE_HAL_SEMAPHORE_FAILURE_VALUE ||
        !iree_status_is_ok(semaphore->failure_status)) {
      status = iree_status_from_code(IREE_STATUS_ABORTED);
    } else if (semaphore->current_value >= value) {
      status = iree_ok_status();
    } else if (iree_timeout_is_immediate(timeout)) {
      status = iree_status_from_code(IREE_STATUS_DEADLINE_EXCEEDED);
    }
    iree_slim_mutex_unlock(&semaphore->mutex);
  }

  return status;
}

static iree_status_t iree_hal_hexagon_semaphore_import_timepoint(
    iree_hal_semaphore_t *base_semaphore, uint64_t value,
    iree_hal_queue_affinity_t queue_affinity,
    iree_hal_external_timepoint_t external_timepoint) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "timepoint import is not yet implemented");
}

static iree_status_t iree_hal_hexagon_semaphore_export_timepoint(
    iree_hal_semaphore_t *base_semaphore, uint64_t value,
    iree_hal_queue_affinity_t queue_affinity,
    iree_hal_external_timepoint_type_t requested_type,
    iree_hal_external_timepoint_flags_t requested_flags,
    iree_hal_external_timepoint_t *IREE_RESTRICT out_external_timepoint) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "timepoint export is not yet implemented");
}

static const iree_hal_semaphore_vtable_t iree_hal_hexagon_semaphore_vtable = {
    .destroy = iree_hal_hexagon_semaphore_destroy,
    .query = iree_hal_hexagon_semaphore_query,
    .signal = iree_hal_hexagon_semaphore_signal,
    .fail = iree_hal_hexagon_semaphore_fail,
    .wait = iree_hal_hexagon_semaphore_wait,
    .import_timepoint = iree_hal_hexagon_semaphore_import_timepoint,
    .export_timepoint = iree_hal_hexagon_semaphore_export_timepoint,
};

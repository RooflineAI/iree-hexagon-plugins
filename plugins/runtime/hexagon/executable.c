// Copyright 2025 RooflineAI GmbH

#include "hexagon/executable.h"

#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "AEEStdErr.h"
#include "hexagon/rpc_types_impl.h"
#include "hexagon/utils.h"
#include "hexagon_dsp.h"
#include "iree/base/api.h"
#include "iree/hal/api.h"

//===----------------------------------------------------------------------===//
// iree_hal_hexagon_executable_t
//===----------------------------------------------------------------------===//

typedef struct iree_hal_hexagon_executable_t {
  iree_hal_resource_t resource;
  iree_allocator_t host_allocator;
  const iree_hal_hexagon_rpc_session_t *rpc_session; // not owned, owner: device
  const char *so_file_path;
  rpc_executable_handle_t executable_handle;
} iree_hal_hexagon_executable_t;

static const iree_hal_executable_vtable_t iree_hal_hexagon_executable_vtable;

static iree_hal_hexagon_executable_t *
iree_hal_hexagon_executable_cast(iree_hal_executable_t *base_value) {
  IREE_HAL_ASSERT_TYPE(base_value, &iree_hal_hexagon_executable_vtable);
  return (iree_hal_hexagon_executable_t *)base_value;
}

static iree_status_t iree_hal_hexagon_executable_set_up(
    const iree_hal_executable_params_t *executable_params,
    const iree_hal_hexagon_rpc_session_t *rpc_session,
    iree_hal_hexagon_executable_t *executable) {
  // To load the instructions to DSP memory and have it executable, it needs
  // to be written to an .so file which needs to be dlopen-ed on the DSP.

  // Create .so file in temporary directory.
  char *tmpdir = getenv("TMPDIR");
  if (!tmpdir) {
    tmpdir = "/tmp";
  }
  // Template for mkstemps() below. The XXXXXX part will be randomized, so we
  // need a writable buffer here. The suffix length (how many characters follow
  // the XXXXXX part) is needed by the function call, so keep this in sync.
  static const char *so_file_slash_name = "/iree-run-module-hexagon-XXXXXX.so";
  static const int so_file_suffix_length = 3;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(executable->host_allocator,
                            strlen(tmpdir) + strlen(so_file_slash_name) + 1,
                            (void **)&executable->so_file_path));
  strcpy(executable->so_file_path, tmpdir);
  strcat(executable->so_file_path, so_file_slash_name);
  int so_fd = mkstemps(executable->so_file_path, so_file_suffix_length);
  if (so_fd == -1) {
    return IREE_HAL_HEXAGON_MAKE_STATUS_FROM_ERRNO("could not open .so file");
  }
  ssize_t so_wr_ret = write(so_fd, executable_params->executable_data.data,
                            executable_params->executable_data.data_length);
  close(so_fd);
  if (so_wr_ret != executable_params->executable_data.data_length) {
    return IREE_HAL_HEXAGON_MAKE_STATUS_FROM_ERRNO("could not write .so file");
  }

  // Load executable on DSP side.
  int dsp_err = hexagon_dsp_executable_load(executable->rpc_session->rpc_handle,
                                            executable->so_file_path,
                                            &executable->executable_handle);
  if (dsp_err != AEE_SUCCESS) {
    return IREE_HAL_HEXAGON_MAKE_STATUS_FROM_DSP_ERR(
        dsp_err, "could not load .so file on DSP");
  }

  return iree_ok_status();
}

iree_status_t iree_hal_hexagon_executable_create(
    const iree_hal_executable_params_t *executable_params,
    iree_allocator_t host_allocator,
    const iree_hal_hexagon_rpc_session_t *rpc_session,
    iree_hal_executable_t **out_executable) {
  IREE_ASSERT_ARGUMENT(executable_params);
  IREE_ASSERT_ARGUMENT(out_executable);
  IREE_TRACE_ZONE_BEGIN(z0);
  *out_executable = NULL;

  // Allocate storage for the executable and its associated data structures.
  iree_hal_hexagon_executable_t *executable = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_allocator_malloc(host_allocator, sizeof(*executable),
                                (void **)&executable));
  iree_hal_resource_initialize(&iree_hal_hexagon_executable_vtable,
                               &executable->resource);
  executable->host_allocator = host_allocator;
  executable->rpc_session = rpc_session;

  // TODO(hexagon): load executable module(s). Note that the input data should
  // be treated as untrusted and should be verified to the best ability the
  // format provides. A target that cannot provide verification will be treated
  // as unsafe. For JIT-style implementations as much work as possible should be
  // done here so that errors can be propagated back to users - do not defer
  // preparation.
  //
  // In general the executable should only retain information required to
  // service the command buffer implementation that will be dispatching entry
  // points within it. Optionally information can be retained for tracing and
  // debugging.
  //
  // Implementations with flexible formats (ELF, etc) can directly use those for
  // metadata as well with custom sections. If an implementation does not have a
  // flexible format or support linking and requires several modules a wrapper
  // can be used instead. In upstream IREE HALs Flatbuffers is used and is the
  // preferred format (zero-copy, mmappable, verifiable, near header-only dep
  // with no binary size or runtime overheads, etc) and is the easiest to use,
  // but you do you.

  iree_status_t status = iree_hal_hexagon_executable_set_up(
      executable_params, rpc_session, executable);

  if (iree_status_is_ok(status)) {
    *out_executable = (iree_hal_executable_t *)executable;
  } else {
    iree_hal_executable_destroy((iree_hal_executable_t *)executable);
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

static void
iree_hal_hexagon_executable_destroy(iree_hal_executable_t *base_executable) {
  iree_hal_hexagon_executable_t *executable =
      iree_hal_hexagon_executable_cast(base_executable);
  iree_allocator_t host_allocator = executable->host_allocator;
  IREE_TRACE_ZONE_BEGIN(z0);

  // TODO(hexagon): release any implementation resources.
  if (executable->so_file_path) {
    unlink(executable->so_file_path);
    free(executable->so_file_path);
  }

  iree_allocator_free(host_allocator, executable);

  IREE_TRACE_ZONE_END(z0);
}

static const iree_hal_executable_vtable_t iree_hal_hexagon_executable_vtable = {
    .destroy = iree_hal_hexagon_executable_destroy,
};

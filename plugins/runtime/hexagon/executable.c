// Copyright 2025 RooflineAI GmbH
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "hexagon/executable.h"

#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "AEEStdErr.h"
#include "hexagon/schemas/hexagon_executable_def_reader.h"
#include "hexagon/schemas/hexagon_executable_def_verifier.h"
#include "hexagon/serialize/rpc_types.h"
#include "hexagon/utils.h"
#include "hexagon_dsp.h"
#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/hal/utils/executable_header.h"

//===----------------------------------------------------------------------===//
// iree_hal_hexagon_executable_t
//===----------------------------------------------------------------------===//

typedef struct iree_hal_hexagon_executable_t {
  iree_hal_resource_t resource;
  iree_allocator_t host_allocator;
  rpc_session_handle_t rpc_session_handle; // not owned, owner: device
  char *so_file_path;
  rpc_executable_handle_t rpc_executable_handle;
  // Export names in dense ordinal order (index == the ordinal the DSP
  // dispatches by via exports->ptrs[ordinal]), copied from the executable-def
  // flatbuffer so lookup_function_by_name can resolve names to ordinals.
  iree_host_size_t export_count;
  char **export_names;
} iree_hal_hexagon_executable_t;

static const iree_hal_executable_vtable_t iree_hal_hexagon_executable_vtable;

static iree_hal_hexagon_executable_t *
iree_hal_hexagon_executable_cast(iree_hal_executable_t *base_value) {
  IREE_HAL_ASSERT_TYPE(base_value, &iree_hal_hexagon_executable_vtable);
  return (iree_hal_hexagon_executable_t *)base_value;
}

static iree_status_t iree_hal_hexagon_executable_set_up(
    const iree_hal_executable_params_t *executable_params,
    rpc_session_handle_t rpc_session_handle,
    iree_hal_hexagon_executable_t *executable) {
  // To load the instructions to DSP memory and have it executable, it needs
  // to be written to an .so file which needs to be dlopen-ed on the DSP.

  // The executable data is the Hexagon executable-def flatbuffer (built by the
  // compiler in HexagonExecutableSerialization.cpp): host-readable export names
  // in dense ordinal order + the linked ELF. Parse the names so
  // lookup_function_by_name can resolve the by-name dispatch references
  // introduced by IREE #24036, then extract the ELF to load on the DSP.
  iree_const_byte_span_t flatbuffer_data = iree_const_byte_span_empty();
  IREE_RETURN_IF_ERROR(iree_hal_read_executable_flatbuffer_header(
      executable_params->executable_data, /*unsafe_infer_size=*/false,
      iree_hal_hexagon_ExecutableDef_file_identifier, &flatbuffer_data));
  int verify_ret = iree_hal_hexagon_ExecutableDef_verify_as_root(
      flatbuffer_data.data, flatbuffer_data.data_length);
  if (verify_ret != flatcc_verify_ok) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "hexagon executable flatbuffer verification failed: %s",
        flatcc_verify_error_string(verify_ret));
  }
  iree_hal_hexagon_ExecutableDef_table_t executable_def =
      iree_hal_hexagon_ExecutableDef_as_root(flatbuffer_data.data);

  // Copy the export names (dense ordinal order) for by-name lookup.
  flatbuffers_string_vec_t entry_points =
      iree_hal_hexagon_ExecutableDef_entry_points_get(executable_def);
  iree_host_size_t export_count = flatbuffers_string_vec_len(entry_points);
  if (export_count > 0) {
    IREE_RETURN_IF_ERROR(iree_allocator_malloc(
        executable->host_allocator, export_count * sizeof(char *),
        (void **)&executable->export_names));
    executable->export_count = export_count;
    for (iree_host_size_t i = 0; i < export_count; ++i) {
      flatbuffers_string_t name = flatbuffers_string_vec_at(entry_points, i);
      size_t name_len = flatbuffers_string_len(name);
      char *name_copy = NULL;
      IREE_RETURN_IF_ERROR(iree_allocator_malloc(
          executable->host_allocator, name_len + 1, (void **)&name_copy));
      memcpy(name_copy, name, name_len);
      name_copy[name_len] = '\0';
      executable->export_names[i] = name_copy;
    }
  }

  // The linked ELF the DSP dlopen-s.
  flatbuffers_uint8_vec_t elf =
      iree_hal_hexagon_ExecutableDef_elf_get(executable_def);
  // A flatcc scalar-vector handle points at element 0, so it is the data ptr.
  const void *elf_data = (const void *)elf;
  iree_host_size_t elf_length = flatbuffers_uint8_vec_len(elf);

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
  ssize_t so_wr_ret = write(so_fd, elf_data, elf_length);
  close(so_fd);
  if (so_wr_ret != (ssize_t)elf_length) {
    return IREE_HAL_HEXAGON_MAKE_STATUS_FROM_ERRNO("could not write .so file");
  }

  // Load executable on DSP side.
  int dsp_err = hexagon_dsp_executable_load(executable->rpc_session_handle,
                                            executable->so_file_path,
                                            &executable->rpc_executable_handle);
  if (dsp_err != AEE_SUCCESS) {
    return IREE_HAL_HEXAGON_MAKE_STATUS_FROM_DSP_ERR(
        dsp_err, "could not load .so file on DSP");
  }

  return iree_ok_status();
}

iree_status_t iree_hal_hexagon_executable_create(
    const iree_hal_executable_params_t *executable_params,
    iree_allocator_t host_allocator, rpc_session_handle_t rpc_session_handle,
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
  executable->rpc_session_handle = rpc_session_handle;

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
      executable_params, rpc_session_handle, executable);

  if (iree_status_is_ok(status)) {
    *out_executable = (iree_hal_executable_t *)executable;
  } else {
    iree_hal_executable_destroy((iree_hal_executable_t *)executable);
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

rpc_executable_handle_t iree_hal_hexagon_executable_get_rpc_executable(
    iree_hal_executable_t *base_executable) {
  iree_hal_hexagon_executable_t *executable =
      iree_hal_hexagon_executable_cast(base_executable);
  if (!iree_hal_resource_is(&executable->resource,
                            &iree_hal_hexagon_executable_vtable)) {
    return 0;
  }
  return executable->rpc_executable_handle;
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
    iree_allocator_free(host_allocator, executable->so_file_path);
  }
  if (executable->export_names) {
    for (iree_host_size_t i = 0; i < executable->export_count; ++i) {
      iree_allocator_free(host_allocator, executable->export_names[i]);
    }
    iree_allocator_free(host_allocator, executable->export_names);
  }

  iree_allocator_free(host_allocator, executable);

  IREE_TRACE_ZONE_END(z0);
}

// The IREE HAL VM module resolves executable functions by name at module init
// (iree_hal_module_executable_lookup_function -> lookup_function_by_name) and
// encodes the resolved index as the dispatch's export ordinal. A hexagon
// executable may be a LINKED, multi-export module (e.g. a whole model linked
// into one executable), so we resolve each name to its dense ordinal via the
// export-name table parsed from the executable-def flatbuffer; the DSP then
// dispatches exports->ptrs[ordinal].
static iree_host_size_t
iree_hal_hexagon_executable_function_count(iree_hal_executable_t *base) {
  iree_hal_hexagon_executable_t *executable =
      iree_hal_hexagon_executable_cast(base);
  return executable->export_count;
}

static iree_status_t iree_hal_hexagon_executable_lookup_function_by_name(
    iree_hal_executable_t *base, iree_string_view_t name,
    iree_hal_executable_function_t *out_function) {
  iree_hal_hexagon_executable_t *executable =
      iree_hal_hexagon_executable_cast(base);
  for (iree_host_size_t i = 0; i < executable->export_count; ++i) {
    if (iree_string_view_equal(
            name, iree_make_cstring_view(executable->export_names[i]))) {
      *out_function = iree_hal_executable_function_from_index((uint32_t)i);
      return iree_ok_status();
    }
  }
  return iree_make_status(IREE_STATUS_NOT_FOUND,
                          "no hexagon export named '%.*s'", (int)name.size,
                          name.data);
}

// Reports the reflection info for an export. The executable-def flatbuffer only
// carries export names (there is no host-readable reflection metadata for the
// foreign DSP arch), so we return the name and leave the remaining fields
// zeroed.
static iree_status_t iree_hal_hexagon_executable_function_info(
    iree_hal_executable_t *base, iree_hal_executable_function_t function,
    iree_hal_executable_function_info_t *out_info) {
  iree_hal_hexagon_executable_t *executable =
      iree_hal_hexagon_executable_cast(base);
  uint32_t ordinal = iree_hal_executable_function_index(function);
  if (ordinal >= executable->export_count) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "export ordinal %u out of range", ordinal);
  }
  memset(out_info, 0, sizeof(*out_info));
  out_info->name = iree_make_cstring_view(executable->export_names[ordinal]);
  return iree_ok_status();
}

// Hexagon executables carry no per-parameter reflection, so
// function_info::parameter_count is always 0 and there is nothing to write.
static iree_status_t iree_hal_hexagon_executable_function_parameters(
    iree_hal_executable_t *base, iree_hal_executable_function_t function,
    iree_host_size_t capacity,
    iree_hal_executable_function_parameter_t *out_parameters) {
  iree_hal_hexagon_executable_t *executable =
      iree_hal_hexagon_executable_cast(base);
  uint32_t ordinal = iree_hal_executable_function_index(function);
  if (ordinal >= executable->export_count) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "export ordinal %u out of range", ordinal);
  }
  (void)capacity;
  (void)out_parameters;
  return iree_ok_status();
}

// Hexagon executables do not expose global buffers.
static iree_status_t iree_hal_hexagon_executable_lookup_global_by_name(
    iree_hal_executable_t *base, iree_string_view_t name,
    iree_hal_queue_affinity_t queue_affinity, iree_hal_buffer_t **out_buffer) {
  (void)base;
  (void)name;
  (void)queue_affinity;
  *out_buffer = NULL;
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "hexagon executable global lookup not implemented");
}

static const iree_hal_executable_vtable_t iree_hal_hexagon_executable_vtable = {
    .destroy = iree_hal_hexagon_executable_destroy,
    .function_count = iree_hal_hexagon_executable_function_count,
    .function_info = iree_hal_hexagon_executable_function_info,
    .function_parameters = iree_hal_hexagon_executable_function_parameters,
    .lookup_function_by_name =
        iree_hal_hexagon_executable_lookup_function_by_name,
    .lookup_global_by_name = iree_hal_hexagon_executable_lookup_global_by_name,
};

// Copyright 2020 The IREE Authors
// Copyright 2026 RooflineAI GmbH
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Derived from:
//   runtime/src/iree/hal/local/executable_library.h
// IREE revision: a45adeaa6115e446c898e6eb21fb6edc0e65ddc4
//
// The Hexagon executable library ABI. The compiler (Target/LibraryBuilder and
// CodeGen/Conversion/HexagonDispatchABI) produces these structures and the DSP
// runtime (dsp/executable.c, dsp/command_buffer.c) consumes them.
// Formatted like upstream, not like this repository, so the copy can be diffed
// against it.
// clang-format off

#ifndef HEXAGON_DSP_RT_EXECUTABLE_LIBRARY_H
#define HEXAGON_DSP_RT_EXECUTABLE_LIBRARY_H

// NOTE: this file is designed to be a standalone header: it is embedded in the
// compiler and must not take any dependencies on the runtime HAL code.
// Changes here will require changes to the compiler and must be versioned as if
// this was a schema: backwards-incompatible changes require version bumps or
// the ability to feature-detect at runtime.

#include <assert.h>
#include <stddef.h>
#include <stdint.h>

// The Hexagon C toolchain's assert.h does not provide the C11 `static_assert`
// macro.
#if !defined(__cplusplus) && !defined(static_assert)
#define static_assert _Static_assert
#define HEXAGON_DSP_RT_EXECUTABLE_LIBRARY_DEFINED_STATIC_ASSERT
#endif

//===----------------------------------------------------------------------===//
// Versioning and interface querying
//===----------------------------------------------------------------------===//

typedef struct hexagon_executable_environment_t
    hexagon_executable_environment_t;

// Version code of the library structures. The runtime only loads libraries
// built with exactly HEXAGON_EXECUTABLE_LIBRARY_VERSION_LATEST.
typedef uint32_t hexagon_executable_library_version_t;

// Hexagon versions carry 'HX' in the upper half so they can not be confused
// with an upstream IREE version (0x0000000N).
#define HEXAGON_EXECUTABLE_LIBRARY_VERSION_1 0x48580001u

#define HEXAGON_EXECUTABLE_LIBRARY_VERSION_LATEST                             \
  HEXAGON_EXECUTABLE_LIBRARY_VERSION_1

// A header present at the top of all versions of the library API used by the
// runtime to ensure version compatibility.
typedef struct hexagon_executable_library_header_t {
  // Version of the API this library was built with.
  hexagon_executable_library_version_t version;
} hexagon_executable_library_header_t;

// Exported function from dynamic libraries for querying library information.
//
// Returns NULL if |max_version| differs from the version the library was built
// with. |environment| is not read by the query function.
typedef const hexagon_executable_library_header_t** (
    *hexagon_executable_library_query_fn_t)(
    hexagon_executable_library_version_t max_version,
    const hexagon_executable_environment_t* environment);

// Function name exported from dynamic libraries (pass to dlsym).
#define HEXAGON_EXECUTABLE_LIBRARY_EXPORT_NAME \
  "hexagon_executable_library_query"

//===----------------------------------------------------------------------===//
// HEXAGON_EXECUTABLE_LIBRARY_VERSION_*
//===----------------------------------------------------------------------===//

// Declares imports requested by the executable library.
//
// FIXME: dynamic HAL imports are not supported.To be removed
typedef struct hexagon_executable_import_table_t {
  // Total number of imports in the table.
  uint32_t count;
  // Import symbol names; a `?` prefix marks an optional import.
  const char* const* symbols;
} hexagon_executable_import_table_t;

// Defines the environment in which the executable is being used. The same
// environment is passed to all dispatches of an executable.
typedef struct hexagon_executable_environment_t {
  // Executable-level constants, read by `hal.executable.constant.load`.
  //
  // FIXME: not populated. The runtime leaves this NULL and LibraryBuilder
  // always declares 0 executable constants, so a dispatch that loads one
  // dereferences NULL.
  const uint32_t* constants;
} hexagon_executable_environment_t;

// Read-only per-dispatch state passed to each workgroup in a dispatch.
// The DSP runtime passes this as the first member of
// hexagon_dsp_extended_dispatch_state_t (dispatch_state.h).
typedef struct hexagon_executable_dispatch_state_t {
  // Workgroup size chosen for the dispatch.
  uint32_t workgroup_size_x;
  uint32_t workgroup_size_y;
  uint16_t workgroup_size_z;

  // Total workgroup count for the dispatch.
  uint32_t workgroup_count_x;
  uint32_t workgroup_count_y;
  uint16_t workgroup_count_z;

  // 4 byte push constant values.
  const uint32_t* constants;
  // Base pointers to each binding buffer.
  void* const* binding_ptrs;
  // The length of each binding in bytes, 1:1 with |binding_ptrs|.
  const size_t* binding_lengths;
} hexagon_executable_dispatch_state_t;
static_assert(sizeof(hexagon_executable_dispatch_state_t) <= 64,
              "try keeping dispatch state small enough to fit in a cache line");

// Read-only per-workgroup state passed to each workgroup in a dispatch.
typedef struct hexagon_executable_workgroup_state_t {
  // Workgroup ID of the currently executing workgroup.
  // This is in the range of 0-workgroup_count and each unique workgroup is to
  // perform workgroup_size invocations.
  uint32_t workgroup_id_x;
  uint32_t workgroup_id_y;
  uint16_t workgroup_id_z;

  // Reserved for future use.
  uint16_t reserved;

  // Logical processor identifier used to index into processor info fields.
  // Depending on the implementation this may be an ordinal, a bitfield, or an
  // opaque unique identifier.
  //
  // NOTE: we could steal bits from the |processor_id| if needed; today the ID
  // is the global ID but it really only needs to be within the current node
  // (8-bits, or 16-bit for single-node thousand-core future proofing).
  uint32_t processor_id;
} hexagon_executable_workgroup_state_t;
static_assert(
    sizeof(hexagon_executable_workgroup_state_t) <= 64,
    "try keeping workgroup state small enough to fit in a cache line");

// Function signature of exported executable dispatch functions.
// The same |environment| is passed to all dispatches.
// The same |dispatch_state| is passed to all workgroups within a dispatch.
// A unique |workgroup_state| is passed to every workgroup within a dispatch.
//
// Returns 0 on success and non-zero on failure. Failures will cause device loss
// and should only be used to communicate serious issues that should abort all
// execution within the current device. Buffer overflows are a good example of
// a useful failure though the HAL does not mandate that all overflows are
// caught and only that they are not harmful - clamping byte ranges and never
// returning a failure is sufficient.
typedef int (*hexagon_executable_dispatch_fn_t)(
    const hexagon_executable_environment_t* environment,
    const hexagon_executable_dispatch_state_t* dispatch_state,
    const hexagon_executable_workgroup_state_t* workgroup_state);

// Resource flags stored in hexagon_executable_dispatch_attrs_t.
enum hexagon_dispatch_flag_bits_e {
  HEXAGON_DISPATCH_FLAG_NONE = 0ull,
  // The compiler recorded the resources below and `vtcm_bytes`. If clear the
  // dispatch must be treated as using HVX, HMX and all of VTCM.
  HEXAGON_DISPATCH_FLAG_RESOURCES_VALID = 1ull << 0,
  // The dispatch uses HVX vector units.
  HEXAGON_DISPATCH_FLAG_USES_HVX = 1ull << 1,
  // The dispatch uses the HMX matrix unit.
  HEXAGON_DISPATCH_FLAG_USES_HMX = 1ull << 2,
};
typedef uint64_t hexagon_dispatch_flags_t;

// Attributes of an exported dispatch function: the resources it needs,
// measured by the compiler's HexagonAnnotateDispatchResourcesPass.
typedef struct hexagon_executable_dispatch_attrs_t {
  // Bitwise OR of hexagon_dispatch_flag_bits_e.
  hexagon_dispatch_flags_t hexagon_flags;
  // Bytes of VTCM the dispatch allocates at peak, as bounded by the compiler.
  // INT64_MAX if the compiler could not bound it: the dispatch needs all of
  // VTCM. Only meaningful if `hexagon_flags` has
  // HEXAGON_DISPATCH_FLAG_RESOURCES_VALID set.
  uint64_t vtcm_bytes;
} hexagon_executable_dispatch_attrs_t;
static_assert(sizeof(hexagon_executable_dispatch_attrs_t) <= 64,
              "try keeping dispatch attrs small enough to fit in a cache line");
// The compiler's LibraryBuilder emits this layout as an LLVM struct
// (makeDispatchAttrsType); keep both in sync.
static_assert(sizeof(hexagon_executable_dispatch_attrs_t) == 16,
              "Hexagon dispatch attrs layout changed");
static_assert(offsetof(hexagon_executable_dispatch_attrs_t,
                       hexagon_flags) == 0,
              "Hexagon dispatch attrs layout changed");
static_assert(offsetof(hexagon_executable_dispatch_attrs_t, vtcm_bytes) ==
                  8,
              "Hexagon dispatch attrs layout changed");


// A table of exported functions arranged as a struct-of-arrays for more
// efficient packing and faster lookup. Each subarray - when not omitted and
// NULL - is indexed by export ordinal and has up to |count| entries.
typedef struct hexagon_executable_export_table_t {
  // Total number of exports in the table.
  uint32_t count;

  // Function pointers for each exported dispatch function.
  const hexagon_executable_dispatch_fn_t* ptrs;

  // Table of attributes 1:1 with ptrs.
  const hexagon_executable_dispatch_attrs_t* attrs;
  const char* const* names;
} hexagon_executable_export_table_t;

// Top-level structure of an executable library.
// The entire structure is designed to be read-only and able to live embedded in
// the binary .rdata section.
typedef struct hexagon_executable_library_t {
  // Version/metadata header.
  // Will have a version of HEXAGON_EXECUTABLE_LIBRARY_VERSION_*.
  const hexagon_executable_library_header_t* header;

  // Table of imported functions available to functions in the executable.
  hexagon_executable_import_table_t imports;

  // Table of exported functions from the executable.
  hexagon_executable_export_table_t exports;
} hexagon_executable_library_t;

#ifdef HEXAGON_DSP_RT_EXECUTABLE_LIBRARY_DEFINED_STATIC_ASSERT
#undef static_assert
#undef HEXAGON_DSP_RT_EXECUTABLE_LIBRARY_DEFINED_STATIC_ASSERT
#endif

#endif  // HEXAGON_DSP_RT_EXECUTABLE_LIBRARY_H
// clang-format on

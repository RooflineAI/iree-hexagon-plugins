// Copyright 2025 RooflineAI GmbH

// This file provides typedefs for some aliases for DSP RPC types based on only
// built-in C types. This is done to decouple some parts of the Hexagon driver
// implementation from the actual DSP headers, e.g., for unit tests.
// The type aliases defined in here must be based on the same underlying type
// as the actual DSP RPC types defined in the generated hexagon_dsp.h, in order
// to make the types work the Hexagon SDK. This is checked in rcp_types.c.
// This file can be compiled on any architecture and thus enable the compilation
// of unit tests with the types defined in this file.

#ifndef IREE_HAL_DRIVERS_HEXAGON_SERIALIZE_RPC_TYPES_H_
#define IREE_HAL_DRIVERS_HEXAGON_SERIALIZE_RPC_TYPES_H_

/// RPC session handle
typedef unsigned long rpc_session_handle_t;

/// RPC executable handle
typedef long long rpc_executable_handle_t;

/// RPC command buffer handle
typedef long long rpc_command_buffer_handle_t;

/// DSP virtual memory address
typedef long long rpc_dsp_vaddr_t;

#endif // IREE_HAL_DRIVERS_HEXAGON_SERIALIZE_RPC_TYPES_H_

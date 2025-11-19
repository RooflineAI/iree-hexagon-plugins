// Copyright 2025 RooflineAI GmbH

// This file provides some typedefs for some DSP RPC types defined in the
// generated hexagon_dsp.h in order to have more clear type names.

#ifndef IREE_HAL_DRIVERS_HEXAGON_RPC_TYPES_H_
#define IREE_HAL_DRIVERS_HEXAGON_RPC_TYPES_H_

#include "hexagon_dsp.h"

/// RPC session handle
typedef remote_handle64 rpc_session_handle_t;

/// RPC executable handle
typedef int64 rpc_executable_handle_t;

#endif // IREE_HAL_DRIVERS_HEXAGON_RPC_TYPES_H_

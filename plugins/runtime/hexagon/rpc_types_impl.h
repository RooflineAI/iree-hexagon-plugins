// Copyright 2025 RooflineAI GmbH

// This file provides the actual struct definitions for the wrapper types
// defined in rpc_types.h. For the shielding of the generated hexagon_dsp.h
// to work, NEVER include this file in any header file. Include rpc_types.h
// instead. If this does not work with an API (e.g. due to passing the structs
// defined in here by value), reconsider the API and pass pointers instead.

#ifndef IREE_HAL_DRIVERS_HEXAGON_RPC_TYPES_IMPL_H_
#define IREE_HAL_DRIVERS_HEXAGON_RPC_TYPES_IMPL_H_

#include "hexagon/rpc_types.h"

#include "hexagon_dsp.h"

struct iree_hal_hexagon_rpc_session_s {
  /// 1 if DSP RPC session is not open, 0 if not open, will only be 0 on failed
  /// open during immediate cleanup
  char rpc_is_open;
  /// handle of DSP RPC session, can be any value
  remote_handle64 rpc_handle;
};

typedef int64 rpc_executable_handle_t;

#endif // IREE_HAL_DRIVERS_HEXAGON_RPC_TYPES_IMPL_H_

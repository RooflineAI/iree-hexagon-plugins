// Copyright 2025 RooflineAI GmbH

// This file provides some wrappers for some DSP RPC types defined in the
// generated hexagon_dsp.h in order to shield the types from the IREE HAL
// driver headers, i.e., avoiding the need to include hexagon_dsp.h in those
// headers.

#ifndef IREE_HAL_DRIVERS_HEXAGON_RPC_TYPES_H_
#define IREE_HAL_DRIVERS_HEXAGON_RPC_TYPES_H_

/// wrapper for DSP RPC session handle
typedef struct iree_hal_hexagon_rpc_session_s iree_hal_hexagon_rpc_session_t;

#endif // IREE_HAL_DRIVERS_HEXAGON_RPC_TYPES_H_

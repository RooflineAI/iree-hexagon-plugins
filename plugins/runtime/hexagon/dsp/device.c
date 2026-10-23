// Copyright 2025 RooflineAI GmbH
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <assert.h>
#include <stdlib.h>

#include "AEEStdErr.h"
#include "HAP_farf.h"
#include "hexagon/dsp/device.h"
#include "hexagon/dsp/power_mode.h"
#include "hexagon/dsp/rt/async_runtime.h"
#include "hexagon_dsp.h"
#include "qurt.h"

/// Private data of Hexagon DSP RPC session.
typedef struct hexagon_dsp_private_s {
  void *power_context;
  /// max. number of dispatches executing at the same time, >= 1
  uint32_t max_concurrent_dispatches;
  /// number of HVX units reserved by qurt_hvx_reserve, >= 1
  int hvx_reserved_units;
} hexagon_dsp_private_t;

static inline hexagon_dsp_private_t *
hexagon_dsp_get_priv(remote_handle64 rpc_handle) {
  return (hexagon_dsp_private_t *)rpc_handle;
}

/**
 * @brief Open DSP RPC session.
 * @param[in] uri URI of the DSP RPC session,
 *                contains name of shared lib and name of DSP device
 * @param[out] rpc_handle pass to close and other functions
 * @return AEE_SUCCESS for success
 */
int hexagon_dsp_open(const char *uri, remote_handle64 *rpc_handle) {
  // allocate data structure for managing the PRC session
  hexagon_dsp_private_t *priv =
      (hexagon_dsp_private_t *)calloc(1, sizeof(hexagon_dsp_private_t));
  if (!priv) {
    return AEE_ENOMEMORY;
  }

  int rc = hexagon_dsp_power_state_apply_max_performance(&priv->power_context);
  if (rc != AEE_SUCCESS) {
    free(priv);
    return rc;
  }

  // sequential until the host sets something else
  priv->max_concurrent_dispatches = 1;

  // reserve the HVX units for this process
  int reserved = qurt_hvx_reserve(QURT_HVX_RESERVE_ALL);
  if (reserved <= 0) {
    FARF(RUNTIME_HIGH, "HEXAGON-RUNTIME-WARNING: qurt_hvx_reserve failed (%d)",
         reserved);
    return AEE_EBADPARM;
  }
  priv->hvx_reserved_units = reserved;

  // return pointer to RPC session data structure as handle
  *rpc_handle = (remote_handle64)priv;
  return AEE_SUCCESS;
}

/**
 * @brief Close DSP RPC session.
 * @param[in] rpc_handle RPC handle, the value returned by hexagon_dsp_open
 * @return AEE_SUCCESS for success, should always succeed
 */
int hexagon_dsp_close(remote_handle64 rpc_handle) {
  hexagon_dsp_private_t *priv = hexagon_dsp_get_priv(rpc_handle);

  hexagon_dsp_power_state_cleanup(priv->power_context);

  int err = qurt_hvx_cancel_reserve();
  if (err != 0) {
    FARF(ERROR, "HEXAGON-RUNTIME-ERROR: qurt_hvx_cancel_reserve failed (%d)",
         err);
  }

  // free management data structure
  free(priv);
  return AEE_SUCCESS;
}

/**
 * @brief Set the max. number of dispatches executing at the same time.
 * @param[in] rpc_handle RPC handle, the value returned by hexagon_dsp_open
 * @param[in] max_concurrent_dispatches limit, 0 to use the number of HVX
 *            contexts (limited by the number of worker threads)
 * @return AEE_SUCCESS for success, AEE_EBADPARM if the limit exceeds the
 *         number of HVX contexts, reserved HVX units or worker threads
 */
int hexagon_dsp_set_max_concurrent_dispatches(remote_handle64 rpc_handle,
                                              int max_concurrent_dispatches) {
  hexagon_dsp_private_t *priv = hexagon_dsp_get_priv(rpc_handle);

  // bits 15..8 hold the number of 128 byte HVX contexts
  int64_t hvx_contexts = (qurt_hvx_get_units() >> 8) & 0xFF;
  if (max_concurrent_dispatches == 1) {
    // sequential, does not need any HVX context or worker beyond the master
    priv->max_concurrent_dispatches = 1;
    return AEE_SUCCESS;
  }
  int64_t num_workers = hexagon_rt_async_num_workers();
  int64_t limit = hvx_contexts < num_workers ? hvx_contexts : num_workers;
  assert(limit >= 1);
  if (priv->hvx_reserved_units < limit) {
    limit = priv->hvx_reserved_units;
  }
  if (max_concurrent_dispatches == 0) {
    max_concurrent_dispatches = (int)limit;
  }
  if (max_concurrent_dispatches < 0 || max_concurrent_dispatches > limit) {
    FARF(ERROR,
         "HEXAGON-RUNTIME-ERROR: max concurrent dispatches %d exceeds %lld "
         "(HVX contexts: %lld, reserved HVX units: %d, worker threads: %lld)",
         max_concurrent_dispatches, (long long)limit, (long long)hvx_contexts,
         priv->hvx_reserved_units, (long long)num_workers);
    return AEE_EBADPARM;
  }
  priv->max_concurrent_dispatches = (uint32_t)max_concurrent_dispatches;
  return AEE_SUCCESS;
}

uint32_t hexagon_dsp_get_max_concurrent_dispatches(remote_handle64 rpc_handle) {
  return hexagon_dsp_get_priv(rpc_handle)->max_concurrent_dispatches;
}

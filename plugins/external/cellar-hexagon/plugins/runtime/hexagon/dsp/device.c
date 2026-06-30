// Copyright 2025 RooflineAI GmbH

#include <stdlib.h>

#include "AEEStdErr.h"
#include "hexagon/dsp/power_mode.h"
#include "hexagon_dsp.h"

/// Private data of Hexagon DSP RPC session.
typedef struct hexagon_dsp_private_s {
  void *power_context;
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

  // free management data structure
  free(priv);
  return AEE_SUCCESS;
}

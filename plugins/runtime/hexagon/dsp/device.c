// Copyright 2025 RooflineAI GmbH

#include <stdlib.h>

#include "AEEStdErr.h"
#include "hexagon_dsp.h"

/// Private data of Hexagon DSP RPC session.
typedef struct hexagon_dsp_private_s {
  // Nothing required here for now.
  // Cannot be empty, because of undefined behavior of malloc(0).
  unsigned char dummy;
} hexagon_dsp_private_t;

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
  hexagon_dsp_private_t *priv = (hexagon_dsp_private_t *)rpc_handle;
  // free management data structure
  free(priv);
  return AEE_SUCCESS;
}

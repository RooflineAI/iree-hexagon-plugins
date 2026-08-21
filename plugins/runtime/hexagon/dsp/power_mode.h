// Copyright 2026 RooflineAI GmbH
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef HEXAGON_DSP_POWER_MODE_H
#define HEXAGON_DSP_POWER_MODE_H

/**
 * @brief Apply the session-scoped max-performance and HMX vote.
 * @param[out] out_power_context HAP power client context owned by the DSP RPC
 *                               session
 * @retval AEE_SUCCESS for success
 */
int hexagon_dsp_power_state_apply_max_performance(void **out_power_context);

/**
 * @brief Destroy HAP power client context.
 * @param[in] power_context HAP power client context owned by the DSP RPC
 * session
 * @retval AEE_SUCCESS for success
 */
int hexagon_dsp_power_state_cleanup(void *power_context);

/**
 * @brief Log current DSP power status for diagnostics.
 * @param[in] stage diagnostic stage name to include in the FARF log line
 */
void hexagon_dsp_power_log_status(const char *stage);

#endif // HEXAGON_DSP_POWER_MODE_H

// Copyright 2026 RooflineAI GmbH

#ifndef IREE_HAL_DRIVERS_HEXAGON_DSP_POWER_MODE_H_
#define IREE_HAL_DRIVERS_HEXAGON_DSP_POWER_MODE_H_

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

#endif // IREE_HAL_DRIVERS_HEXAGON_DSP_POWER_MODE_H_

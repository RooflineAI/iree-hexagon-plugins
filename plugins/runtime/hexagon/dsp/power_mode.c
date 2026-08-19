// Copyright 2026 RooflineAI GmbH
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "hexagon/dsp/power_mode.h"

#include <string.h>

#include "AEEStdDef.h"
#include "AEEStdErr.h"
#include "HAP_dcvs.h"
#include "HAP_farf.h"
#include "HAP_power.h"

static int hexagon_dsp_power_set_hvx(void *power_context, int power_up) {
  HAP_power_request_t request;
  memset(&request, 0, sizeof(request));
  request.type = HAP_power_set_HVX;
  request.hvx.power_up = power_up;
  return HAP_power_set(power_context, &request);
}

static int hexagon_dsp_power_set_hmx(void *power_context, int power_up) {
  HAP_power_request_t request;
  memset(&request, 0, sizeof(request));
  request.type = HAP_power_set_HMX;
  request.hmx.power_up = power_up;
  return HAP_power_set(power_context, &request);
}

static int hexagon_dsp_power_apply_fixed_max_vote(void *power_context) {
  int rc = AEE_SUCCESS;

  HAP_power_request_t request;
  memset(&request, 0, sizeof(request));
  request.type = HAP_power_set_DCVS_v3;
  request.dcvs_v3.set_dcvs_enable = TRUE;
  request.dcvs_v3.dcvs_enable = FALSE;
  request.dcvs_v3.set_latency = TRUE;
  request.dcvs_v3.latency = 10;
  request.dcvs_v3.set_sleep_disable = TRUE;
  request.dcvs_v3.sleep_disable = HAP_DCVS_LPM_LEVEL1;
  request.dcvs_v3.set_core_params = TRUE;
  request.dcvs_v3.core_params.min_corner = HAP_DCVS_VCORNER_MAX;
  request.dcvs_v3.core_params.target_corner = HAP_DCVS_VCORNER_MAX;
  request.dcvs_v3.core_params.max_corner = HAP_DCVS_VCORNER_MAX;
  request.dcvs_v3.set_bus_params = TRUE;
  request.dcvs_v3.bus_params.min_corner = HAP_DCVS_VCORNER_MAX;
  request.dcvs_v3.bus_params.target_corner = HAP_DCVS_VCORNER_MAX;
  request.dcvs_v3.bus_params.max_corner = HAP_DCVS_VCORNER_MAX;

  rc |= HAP_set_dcvs_v3_protected_bus_corners(&request, 1);
  rc |= HAP_set_ddr_perf_mode(&request, 1);
  rc |= HAP_set_dcvs_v3_core_perf_mode(&request, HAP_DCVS_CLK_PERF_HIGH);
  rc |= HAP_set_dcvs_v3_bus_perf_mode(&request, HAP_DCVS_CLK_PERF_HIGH);

  rc |= HAP_power_set(power_context, &request);

  return rc;
}

int hexagon_dsp_power_state_apply_max_performance(void **out_power_context) {
  int rc = AEE_SUCCESS;

  if (!out_power_context) {
    return AEE_EBADPARM;
  }
  *out_power_context = NULL;

  void *power_context = HAP_utils_create_context();
  if (!power_context) {
    FARF(RUNTIME_HIGH, "HEXAGON_RUNTIME_PM_APPLY,stage=create_context,rc=%d",
         AEE_ENOMEMORY);
    return AEE_ENOMEMORY;
  }

  rc |= hexagon_dsp_power_set_hvx(power_context, 1);
  if (rc != AEE_SUCCESS) {
    hexagon_dsp_power_state_cleanup(power_context);
    return rc;
  }

  rc |= hexagon_dsp_power_set_hmx(power_context, 1);
  if (rc != AEE_SUCCESS) {
    hexagon_dsp_power_state_cleanup(power_context);
    return rc;
  }

  rc |= hexagon_dsp_power_apply_fixed_max_vote(power_context);
  if (rc != AEE_SUCCESS) {
    hexagon_dsp_power_state_cleanup(power_context);
    return rc;
  }

  *out_power_context = power_context;

  return AEE_SUCCESS;
}

int hexagon_dsp_power_state_cleanup(void *power_context) {
  int rc = AEE_SUCCESS;

  if (power_context) {
    int destroy_rc = HAP_power_destroy(power_context);
    if (destroy_rc != AEE_SUCCESS && rc == AEE_SUCCESS) {
      rc = destroy_rc;
    }
    HAP_utils_destroy_context(power_context);
    power_context = NULL;
  }

  return rc;
}

// The HAP API relies on a union for the return value.
// This function accesses only the clkFreqHz field of the union, which is valid
// for all HAP_Power_response_type values that return a clock frequency.
static int
hexagon_dsp_power_query_HAP_power_type_clk_freq_Hz(HAP_Power_response_type type,
                                                   unsigned int *out_value) {
  HAP_power_response_t response;
  memset(&response, 0, sizeof(response));
  response.type = type;
  int rc = HAP_power_get(NULL, &response);
  if (rc == AEE_SUCCESS) {
    *out_value = response.clkFreqHz;
  }
  return rc;
}

void hexagon_dsp_power_log_status(const char *stage) {
  unsigned int clk_freq_hz = 0;
  unsigned int hmx_clk_freq_hz = 0;
  unsigned int dcvs_enabled = 0;

  int rc_clk = hexagon_dsp_power_query_HAP_power_type_clk_freq_Hz(
      HAP_power_get_clk_Freq, &clk_freq_hz);
  int rc_hmx = hexagon_dsp_power_query_HAP_power_type_clk_freq_Hz(
      HAP_power_get_hmx_core_clk_Freq, &hmx_clk_freq_hz);

  HAP_power_response_t dcvs_response;
  memset(&dcvs_response, 0, sizeof(dcvs_response));
  dcvs_response.type = HAP_power_get_dcvsEnabled;
  int rc_dcvs = HAP_power_get(NULL, &dcvs_response);
  if (rc_dcvs == AEE_SUCCESS) {
    dcvs_enabled = dcvs_response.dcvsEnabled ? 1u : 0u;
  }

  FARF(RUNTIME_HIGH,
       "HEXAGON_RUNTIME_PM_STATUS,stage=%s,clk_freq_hz=%u,"
       "hmx_clk_freq_hz=%u,dcvs_enabled=%u,rc_clk=%d,rc_hmx=%d,rc_dcvs=%d",
       stage ? stage : "unknown", clk_freq_hz, hmx_clk_freq_hz, dcvs_enabled,
       rc_clk, rc_hmx, rc_dcvs);
}

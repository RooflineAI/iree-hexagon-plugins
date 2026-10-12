// Copyright 2025 RooflineAI GmbH
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef HEXAGON_DSP_RT_VTCM_POOL_H
#define HEXAGON_DSP_RT_VTCM_POOL_H

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

#include <stdlib.h>

/**
 * @brief Allocate memory in VTCM.
 * @param[in] nbytes size of the memory to allocate in bytes
 * @return pointer to allocated memory or NULL
 */
void *hexagon_dsp_vtcm_pool_allocate(size_t nbytes);

/**
 * @brief Free memory in VTCM.
 * @param[in] ptr pointer to allocated memory (in contrast to normal free(),
                  NULL is not allowed here (comes from VTCMPool, FIXME))
 */
void hexagon_dsp_vtcm_pool_free(void *ptr);

/**
 * @brief Get the number of VTCM bytes currently allocated.
 *
 * Counts every allocation from the VTCM pool, including the ones generated
 * code makes through hexagon-mlir's runtime, at their allocated (rounded)
 * size.
 * @return allocated bytes, 0 if VTCM has not been acquired
 */
size_t hexagon_dsp_vtcm_pool_get_in_use(void);

/**
 * @brief Restart peak tracking from the bytes currently allocated.
 */
void hexagon_dsp_vtcm_pool_reset_peak(void);

/**
 * @brief Get the highest number of VTCM bytes allocated at once since the last
 *        hexagon_dsp_vtcm_pool_reset_peak().
 * @return peak allocated bytes, 0 if VTCM has not been acquired
 */
size_t hexagon_dsp_vtcm_pool_get_peak(void);

#ifdef __cplusplus
} // extern "C"
#endif // __cplusplus

#endif // #ifndef HEXAGON_DSP_RT_VTCM_POOL_H

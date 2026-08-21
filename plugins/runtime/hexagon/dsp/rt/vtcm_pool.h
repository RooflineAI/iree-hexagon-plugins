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

#ifdef __cplusplus
} // extern "C"
#endif // __cplusplus

#endif // #ifndef HEXAGON_DSP_RT_VTCM_POOL_H

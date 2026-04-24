// Copyright 2025 RooflineAI GmbH

#ifndef HEXAGON_DSP_VTCM_POOL_H
#define HEXAGON_DSP_VTCM_POOL_H

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

#endif // #ifndef HEXAGON_DSP_VTCM_POOL_H

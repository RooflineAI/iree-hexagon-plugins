// Copyright 2026 RooflineAI GmbH
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef HEXAGON_DSP_RT_ASYNC_RUNTIME_H
#define HEXAGON_DSP_RT_ASYNC_RUNTIME_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

/**
 * @brief Run fn(arg) on one of the worker threads of hexagon-mlir's
 *        AsyncRuntime thread pool. The pool is created on first use.
 * @param[in] fn function to run
 * @param[in] arg argument passed to fn
 */
void hexagon_rt_async_execute(void (*fn)(void *), void *arg);

/**
 * @brief Number of worker threads of the AsyncRuntime thread pool (creates
 *        the pool if it does not exist yet).
 */
int64_t hexagon_rt_async_num_workers(void);

#ifdef __cplusplus
} // extern "C"
#endif // __cplusplus

#endif // #ifndef HEXAGON_DSP_RT_ASYNC_RUNTIME_H

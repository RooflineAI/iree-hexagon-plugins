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

// Entry points of the threads inside a VTCM tile, called by generated code.

/**
 * @brief Allocate the coroutine frame of a thread task.
 *
 * Thread-safe; any thread may free a frame another thread allocated. The
 * frame is small control data in DDR. Aborts if out of memory.
 * @param[in] alignment alignment in bytes, a power of two
 * @param[in] size size in bytes
 * @return the frame
 */
void *hexagon_runtime_coro_alloc(int64_t alignment, int64_t size);

/**
 * @brief Free a frame allocated by hexagon_runtime_coro_alloc. Thread-safe.
 * @param[in] frame frame to free
 */
void hexagon_runtime_coro_free(void *frame);

/**
 * @brief Create the group of the thread tasks of one threaded loop.
 *
 * @param[in] num_tasks number of thread tasks that will be added to the group
 * @return the group, an mlirAsyncRuntime AsyncGroup
 */
void *hexagon_runtime_thread_group_create(int64_t num_tasks);

/**
 * @brief Queue task: resume(handle) on a pool worker.
 * @param[in] handle coroutine handle of the task
 * @param[in] resume resume function of the task
 */
void hexagon_runtime_thread_execute(void *handle, void (*resume)(void *));

/**
 * @brief Join the thread pool until all thread tasks of the group have
 *        completed: run queued tasks while there are any, block otherwise.
 * @param[in] group group created by hexagon_runtime_thread_group_create
 */
void hexagon_runtime_thread_await_all(void *group);

#ifdef __cplusplus
} // extern "C"
#endif // __cplusplus

#endif // #ifndef HEXAGON_DSP_RT_ASYNC_RUNTIME_H

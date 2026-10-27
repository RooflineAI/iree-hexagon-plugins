// Copyright 2026 RooflineAI GmbH
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "hexagon/dsp/rt/async_runtime.h"

#include <stdlib.h>

#include "HAP_farf.h"
#include "qcom_hexagon_backend/bin/runtime/multithreading/AsyncRuntime.h"
#include "qurt.h"

// The AsyncRuntime header is C++ only, this file exposes the parts the C
// runtime needs.

void hexagon_rt_async_execute(void (*fn)(void *), void *arg) {
  // A task is a "coroutine handle" plus a resume function, which the worker
  // calls as resume(handle).
  mlir::runtime::mlirAsyncRuntimeExecute(arg, fn);
}

int64_t hexagon_rt_async_num_workers(void) {
  // (sic) the function name is misspelled upstream
  return mlir::runtime::mlirAsyncRuntimGetNumWorkerThreads();
}

// FIXME:
// could probably re-use a pre-allocated pool per dispatch
void *hexagon_runtime_coro_alloc(int64_t alignment, int64_t size) {
  void *frame = memalign((size_t)alignment, (size_t)size);
  if (!frame) {
    FARF(ERROR,
         "HEXAGON-RUNTIME-ERROR: out of memory for a thread task frame of "
         "%lld bytes",
         (long long)size);
    abort();
  }
  return frame;
}

void hexagon_runtime_coro_free(void *frame) { free(frame); }

void *hexagon_runtime_thread_group_create(int64_t num_tasks) {
  return mlir::runtime::mlirAsyncRuntimeCreateGroup(num_tasks);
}

void hexagon_runtime_thread_execute(void *handle, void (*resume)(void *)) {
  mlir::runtime::mlirAsyncRuntimeExecute(handle, resume);
}

// The caller runs queued tasks of the thread pool until all tasks are done
void hexagon_runtime_thread_await_all(void *group) {
  mlir::runtime::mlirAsyncRuntimeAwaitAllInGroupRunningTasks(
      static_cast<mlir::runtime::AsyncGroup *>(group));
}

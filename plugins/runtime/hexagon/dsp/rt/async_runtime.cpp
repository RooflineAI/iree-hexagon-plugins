// Copyright 2026 RooflineAI GmbH
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "hexagon/dsp/rt/async_runtime.h"

#include "qcom_hexagon_backend/bin/runtime/multithreading/AsyncRuntime.h"

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

// Copyright 2026 RooflineAI GmbH
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "llm/vlm.h"

#include "iree/tooling/function_util.h"

struct iree_llm_vision_t {
  iree_allocator_t host_allocator;
  iree_vm_context_t *context;
  iree_hal_device_t *device;
  // Keeps |function| alive.
  iree_vm_module_t *main_module;
  iree_vm_function_t function;
};

iree_status_t iree_llm_vision_create(iree_vm_context_t *context,
                                     iree_hal_device_t *device,
                                     iree_vm_module_t *main_module,
                                     iree_string_view_t function_name,
                                     iree_allocator_t host_allocator,
                                     iree_llm_vision_t **out_vision) {
  *out_vision = NULL;
  iree_vm_function_t function;
  IREE_RETURN_IF_ERROR(iree_vm_module_lookup_function_by_name(
                           main_module, IREE_VM_FUNCTION_LINKAGE_EXPORT,
                           function_name, &function),
                       "looking up vision function '%.*s'",
                       (int)function_name.size, function_name.data);

  iree_llm_vision_t *vision = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(host_allocator, sizeof(*vision), (void **)&vision));
  vision->host_allocator = host_allocator;
  vision->context = context;
  iree_vm_context_retain(context);
  vision->device = device;
  iree_hal_device_retain(device);
  vision->main_module = main_module;
  iree_vm_module_retain(main_module);
  vision->function = function;
  *out_vision = vision;
  return iree_ok_status();
}

void iree_llm_vision_free(iree_llm_vision_t *vision) {
  if (!vision)
    return;
  iree_vm_module_release(vision->main_module);
  iree_hal_device_release(vision->device);
  iree_vm_context_release(vision->context);
  iree_allocator_free(vision->host_allocator, vision);
}

iree_status_t iree_llm_vision_arguments_cconv(const iree_llm_vision_t *vision,
                                              iree_string_view_t *out_cconv) {
  *out_cconv = iree_string_view_empty();
  iree_vm_function_signature_t signature =
      iree_vm_function_signature(&vision->function);
  iree_string_view_t results_cconv = iree_string_view_empty();
  return iree_vm_function_call_get_cconv_fragments(&signature, out_cconv,
                                                   &results_cconv);
}

iree_status_t iree_llm_vision_run(iree_llm_vision_t *vision,
                                  iree_vm_list_t *inputs,
                                  iree_vm_list_t **out_results,
                                  iree_duration_t *out_duration) {
  *out_results = NULL;
  if (out_duration)
    *out_duration = 0;
  iree_allocator_t host_allocator = vision->host_allocator;
  iree_string_view_t name = iree_vm_function_name(&vision->function);

  // A shallow copy, so the fences of an async-external function are not
  // appended to the caller's |inputs|.
  iree_vm_list_t *arguments = NULL;
  IREE_RETURN_IF_ERROR(iree_vm_list_clone(inputs, host_allocator, &arguments));
  iree_hal_fence_t *finish_fence = NULL;
  iree_status_t status = iree_tooling_append_async_fences(
      arguments, vision->function, vision->device,
      /*wait_fence=*/NULL, &finish_fence);
  iree_vm_list_t *results = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_vm_list_create(iree_vm_make_undefined_type_def(), 8,
                                 host_allocator, &results);
  }
  const iree_time_t start = iree_time_now();
  if (iree_status_is_ok(status)) {
    status = iree_status_annotate_f(
        iree_vm_invoke(vision->context, vision->function,
                       IREE_VM_INVOCATION_FLAG_NONE, /*policy=*/NULL, arguments,
                       results, host_allocator),
        "invoking '%.*s'", (int)name.size, name.data);
  }
  if (iree_status_is_ok(status) && finish_fence) {
    status = iree_hal_fence_wait(finish_fence, iree_infinite_timeout(),
                                 IREE_ASYNC_WAIT_FLAG_NONE);
  }
  if (out_duration)
    *out_duration = iree_time_now() - start;
  if (iree_status_is_ok(status)) {
    *out_results = results;
  } else {
    iree_vm_list_release(results);
  }
  iree_hal_fence_release(finish_fence);
  iree_vm_list_release(arguments);
  return status;
}

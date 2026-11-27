// Copyright 2026 RooflineAI GmbH
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// VLM support for the LLM inference engine (llm/engine.h).
//
// A VLM's module exports a vision function (e.g. `vision`, see
// integration_tests/models/qwen3_vl.py) next to prefill and decode. It takes
// the preprocessed image and returns the image features, which prefill takes
// after input_length. Running it in the model's VM context leaves its results
// on the device, so they are passed on to prefill as they are, as
// iree_llm_generate_params_t's prefill_inputs. The prompt must already hold the
// image's tokens (e.g. Qwen3-VL's <|image_pad|> once per image embedding).
//
// Like the engine, this reads no command line flags: the caller creates the
// VM context and HAL device(s) and the input values.
//
// Thread safety: as the model's (see llm/engine.h); the vision function runs
// in the model's context, so callers serialize its calls with the model's.

#ifndef IREE_HEXAGON_PLUGINS_TOOLS_LLM_VLM_H_
#define IREE_HEXAGON_PLUGINS_TOOLS_LLM_VLM_H_

#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/vm/api.h"

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

typedef struct iree_llm_vision_t iree_llm_vision_t;

// Creates a handle on the vision function |function_name| that |main_module|
// exports, to run in |context|, which must contain |main_module| and the
// modules it depends on. |device| is the model's lead device (see
// iree_llm_model_create): the results stay on it, for prefill.
//
// The handle retains |context|, |device| and |main_module|; the caller keeps
// its own references.
iree_status_t iree_llm_vision_create(iree_vm_context_t *context,
                                     iree_hal_device_t *device,
                                     iree_vm_module_t *main_module,
                                     iree_string_view_t function_name,
                                     iree_allocator_t host_allocator,
                                     iree_llm_vision_t **out_vision);

// Frees |vision|. NULL is a no-op.
void iree_llm_vision_free(iree_llm_vision_t *vision);

// Returns the calling convention of the vision function's arguments (e.g. "r"
// for one tensor), as iree_tooling_parse_variants takes it. Valid as long as
// |vision| is.
iree_status_t iree_llm_vision_arguments_cconv(const iree_llm_vision_t *vision,
                                              iree_string_view_t *out_cconv);

// Runs the vision function on |inputs| and waits until it is done on the
// device. |inputs| is not modified; an async function gets its fences on a
// copy.
//
// |out_results| receives the function's results as it returned them: in
// device memory, usable as iree_llm_model_prefill's |extra_inputs| (to copy
// them to the host, use e.g. iree_tooling_transfer_variants). The caller
// releases them. If |out_duration| is not NULL, it receives the time the
// function took, from its invocation until the device finished.
iree_status_t iree_llm_vision_run(iree_llm_vision_t *vision,
                                  iree_vm_list_t *inputs,
                                  iree_vm_list_t **out_results,
                                  iree_duration_t *out_duration);

#ifdef __cplusplus
} // extern "C"
#endif // __cplusplus

#endif // IREE_HEXAGON_PLUGINS_TOOLS_LLM_VLM_H_

// Copyright 2026 RooflineAI GmbH
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "llm/engine.h"

#include <string.h>

#include "iree/hal/api.h"
#include "iree/modules/hal/types.h"
#include "iree/tooling/function_util.h"

//===----------------------------------------------------------------------===//
// Model
//===----------------------------------------------------------------------===//

struct iree_llm_model_t {
  iree_allocator_t host_allocator;
  // The function names are not kept: they are only valid during creation.
  iree_llm_model_options_t options;
  iree_vm_context_t *context;
  iree_hal_device_t *lead_device;
  iree_hal_allocator_t *device_allocator; // Unretained; lead_device's.
  // Retained, as prefill and decode point into it.
  iree_vm_module_t *main_module;
  iree_vm_function_t prefill;
  iree_vm_function_t decode;
  // The position the next decode runs at; 0 until the first prefill.
  iree_host_size_t position;
  // 0, 1, 2, ... up to max_seq_len: every cache_position argument is a prefix.
  int64_t *positions;
  // max_seq_len tokens to pad the prompt in.
  int64_t *input_ids;
};

void iree_llm_model_options_initialize(iree_llm_model_options_t *out_options) {
  *out_options = (iree_llm_model_options_t)IREE_LLM_MODEL_OPTIONS_DEFAULTS;
}

void iree_llm_model_free(iree_llm_model_t *model) {
  if (!model)
    return;
  iree_vm_module_release(model->main_module);
  iree_vm_context_release(model->context);
  iree_hal_device_release(model->lead_device);
  iree_allocator_t host_allocator = model->host_allocator;
  iree_allocator_free(host_allocator, model->input_ids);
  iree_allocator_free(host_allocator, model->positions);
  iree_allocator_free(host_allocator, model);
}

static iree_status_t iree_llm_lookup_function(iree_vm_module_t *module,
                                              iree_string_view_t name,
                                              iree_vm_function_t *out) {
  return iree_status_annotate_f(
      iree_vm_module_lookup_function_by_name(
          module, IREE_VM_FUNCTION_LINKAGE_EXPORT, name, out),
      "looking up function '%.*s'", (int)name.size, name.data);
}

iree_status_t iree_llm_model_create(iree_vm_context_t *context,
                                    iree_hal_device_t *lead_device,
                                    iree_vm_module_t *main_module,
                                    const iree_llm_model_options_t *options,
                                    iree_allocator_t host_allocator,
                                    iree_llm_model_t **out_model) {
  *out_model = NULL;
  if (!context || !lead_device || !main_module) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "a context, a lead device and a main module are "
                            "required");
  }
  if (options->max_seq_len == 0 || options->block_seq_stride == 0 ||
      options->min_prefill_blocks == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "max_seq_len, block_seq_stride and "
                            "min_prefill_blocks must be positive");
  }

  iree_llm_model_t *model = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(host_allocator, sizeof(*model), (void **)&model));
  memset(model, 0, sizeof(*model));
  model->host_allocator = host_allocator;
  model->options = *options;
  model->options.prefill_function = iree_string_view_empty();
  model->options.decode_function = iree_string_view_empty();
  model->context = context;
  iree_vm_context_retain(context);
  model->lead_device = lead_device;
  iree_hal_device_retain(lead_device);
  model->device_allocator = iree_hal_device_allocator(lead_device);
  model->main_module = main_module;
  iree_vm_module_retain(main_module);

  const iree_host_size_t max_seq_len = options->max_seq_len;
  iree_status_t status = iree_allocator_malloc_array(
      host_allocator, max_seq_len, sizeof(int64_t), (void **)&model->positions);
  if (iree_status_is_ok(status)) {
    for (iree_host_size_t i = 0; i < max_seq_len; ++i)
      model->positions[i] = (int64_t)i;
    status = iree_allocator_malloc_array(host_allocator, max_seq_len,
                                         sizeof(int64_t),
                                         (void **)&model->input_ids);
  }
  if (iree_status_is_ok(status)) {
    status = iree_llm_lookup_function(main_module, options->prefill_function,
                                      &model->prefill);
  }
  if (iree_status_is_ok(status)) {
    status = iree_llm_lookup_function(main_module, options->decode_function,
                                      &model->decode);
  }

  if (iree_status_is_ok(status)) {
    *out_model = model;
  } else {
    iree_llm_model_free(model);
  }
  return status;
}

const iree_llm_model_options_t *
iree_llm_model_options(const iree_llm_model_t *model) {
  return &model->options;
}

iree_host_size_t
iree_llm_model_padded_prompt_length(const iree_llm_model_t *model,
                                    iree_host_size_t prompt_length) {
  const iree_host_size_t stride = model->options.block_seq_stride;
  iree_host_size_t blocks = iree_host_size_ceil_div(prompt_length, stride);
  blocks = iree_max(blocks, model->options.min_prefill_blocks);
  return blocks * stride;
}

iree_host_size_t iree_llm_model_position(const iree_llm_model_t *model) {
  return model->position;
}

static iree_status_t
iree_llm_copy_to_mapping(iree_hal_buffer_mapping_t *mapping, void *user_data) {
  const iree_const_byte_span_t *data =
      (const iree_const_byte_span_t *)user_data;
  if (mapping->contents.data_length < data->data_length) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "mapping is smaller than the tensor");
  }
  memcpy(mapping->contents.data, data->data, data->data_length);
  return iree_ok_status();
}

// Appends an i64 tensor of |shape| holding |values| to |list|.
//
// Written through a host mapping of the buffer, as iree-run-module does for
// its inputs: an upload with iree_hal_device_transfer_h2d would need command
// buffer update_buffer, which the Hexagon driver does not implement.
static iree_status_t iree_llm_append_i64(iree_llm_model_t *model,
                                         iree_host_size_t rank,
                                         const iree_hal_dim_t *shape,
                                         const int64_t *values,
                                         iree_vm_list_t *list) {
  iree_host_size_t count = 1;
  for (iree_host_size_t i = 0; i < rank; ++i)
    count *= shape[i];
  const iree_hal_buffer_params_t params = {
      .type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL,
      .usage = IREE_HAL_BUFFER_USAGE_DEFAULT,
  };
  iree_const_byte_span_t data =
      iree_make_const_byte_span(values, count * sizeof(values[0]));
  iree_hal_buffer_view_t *buffer_view = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_buffer_view_generate_buffer(
      model->lead_device, model->device_allocator, rank, shape,
      IREE_HAL_ELEMENT_TYPE_INT_64, IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR,
      params, iree_llm_copy_to_mapping, &data, &buffer_view));
  iree_vm_ref_t ref = iree_hal_buffer_view_move_ref(buffer_view);
  return iree_vm_list_push_ref_move(list, &ref);
}

// Returns the second result in |outputs| of |function|, the logits, retained
// in |out_logits|.
static iree_status_t iree_llm_take_logits(iree_string_view_t name,
                                          iree_vm_list_t *outputs,
                                          iree_hal_buffer_view_t **out_logits) {
  if (iree_vm_list_size(outputs) < 2 ||
      !iree_vm_list_get_buffer_view_assign(outputs, 1)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "'%.*s' returned no logits tensor as its second "
                            "result",
                            (int)name.size, name.data);
  }
  *out_logits = iree_vm_list_get_buffer_view_retain(outputs, 1);
  return iree_ok_status();
}

// Invokes |function| with |inputs| and waits for it to complete. Returns the
// token in the first result, an integer tensor of one element, and, if
// |out_logits| is not NULL, the logits in the second, see
// iree_llm_take_logits.
static iree_status_t iree_llm_invoke(iree_llm_model_t *model,
                                     iree_vm_function_t function,
                                     iree_vm_list_t *inputs, int64_t *out_token,
                                     iree_hal_buffer_view_t **out_logits) {
  iree_string_view_t name = iree_vm_function_name(&function);

  // An async-external function gets fences to wait on; a sync one gets none.
  iree_hal_fence_t *finish_fence = NULL;
  IREE_RETURN_IF_ERROR(
      iree_tooling_append_async_fences(inputs, function, model->lead_device,
                                       /*wait_fence=*/NULL, &finish_fence));

  iree_vm_list_t *outputs = NULL;
  iree_status_t status = iree_vm_list_create(
      iree_vm_make_undefined_type_def(), 2, model->host_allocator, &outputs);
  if (iree_status_is_ok(status)) {
    status = iree_status_annotate_f(
        iree_vm_invoke(model->context, function, IREE_VM_INVOCATION_FLAG_NONE,
                       /*policy=*/NULL, inputs, outputs, model->host_allocator),
        "invoking '%.*s'", (int)name.size, name.data);
  }
  if (iree_status_is_ok(status) && finish_fence) {
    status = iree_hal_fence_wait(finish_fence, iree_infinite_timeout(),
                                 IREE_ASYNC_WAIT_FLAG_NONE);
  }
  iree_hal_fence_release(finish_fence);

  iree_hal_buffer_view_t *token_view = NULL;
  if (iree_status_is_ok(status)) {
    token_view = iree_vm_list_get_buffer_view_assign(outputs, 0);
    if (!token_view) {
      status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "'%.*s' returned no tensor as its first result",
                                (int)name.size, name.data);
    }
  }
  if (iree_status_is_ok(status)) {
    iree_hal_element_type_t type =
        iree_hal_buffer_view_element_type(token_view);
    iree_device_size_t element_size = iree_hal_element_dense_byte_count(type);
    if (!iree_hal_element_numerical_type_is_integer(type) ||
        (element_size != 4 && element_size != 8) ||
        iree_hal_buffer_view_element_count(token_view) != 1) {
      status = iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "'%.*s' must return the next token as one i32 or i64 element",
          (int)name.size, name.data);
    }
    if (iree_status_is_ok(status)) {
      union {
        int32_t i32;
        int64_t i64;
      } value = {0};
      status = iree_hal_device_transfer_d2h(
          model->lead_device, iree_hal_buffer_view_buffer(token_view), 0,
          &value, element_size, IREE_HAL_TRANSFER_BUFFER_FLAG_DEFAULT,
          iree_infinite_timeout());
      *out_token = element_size == 4 ? value.i32 : value.i64;
    }
  }
  if (iree_status_is_ok(status) && out_logits) {
    status = iree_llm_take_logits(name, outputs, out_logits);
  }
  // Releases the remaining results unread, e.g. prefill's logits if not
  // asked for.
  iree_vm_list_release(outputs);
  return status;
}

iree_status_t iree_llm_model_prefill(iree_llm_model_t *model,
                                     const int64_t *prompt,
                                     iree_host_size_t prompt_length,
                                     int64_t pad_token, int64_t *out_token,
                                     iree_hal_buffer_view_t **out_logits) {
  if (out_logits)
    *out_logits = NULL;
  if (prompt_length == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "the prompt has no tokens");
  }
  const iree_host_size_t max_seq_len = model->options.max_seq_len;
  const iree_host_size_t padded_length =
      iree_llm_model_padded_prompt_length(model, prompt_length);
  if (padded_length > max_seq_len) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "the prompt has %zu tokens, %zu padded to a "
                            "multiple of block_seq_stride=%zu, more than "
                            "max_seq_len=%zu",
                            prompt_length, padded_length,
                            model->options.block_seq_stride, max_seq_len);
  }
  // Whatever happens, the previous sequence's cache is no longer valid.
  model->position = 0;

  int64_t *input_ids = model->input_ids;
  memcpy(input_ids, prompt, prompt_length * sizeof(int64_t));
  for (iree_host_size_t i = prompt_length; i < padded_length; ++i) {
    input_ids[i] = pad_token;
  }
  const int64_t input_length = (int64_t)prompt_length - 1;

  iree_vm_list_t *inputs = NULL;
  IREE_RETURN_IF_ERROR(iree_vm_list_create(iree_vm_make_undefined_type_def(), 3,
                                           model->host_allocator, &inputs));
  const iree_hal_dim_t input_ids_shape[2] = {1, padded_length};
  const iree_hal_dim_t positions_shape[1] = {padded_length};
  const iree_hal_dim_t input_length_shape[1] = {1};
  iree_status_t status =
      iree_llm_append_i64(model, 2, input_ids_shape, input_ids, inputs);
  if (iree_status_is_ok(status)) {
    status = iree_llm_append_i64(model, 1, positions_shape, model->positions,
                                 inputs);
  }
  if (iree_status_is_ok(status)) {
    status = iree_llm_append_i64(model, 1, input_length_shape, &input_length,
                                 inputs);
  }
  if (iree_status_is_ok(status)) {
    status =
        iree_llm_invoke(model, model->prefill, inputs, out_token, out_logits);
  }
  iree_vm_list_release(inputs);
  if (iree_status_is_ok(status))
    model->position = prompt_length;
  return status;
}

iree_status_t iree_llm_model_decode(iree_llm_model_t *model, int64_t token,
                                    int64_t *out_token) {
  const iree_host_size_t position = model->position;
  if (position == 0) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "decode needs a prefill first");
  }
  if (position >= model->options.max_seq_len) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "the KV cache is full at max_seq_len=%zu tokens",
                            model->options.max_seq_len);
  }

  iree_vm_list_t *inputs = NULL;
  IREE_RETURN_IF_ERROR(iree_vm_list_create(iree_vm_make_undefined_type_def(), 2,
                                           model->host_allocator, &inputs));
  const iree_hal_dim_t input_ids_shape[2] = {1, 1};
  const iree_hal_dim_t positions_shape[1] = {position + 1};
  iree_status_t status =
      iree_llm_append_i64(model, 2, input_ids_shape, &token, inputs);
  if (iree_status_is_ok(status)) {
    status = iree_llm_append_i64(model, 1, positions_shape, model->positions,
                                 inputs);
  }
  if (iree_status_is_ok(status)) {
    status = iree_llm_invoke(model, model->decode, inputs, out_token,
                             /*out_logits=*/NULL);
  }
  iree_vm_list_release(inputs);
  if (iree_status_is_ok(status))
    ++model->position;
  return status;
}

//===----------------------------------------------------------------------===//
// Generation
//===----------------------------------------------------------------------===//

const char *iree_llm_stop_reason_string(iree_llm_stop_reason_t reason) {
  switch (reason) {
  case IREE_LLM_STOP_REASON_MAX_NEW_TOKENS:
    return "max_new_tokens reached";
  case IREE_LLM_STOP_REASON_STOP_TOKEN:
    return "stop token";
  case IREE_LLM_STOP_REASON_MAX_SEQ_LEN:
    return "max_seq_len reached";
  case IREE_LLM_STOP_REASON_CALLBACK:
    return "stopped by caller";
  default:
    return "unknown";
  }
}

static bool iree_llm_is_stop_token(const iree_llm_generate_params_t *params,
                                   int64_t token) {
  for (iree_host_size_t i = 0; i < params->stop_token_count; ++i) {
    if (params->stop_tokens[i] == token)
      return true;
  }
  return false;
}

iree_status_t iree_llm_generate(iree_llm_model_t *model, const int64_t *prompt,
                                iree_host_size_t prompt_length,
                                const iree_llm_generate_params_t *params,
                                iree_llm_token_callback_t callback,
                                iree_llm_generate_stats_t *out_stats,
                                iree_hal_buffer_view_t **out_prefill_logits) {
  memset(out_stats, 0, sizeof(*out_stats));
  if (out_prefill_logits)
    *out_prefill_logits = NULL;
  out_stats->prompt_length = prompt_length;
  out_stats->padded_length =
      iree_llm_model_padded_prompt_length(model, prompt_length);
  out_stats->stop_reason = IREE_LLM_STOP_REASON_MAX_NEW_TOKENS;
  if (prompt_length == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "the prompt has no tokens");
  }
  const iree_host_size_t max_seq_len = model->options.max_seq_len;
  if (out_stats->padded_length > max_seq_len) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "the prompt has %zu tokens, %zu padded to a multiple of "
        "block_seq_stride=%zu, more than max_seq_len=%zu",
        prompt_length, out_stats->padded_length,
        model->options.block_seq_stride, max_seq_len);
  }
  if (params->max_new_tokens == 0)
    return iree_ok_status();

  int64_t token = 0;
  const iree_time_t prefill_start = iree_time_now();
  iree_status_t status =
      iree_llm_model_prefill(model, prompt, prompt_length, params->pad_token,
                             &token, out_prefill_logits);
  out_stats->prefill_duration = iree_time_now() - prefill_start;
  while (iree_status_is_ok(status)) {
    if (iree_llm_is_stop_token(params, token)) {
      out_stats->stop_reason = IREE_LLM_STOP_REASON_STOP_TOKEN;
      break;
    }
    ++out_stats->generated_count;
    bool stop = false;
    status = callback.fn(callback.user_data, token, &stop);
    if (!iree_status_is_ok(status))
      break;
    if (stop) {
      out_stats->stop_reason = IREE_LLM_STOP_REASON_CALLBACK;
      break;
    }
    if (out_stats->generated_count == params->max_new_tokens)
      break;
    if (iree_llm_model_position(model) >= max_seq_len) {
      out_stats->stop_reason = IREE_LLM_STOP_REASON_MAX_SEQ_LEN;
      break;
    }
    // Only the decode itself is timed, not the callback or the checks.
    const iree_time_t decode_start = iree_time_now();
    status = iree_llm_model_decode(model, token, &token);
    out_stats->decode_duration += iree_time_now() - decode_start;
    ++out_stats->decode_steps;
  }
  return status;
}

// Copyright 2026 RooflineAI GmbH
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Runs a compiled causal LM in a greedy token generation loop.
//
// The module is expected to export the two functions of
// integration_tests/models/kv_cache.py's KvCacheLM, which keeps its KV cache
// in module globals shared by both:
//
//   prefill(input_ids: i64[1, P], cache_position: i64[P],
//           input_length: i64[1]) -> (next_token: i64[1, 1], logits)
//     Runs the prompt, padded to P, a multiple of --block_seq_stride of at
//     least --min_prefill_blocks blocks. input_length is the index of the last
//     real token, whose argmax is next_token.
//   decode(input_ids: i64[1, 1], cache_position: i64[pos + 1])
//       -> next_token: i64[1, 1]
//     Runs one token at position pos against the cache.
//
// Both run in one VM context, so the cache prefill writes is the one decode
// reads. The prompt is tokenized with a HuggingFace tokenizer.json, and each
// generated token is detokenized and printed to stdout as it is produced.
//
// Example:
//   iree-run-llm --device=hexagon --module=model.vmfb \
//       --tokenizer=tokenizer.json --prompt="The capital of France is" \
//       --max_new_tokens=32

#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "iree/base/api.h"
#include "iree/base/tooling/flags.h"
#include "iree/hal/api.h"
#include "iree/hal/replay/recorder.h"
#include "iree/io/file_contents.h"
#include "iree/modules/hal/types.h"
#include "iree/tokenizer/format/huggingface/tokenizer_json.h"
#include "iree/tokenizer/tokenizer.h"
#include "iree/tokenizer/vocab/vocab.h"
#include "iree/tooling/context_util.h"
#include "iree/tooling/function_util.h"
#include "iree/vm/api.h"

//===----------------------------------------------------------------------===//
// Flags
//===----------------------------------------------------------------------===//

IREE_FLAG(string, tokenizer, "", "Path to a HuggingFace tokenizer.json.");
IREE_FLAG(string, prompt, "",
          "Prompt text, used as is: apply any chat template beforehand.");
IREE_FLAG(string, prompt_file, "",
          "File to read the prompt from, instead of --prompt.");
IREE_FLAG(bool, add_special_tokens, true,
          "Add the tokenizer's special tokens (e.g. BOS) to the prompt, as "
          "HuggingFace's tokenizer(prompt) does by default.");
IREE_FLAG(int32_t, max_new_tokens, 64, "Maximum number of tokens to generate.");
IREE_FLAG(int32_t, max_seq_len, 128,
          "Context length the module was exported for (KvCacheLM's "
          "max_seq_len): the padded prompt and every decode position must "
          "fit in it.");
IREE_FLAG(int32_t, block_seq_stride, 32,
          "The prefill sequence length must be a multiple of this (KvCacheLM's "
          "block_seq_stride).");
IREE_FLAG(int32_t, min_prefill_blocks, 2,
          "Minimum number of --block_seq_stride blocks prefill accepts.");
IREE_FLAG(int32_t, pad_token_id, -1,
          "Token ID used to pad the prompt. Padding is never attended to, so "
          "any valid ID works. Defaults to the tokenizer's pad token, else 0.");
IREE_FLAG_LIST(string, stop_token_id,
               "Token ID that ends generation; repeatable. Defaults to the "
               "tokenizer's EOS token, if it has one.");
IREE_FLAG(bool, ignore_eos, false,
          "Generate --max_new_tokens tokens regardless of stop tokens.");
IREE_FLAG(bool, print_special_tokens, false,
          "Print special tokens (BOS/EOS/...) in the generated text.");
IREE_FLAG(bool, print_token_ids, false,
          "Print prompt and generated token IDs to stderr.");
IREE_FLAG(bool, print_stats, true, "Print timing statistics to stderr.");
IREE_FLAG(string, prefill_function, "prefill",
          "Name of the exported prefill function.");
IREE_FLAG(string, decode_function, "decode",
          "Name of the exported decode function.");

//===----------------------------------------------------------------------===//
// Model
//===----------------------------------------------------------------------===//

typedef struct iree_run_llm_model_t {
  iree_allocator_t host_allocator;
  iree_vm_context_t *context;
  iree_hal_device_list_t *device_list;
  iree_hal_device_t *device; // Unretained; the first of device_list.
  iree_hal_allocator_t *device_allocator;
  iree_hal_replay_recorder_t *replay_recorder;
  iree_vm_function_t prefill;
  iree_vm_function_t decode;
} iree_run_llm_model_t;

static void iree_run_llm_model_deinitialize(iree_run_llm_model_t *model) {
  iree_vm_context_release(model->context);
  if (model->replay_recorder) {
    iree_status_ignore(iree_hal_replay_recorder_close(model->replay_recorder));
    iree_hal_replay_recorder_release(model->replay_recorder);
  }
  iree_hal_allocator_release(model->device_allocator);
  iree_hal_device_list_free(model->device_list);
  memset(model, 0, sizeof(*model));
}

static iree_status_t iree_run_llm_lookup_function(iree_vm_module_t *module,
                                                  const char *name,
                                                  iree_vm_function_t *out) {
  return iree_status_annotate_f(iree_vm_module_lookup_function_by_name(
                                    module, IREE_VM_FUNCTION_LINKAGE_EXPORT,
                                    iree_make_cstring_view(name), out),
                                "looking up function '%s'", name);
}

// Loads the --module= modules and creates one context for all of them, on the
// --device= device. The functions are looked up in the last module.
static iree_status_t
iree_run_llm_model_initialize(iree_vm_instance_t *instance,
                              iree_allocator_t host_allocator,
                              iree_run_llm_model_t *out_model) {
  memset(out_model, 0, sizeof(*out_model));
  out_model->host_allocator = host_allocator;

  iree_tooling_module_list_t module_list;
  iree_tooling_module_list_initialize(&module_list);
  IREE_RETURN_IF_ERROR(iree_tooling_load_modules_from_flags(
                           instance, host_allocator, &module_list),
                       "loading modules");
  iree_vm_module_t *main_module = iree_tooling_module_list_back(&module_list);
  iree_status_t status = iree_ok_status();
  if (!main_module) {
    status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "no module given; use --module=model.vmfb");
  }
  if (iree_status_is_ok(status)) {
    status = iree_status_annotate_f(
        iree_tooling_create_context_from_flags(
            instance, module_list.count, module_list.values,
            /*default_device_uri=*/iree_string_view_empty(), host_allocator,
            &out_model->context, &out_model->device_list,
            &out_model->device_allocator, &out_model->replay_recorder),
        "creating VM context");
  }
  if (iree_status_is_ok(status)) {
    out_model->device = out_model->device_list
                            ? iree_hal_device_list_at(out_model->device_list, 0)
                            : NULL;
    if (!out_model->device) {
      status = iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                                "the module does not use a HAL device");
    }
  }
  if (iree_status_is_ok(status)) {
    status = iree_run_llm_lookup_function(main_module, FLAG_prefill_function,
                                          &out_model->prefill);
  }
  if (iree_status_is_ok(status)) {
    status = iree_run_llm_lookup_function(main_module, FLAG_decode_function,
                                          &out_model->decode);
  }
  iree_tooling_module_list_reset(&module_list);
  if (!iree_status_is_ok(status))
    iree_run_llm_model_deinitialize(out_model);
  return status;
}

static iree_status_t
iree_run_llm_copy_to_mapping(iree_hal_buffer_mapping_t *mapping,
                             void *user_data) {
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
static iree_status_t iree_run_llm_append_i64(iree_run_llm_model_t *model,
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
      model->device, model->device_allocator, rank, shape,
      IREE_HAL_ELEMENT_TYPE_INT_64, IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR,
      params, iree_run_llm_copy_to_mapping, &data, &buffer_view));
  iree_vm_ref_t ref = iree_hal_buffer_view_move_ref(buffer_view);
  return iree_vm_list_push_ref_move(list, &ref);
}

// Invokes |function| with |inputs| and waits for it to complete. Returns the
// token in the first result, an integer tensor of one element.
static iree_status_t iree_run_llm_invoke(iree_run_llm_model_t *model,
                                         iree_vm_function_t function,
                                         iree_vm_list_t *inputs,
                                         int64_t *out_token) {
  iree_string_view_t name = iree_vm_function_name(&function);

  // An async-external function gets fences to wait on; a sync one gets none.
  iree_hal_fence_t *finish_fence = NULL;
  IREE_RETURN_IF_ERROR(iree_tooling_append_async_fences(
      inputs, function, model->device, /*wait_fence=*/NULL, &finish_fence));

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
          model->device, iree_hal_buffer_view_buffer(token_view), 0, &value,
          element_size, IREE_HAL_TRANSFER_BUFFER_FLAG_DEFAULT,
          iree_infinite_timeout());
      *out_token = element_size == 4 ? value.i32 : value.i64;
    }
  }
  // Releases the remaining results unread, e.g. prefill's logits.
  iree_vm_list_release(outputs);
  return status;
}

// Runs prefill on |prompt|, padded to |padded_length|, and returns the token
// following it. |positions| holds 0, 1, ... for at least |padded_length|.
static iree_status_t
iree_run_llm_prefill(iree_run_llm_model_t *model, const int64_t *prompt,
                     iree_host_size_t prompt_length,
                     iree_host_size_t padded_length, int64_t pad_token,
                     const int64_t *positions, int64_t *out_token) {
  int64_t *input_ids = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc_array(model->host_allocator, padded_length,
                                  sizeof(int64_t), (void **)&input_ids));
  memcpy(input_ids, prompt, prompt_length * sizeof(int64_t));
  for (iree_host_size_t i = prompt_length; i < padded_length; ++i) {
    input_ids[i] = pad_token;
  }
  const int64_t input_length = (int64_t)prompt_length - 1;

  iree_vm_list_t *inputs = NULL;
  iree_status_t status = iree_vm_list_create(iree_vm_make_undefined_type_def(),
                                             3, model->host_allocator, &inputs);
  const iree_hal_dim_t input_ids_shape[2] = {1, padded_length};
  const iree_hal_dim_t positions_shape[1] = {padded_length};
  const iree_hal_dim_t input_length_shape[1] = {1};
  if (iree_status_is_ok(status)) {
    status =
        iree_run_llm_append_i64(model, 2, input_ids_shape, input_ids, inputs);
  }
  if (iree_status_is_ok(status)) {
    status =
        iree_run_llm_append_i64(model, 1, positions_shape, positions, inputs);
  }
  if (iree_status_is_ok(status)) {
    status = iree_run_llm_append_i64(model, 1, input_length_shape,
                                     &input_length, inputs);
  }
  if (iree_status_is_ok(status)) {
    status = iree_run_llm_invoke(model, model->prefill, inputs, out_token);
  }
  iree_vm_list_release(inputs);
  iree_allocator_free(model->host_allocator, input_ids);
  return status;
}

// Runs decode on |token| at |position| and returns the token following it.
// |positions| holds 0, 1, ... for at least |position| + 1.
static iree_status_t iree_run_llm_decode(iree_run_llm_model_t *model,
                                         int64_t token,
                                         iree_host_size_t position,
                                         const int64_t *positions,
                                         int64_t *out_token) {
  iree_vm_list_t *inputs = NULL;
  IREE_RETURN_IF_ERROR(iree_vm_list_create(iree_vm_make_undefined_type_def(), 2,
                                           model->host_allocator, &inputs));
  const iree_hal_dim_t input_ids_shape[2] = {1, 1};
  const iree_hal_dim_t positions_shape[1] = {position + 1};
  iree_status_t status =
      iree_run_llm_append_i64(model, 2, input_ids_shape, &token, inputs);
  if (iree_status_is_ok(status)) {
    status =
        iree_run_llm_append_i64(model, 1, positions_shape, positions, inputs);
  }
  if (iree_status_is_ok(status)) {
    status = iree_run_llm_invoke(model, model->decode, inputs, out_token);
  }
  iree_vm_list_release(inputs);
  return status;
}

//===----------------------------------------------------------------------===//
// Tokenizer
//===----------------------------------------------------------------------===//

static iree_status_t
iree_run_llm_load_tokenizer(iree_allocator_t host_allocator,
                            iree_tokenizer_t **out_tokenizer) {
  if (FLAG_tokenizer[0] == '\0') {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "missing --tokenizer=tokenizer.json");
  }
  iree_io_file_contents_t *contents = NULL;
  IREE_RETURN_IF_ERROR(
      iree_io_file_contents_read(iree_make_cstring_view(FLAG_tokenizer),
                                 host_allocator, &contents),
      "reading '%s'", FLAG_tokenizer);
  iree_status_t status = iree_status_annotate_f(
      iree_tokenizer_from_huggingface_json(
          iree_make_string_view((const char *)contents->const_buffer.data,
                                contents->const_buffer.data_length),
          host_allocator, out_tokenizer),
      "loading tokenizer '%s'", FLAG_tokenizer);
  iree_io_file_contents_free(contents);
  return status;
}

// Tokenizes the prompt from --prompt or --prompt_file. The caller frees
// |out_tokens| with |host_allocator|.
static iree_status_t
iree_run_llm_encode_prompt(const iree_tokenizer_t *tokenizer,
                           iree_allocator_t host_allocator,
                           int64_t **out_tokens, iree_host_size_t *out_count) {
  *out_tokens = NULL;
  *out_count = 0;

  iree_io_file_contents_t *contents = NULL;
  iree_string_view_t text = iree_make_cstring_view(FLAG_prompt);
  if (FLAG_prompt_file[0] != '\0') {
    if (FLAG_prompt[0] != '\0') {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "pass only one of --prompt and --prompt_file");
    }
    IREE_RETURN_IF_ERROR(
        iree_io_file_contents_read(iree_make_cstring_view(FLAG_prompt_file),
                                   host_allocator, &contents),
        "reading '%s'", FLAG_prompt_file);
    text = iree_make_string_view((const char *)contents->const_buffer.data,
                                 contents->const_buffer.data_length);
  }

  iree_tokenizer_encode_flags_t flags =
      IREE_TOKENIZER_ENCODE_FLAG_AT_INPUT_START;
  if (FLAG_add_special_tokens) {
    flags |= IREE_TOKENIZER_ENCODE_FLAG_ADD_SPECIAL_TOKENS;
  }
  // No token is shorter than a byte, and add room for the special tokens.
  iree_host_size_t capacity =
      text.size + iree_tokenizer_max_special_token_count(tokenizer) + 1;
  iree_tokenizer_token_id_t *ids = NULL;
  iree_status_t status = iree_allocator_malloc_array(
      host_allocator, capacity, sizeof(*ids), (void **)&ids);
  iree_host_size_t count = 0;
  if (iree_status_is_ok(status)) {
    status = iree_tokenizer_encode(
        tokenizer, text, flags,
        iree_tokenizer_make_token_output(ids, NULL, NULL, capacity),
        host_allocator, &count);
  }
  int64_t *tokens = NULL;
  if (iree_status_is_ok(status) && count == 0) {
    status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "the prompt has no tokens; pass --prompt");
  }
  if (iree_status_is_ok(status)) {
    status = iree_allocator_malloc_array(host_allocator, count, sizeof(*tokens),
                                         (void **)&tokens);
  }
  if (iree_status_is_ok(status)) {
    for (iree_host_size_t i = 0; i < count; ++i)
      tokens[i] = ids[i];
    *out_tokens = tokens;
    *out_count = count;
  }
  iree_allocator_free(host_allocator, ids);
  iree_io_file_contents_free(contents);
  return status;
}

// Streams detokenized text to stdout, one token at a time. Byte-level tokens
// that only form a character together are held back until it is complete.
typedef struct iree_run_llm_printer_t {
  iree_allocator_t host_allocator;
  void *state_storage;
  iree_tokenizer_decode_state_t *state;
  char text[IREE_TOKENIZER_DECODE_OUTPUT_RECOMMENDED_SIZE];
} iree_run_llm_printer_t;

static iree_status_t
iree_run_llm_printer_initialize(const iree_tokenizer_t *tokenizer,
                                iree_allocator_t host_allocator,
                                iree_run_llm_printer_t *out_printer) {
  memset(out_printer, 0, sizeof(*out_printer));
  out_printer->host_allocator = host_allocator;
  iree_host_size_t state_size = 0;
  IREE_RETURN_IF_ERROR(
      iree_tokenizer_decode_state_calculate_size(tokenizer, &state_size));
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(host_allocator, state_size,
                                             &out_printer->state_storage));
  iree_tokenizer_decode_flags_t flags =
      FLAG_print_special_tokens
          ? IREE_TOKENIZER_DECODE_FLAG_NONE
          : IREE_TOKENIZER_DECODE_FLAG_SKIP_SPECIAL_TOKENS;
  iree_status_t status = iree_tokenizer_decode_state_initialize(
      tokenizer, flags,
      iree_make_byte_span(out_printer->state_storage, state_size),
      &out_printer->state);
  if (!iree_status_is_ok(status)) {
    iree_allocator_free(host_allocator, out_printer->state_storage);
    out_printer->state_storage = NULL;
  }
  return status;
}

static iree_status_t iree_run_llm_printer_print(iree_run_llm_printer_t *printer,
                                                int64_t token) {
  iree_tokenizer_token_id_t id = (iree_tokenizer_token_id_t)token;
  iree_tokenizer_token_id_list_t tokens =
      iree_tokenizer_make_token_id_list(&id, 1);
  while (tokens.count > 0) {
    iree_host_size_t consumed = 0, length = 0;
    IREE_RETURN_IF_ERROR(iree_tokenizer_decode_state_feed(
        printer->state, tokens,
        iree_make_mutable_string_view(printer->text, sizeof(printer->text)),
        &consumed, &length));
    if (consumed == 0 && length == 0) {
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "token %" PRId64 " does not fit the text buffer",
                              token);
    }
    fwrite(printer->text, 1, length, stdout);
    tokens.values += consumed;
    tokens.count -= consumed;
  }
  fflush(stdout);
  return iree_ok_status();
}

// Prints the text still held back and ends the line.
static iree_status_t
iree_run_llm_printer_finish(iree_run_llm_printer_t *printer) {
  iree_host_size_t length = 0;
  IREE_RETURN_IF_ERROR(iree_tokenizer_decode_state_finalize(
      printer->state,
      iree_make_mutable_string_view(printer->text, sizeof(printer->text)),
      &length));
  fwrite(printer->text, 1, length, stdout);
  fputc('\n', stdout);
  fflush(stdout);
  return iree_ok_status();
}

static void iree_run_llm_printer_deinitialize(iree_run_llm_printer_t *printer) {
  if (printer->state)
    iree_tokenizer_decode_state_deinitialize(printer->state);
  iree_allocator_free(printer->host_allocator, printer->state_storage);
  memset(printer, 0, sizeof(*printer));
}

//===----------------------------------------------------------------------===//
// Generation
//===----------------------------------------------------------------------===//

typedef struct iree_run_llm_stop_tokens_t {
  iree_host_size_t count;
  int64_t values[16];
} iree_run_llm_stop_tokens_t;

static iree_status_t
iree_run_llm_parse_stop_tokens(const iree_tokenizer_t *tokenizer,
                               iree_run_llm_stop_tokens_t *out_stop) {
  memset(out_stop, 0, sizeof(*out_stop));
  if (FLAG_ignore_eos)
    return iree_ok_status();
  const iree_flag_string_list_t list = FLAG_stop_token_id_list();
  if (list.count > IREE_ARRAYSIZE(out_stop->values)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "at most %zu --stop_token_id flags",
                            IREE_ARRAYSIZE(out_stop->values));
  }
  for (iree_host_size_t i = 0; i < list.count; ++i) {
    int64_t value = 0;
    if (!iree_string_view_atoi_int64(list.values[i], &value)) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "--stop_token_id='%.*s' is not an integer",
                              (int)list.values[i].size, list.values[i].data);
    }
    out_stop->values[out_stop->count++] = value;
  }
  if (out_stop->count == 0) {
    int32_t eos =
        iree_tokenizer_vocab_special_ids(iree_tokenizer_vocab(tokenizer)).eos;
    if (eos >= 0)
      out_stop->values[out_stop->count++] = eos;
  }
  return iree_ok_status();
}

static bool iree_run_llm_is_stop_token(const iree_run_llm_stop_tokens_t *stop,
                                       int64_t token) {
  for (iree_host_size_t i = 0; i < stop->count; ++i) {
    if (stop->values[i] == token)
      return true;
  }
  return false;
}

static void iree_run_llm_print_ids(const char *label, const int64_t *tokens,
                                   iree_host_size_t count) {
  fprintf(stderr, "%s:", label);
  for (iree_host_size_t i = 0; i < count; ++i) {
    fprintf(stderr, "%s%" PRId64, i ? "," : " ", tokens[i]);
  }
  fputc('\n', stderr);
}

static iree_status_t iree_run_llm_generate(iree_run_llm_model_t *model,
                                           const iree_tokenizer_t *tokenizer,
                                           iree_allocator_t host_allocator) {
  if (FLAG_block_seq_stride <= 0 || FLAG_min_prefill_blocks <= 0 ||
      FLAG_max_seq_len <= 0 || FLAG_max_new_tokens < 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "--block_seq_stride, --min_prefill_blocks and "
                            "--max_seq_len must be positive, and "
                            "--max_new_tokens not negative");
  }
  const iree_host_size_t max_seq_len = (iree_host_size_t)FLAG_max_seq_len;
  const iree_host_size_t stride = (iree_host_size_t)FLAG_block_seq_stride;

  iree_run_llm_stop_tokens_t stop;
  IREE_RETURN_IF_ERROR(iree_run_llm_parse_stop_tokens(tokenizer, &stop));
  int64_t pad_token = FLAG_pad_token_id;
  if (pad_token < 0) {
    int32_t pad =
        iree_tokenizer_vocab_special_ids(iree_tokenizer_vocab(tokenizer)).pad;
    pad_token = pad >= 0 ? pad : 0;
  }

  const iree_time_t encode_start = iree_time_now();
  int64_t *prompt = NULL;
  iree_host_size_t prompt_length = 0;
  IREE_RETURN_IF_ERROR(iree_run_llm_encode_prompt(tokenizer, host_allocator,
                                                  &prompt, &prompt_length));
  const iree_time_t encode_end = iree_time_now();

  iree_host_size_t blocks = iree_host_size_ceil_div(prompt_length, stride);
  blocks = iree_max(blocks, (iree_host_size_t)FLAG_min_prefill_blocks);
  const iree_host_size_t padded_length = blocks * stride;
  iree_status_t status = iree_ok_status();
  if (padded_length > max_seq_len) {
    status = iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "the prompt has %zu tokens, %zu padded to a multiple of "
        "--block_seq_stride=%zu, more than --max_seq_len=%zu",
        prompt_length, padded_length, stride, max_seq_len);
  }
  if (FLAG_print_token_ids) {
    iree_run_llm_print_ids("prompt token ids", prompt, prompt_length);
  }

  // 0, 1, 2, ...: every cache_position argument is a prefix of it.
  int64_t *positions = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_allocator_malloc_array(
        host_allocator, max_seq_len, sizeof(*positions), (void **)&positions);
  }
  int64_t *generated = NULL;
  if (iree_status_is_ok(status)) {
    for (iree_host_size_t i = 0; i < max_seq_len; ++i)
      positions[i] = i;
    status = iree_allocator_malloc_array(
        host_allocator, iree_max(FLAG_max_new_tokens, 1), sizeof(*generated),
        (void **)&generated);
  }
  iree_run_llm_printer_t printer;
  memset(&printer, 0, sizeof(printer));
  if (iree_status_is_ok(status)) {
    status =
        iree_run_llm_printer_initialize(tokenizer, host_allocator, &printer);
  }

  iree_host_size_t generated_count = 0;
  const char *stop_reason = "--max_new_tokens reached";
  iree_time_t prefill_start = 0, prefill_end = 0, decode_end = 0;
  if (iree_status_is_ok(status) && FLAG_max_new_tokens > 0) {
    int64_t token = 0;
    prefill_start = iree_time_now();
    status = iree_run_llm_prefill(model, prompt, prompt_length, padded_length,
                                  pad_token, positions, &token);
    prefill_end = decode_end = iree_time_now();
    // The token at |position| is the one decode runs next.
    iree_host_size_t position = prompt_length;
    while (iree_status_is_ok(status)) {
      if (iree_run_llm_is_stop_token(&stop, token)) {
        stop_reason = "stop token";
        break;
      }
      generated[generated_count++] = token;
      status = iree_run_llm_printer_print(&printer, token);
      if (!iree_status_is_ok(status))
        break;
      if (generated_count == (iree_host_size_t)FLAG_max_new_tokens)
        break;
      if (position >= max_seq_len) {
        stop_reason = "--max_seq_len reached";
        break;
      }
      status = iree_run_llm_decode(model, token, position, positions, &token);
      decode_end = iree_time_now();
      ++position;
    }
  }
  if (iree_status_is_ok(status)) {
    status = iree_run_llm_printer_finish(&printer);
  }

  if (iree_status_is_ok(status) && FLAG_print_token_ids) {
    iree_run_llm_print_ids("generated token ids", generated, generated_count);
  }
  if (iree_status_is_ok(status) && FLAG_print_stats) {
    const double encode_ms = (encode_end - encode_start) / 1e6;
    const double prefill_ms = (prefill_end - prefill_start) / 1e6;
    const double decode_ms = (decode_end - prefill_end) / 1e6;
    // The first token comes from prefill, every further one from a decode.
    const iree_host_size_t decode_steps =
        generated_count > 0 ? generated_count - 1 : 0;
    fprintf(stderr,
            "\n--- iree-run-llm ---\n"
            "prompt:  %zu tokens (padded to %zu), tokenized in %.2f ms\n"
            "prefill: %.2f ms (%.1f tokens/s)\n"
            "decode:  %zu steps in %.2f ms (%.2f ms/token, %.2f tokens/s)\n"
            "generated %zu tokens, stopped: %s\n",
            prompt_length, padded_length, encode_ms, prefill_ms,
            prefill_ms > 0 ? prompt_length / (prefill_ms / 1e3) : 0.0,
            decode_steps, decode_ms,
            decode_steps ? decode_ms / decode_steps : 0.0,
            decode_ms > 0 ? decode_steps / (decode_ms / 1e3) : 0.0,
            generated_count, stop_reason);
  }

  iree_run_llm_printer_deinitialize(&printer);
  iree_allocator_free(host_allocator, generated);
  iree_allocator_free(host_allocator, positions);
  iree_allocator_free(host_allocator, prompt);
  return status;
}

//===----------------------------------------------------------------------===//
// Main
//===----------------------------------------------------------------------===//

int main(int argc, char **argv) {
  IREE_TRACE_APP_ENTER();
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_flags_set_usage(
      "iree-run-llm",
      "Runs a compiled causal LM, exported as `prefill` and `decode` with the\n"
      "KV cache in module state, in a greedy generation loop: tokenizes the\n"
      "prompt, prefills it, decodes token by token and prints the text.\n"
      "\n"
      "Usage:\n"
      "  iree-run-llm --device=hexagon --module=model.vmfb \\\n"
      "      --tokenizer=tokenizer.json --prompt=\"Once upon a time\" \\\n"
      "      --max_new_tokens=32\n"
      "\n"
      "--max_seq_len, --block_seq_stride and --min_prefill_blocks must match\n"
      "the shapes the module was exported with.\n");
  iree_flags_parse_checked(IREE_FLAGS_PARSE_MODE_DEFAULT, &argc, &argv);

  iree_allocator_t host_allocator = iree_allocator_system();
  iree_tokenizer_t *tokenizer = NULL;
  iree_status_t status =
      iree_run_llm_load_tokenizer(host_allocator, &tokenizer);

  iree_vm_instance_t *instance = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_tooling_create_instance(host_allocator, &instance);
  }
  iree_run_llm_model_t model;
  memset(&model, 0, sizeof(model));
  if (iree_status_is_ok(status)) {
    status = iree_run_llm_model_initialize(instance, host_allocator, &model);
  }
  if (iree_status_is_ok(status)) {
    status = iree_run_llm_generate(&model, tokenizer, host_allocator);
  }

  iree_run_llm_model_deinitialize(&model);
  iree_vm_instance_release(instance);
  if (tokenizer)
    iree_tokenizer_free(tokenizer);

  int exit_code = EXIT_SUCCESS;
  if (!iree_status_is_ok(status)) {
    fflush(stdout);
    iree_status_fprint(stderr, status);
    iree_status_free(status);
    exit_code = EXIT_FAILURE;
  }
  IREE_TRACE_ZONE_END(z0);
  IREE_TRACE_APP_EXIT(exit_code);
  return exit_code;
}

// Copyright 2026 RooflineAI GmbH
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Runs a compiled causal LM in a greedy token generation loop.
//
// The command line frontend of the LLM inference engine in llm/engine.h, which
// documents the prefill/decode functions the module must export. The prompt is
// tokenized with a HuggingFace tokenizer.json (llm/tokenizer.h), and each
// generated token is detokenized and printed to stdout as it is produced.
//
// Example:
//   iree-run-llm --device=hexagon --module=model.vmfb \
//       --tokenizer=tokenizer.json --prompt="The capital of France is" \
//       --max_new_tokens=32
//
// A module whose prefill takes more inputs than the prompt, e.g. a VLM's
// language model taking image features, gets them with --prefill_input, or
// from a vision function the module exports next to prefill and decode
// (llm/vlm.h). In one run, iree-run-llm then calls the vision function on
// --vision_input, passes its results to prefill on the device, and decodes:
//   iree-run-llm --device=hexagon --module=model.vmfb \
//       --vision_function=vision --vision_input=@image.npy \
//       --tokenizer=tokenizer.json --prompt_file=prompt.txt
// The prompt must already hold the image's tokens (e.g. Qwen3-VL's
// <|image_pad|> once per image embedding).

#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "iree/base/api.h"
#include "iree/base/tooling/flags.h"
#include "iree/hal/api.h"
#include "iree/hal/replay/recorder.h"
#include "iree/io/file_contents.h"
#include "iree/io/stdio_stream.h"
#include "iree/modules/hal/types.h"
#include "iree/tokenizer/tokenizer.h"
#include "iree/tooling/context_util.h"
#include "iree/tooling/function_io.h"
#include "iree/tooling/function_util.h"
#include "iree/tooling/numpy_io.h"
#include "iree/vm/api.h"
#include "llm/engine.h"
#include "llm/tokenizer.h"
#include "llm/vlm.h"

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
IREE_FLAG(int32_t, max_seq_len, IREE_LLM_DEFAULT_MAX_SEQ_LEN,
          "Context length the module was exported for (KvCacheLM's "
          "max_seq_len): the padded prompt and every decode position must "
          "fit in it.");
IREE_FLAG(int32_t, block_seq_stride, IREE_LLM_DEFAULT_BLOCK_SEQ_STRIDE,
          "The prefill sequence length must be a multiple of this (KvCacheLM's "
          "block_seq_stride).");
IREE_FLAG(int32_t, min_prefill_blocks, IREE_LLM_DEFAULT_MIN_PREFILL_BLOCKS,
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
IREE_FLAG(string, prefill_logits_output, "",
          "File to write prefill's logits to, as .npy: [1, P, vocab] for all "
          "P padded prompt positions.");
IREE_FLAG(string, prefill_function, IREE_LLM_DEFAULT_PREFILL_FUNCTION,
          "Name of the exported prefill function.");
IREE_FLAG(string, decode_function, IREE_LLM_DEFAULT_DECODE_FUNCTION,
          "Name of the exported decode function.");
IREE_FLAG(string, vision_function, "",
          "Name of an exported function of the model's module (e.g. vision) "
          "to run once before prefill, on the --vision_input values. Its "
          "results are passed to prefill after input_length, before any "
          "--prefill_input.");
IREE_FLAG_LIST(string, vision_input,
               "Input of --vision_function, as iree-run-module's --input "
               "(e.g. @image.npy); repeatable, in order.");
IREE_FLAG_LIST(string, vision_output,
               "Where to write a --vision_function result, as "
               "iree-run-module's --output (e.g. @vision_output0.npy); "
               "repeatable, one per result, in order. Only outputs them.");
IREE_FLAG_LIST(string, prefill_input,
               "Value prefill takes after input_length (and after the "
               "--vision_function results), as iree-run-module's --input "
               "(e.g. @image_embeds.npy); repeatable, in order.");

//===----------------------------------------------------------------------===//
// Setup from flags
//===----------------------------------------------------------------------===//

// What IREE's tooling creates from the --module= and --device= flags, and the
// model runs in. Outlives the model.
typedef struct iree_run_llm_runtime_t {
  iree_vm_context_t *context;
  // Runs --vision_function of the model's module; NULL without it.
  iree_llm_vision_t *vision;
  iree_hal_device_list_t *device_list;
  // The lead device's allocator, which identifies the lead device.
  iree_hal_allocator_t *device_allocator;
  // Set with --device_replay_output.
  iree_hal_replay_recorder_t *replay_recorder;
} iree_run_llm_runtime_t;

static void iree_run_llm_runtime_deinitialize(iree_run_llm_runtime_t *runtime) {
  iree_llm_vision_free(runtime->vision);
  iree_vm_context_release(runtime->context);
  if (runtime->replay_recorder) {
    iree_status_ignore(
        iree_hal_replay_recorder_close(runtime->replay_recorder));
    iree_hal_replay_recorder_release(runtime->replay_recorder);
  }
  iree_hal_allocator_release(runtime->device_allocator);
  iree_hal_device_list_free(runtime->device_list);
  memset(runtime, 0, sizeof(*runtime));
}

// Returns the device of |runtime| that its allocator belongs to: the one the
// tooling chose as the lead device (the first unless flags say otherwise).
static iree_hal_device_t *
iree_run_llm_lead_device(const iree_run_llm_runtime_t *runtime) {
  if (!runtime->device_list)
    return NULL;
  for (iree_host_size_t i = 0; i < runtime->device_list->count; ++i) {
    iree_hal_device_t *device =
        iree_hal_device_list_at(runtime->device_list, i);
    if (iree_hal_device_allocator(device) == runtime->device_allocator)
      return device;
  }
  return iree_hal_device_list_at(runtime->device_list, 0);
}

// Loads the --module= modules, creates a context for them on the --device=
// device(s) in |runtime|, and the model on it. The last module is the model's.
static iree_status_t iree_run_llm_create_model(iree_vm_instance_t *instance,
                                               iree_allocator_t host_allocator,
                                               iree_run_llm_runtime_t *runtime,
                                               iree_llm_model_t **out_model) {
  if (FLAG_block_seq_stride <= 0 || FLAG_min_prefill_blocks <= 0 ||
      FLAG_max_seq_len <= 0 || FLAG_max_new_tokens < 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "--block_seq_stride, --min_prefill_blocks and "
                            "--max_seq_len must be positive, and "
                            "--max_new_tokens not negative");
  }
  if (FLAG_vision_function[0] == '\0' &&
      (FLAG_vision_input_list().count > 0 ||
       FLAG_vision_output_list().count > 0)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "--vision_input and --vision_output need --vision_function");
  }
  iree_llm_model_options_t options;
  iree_llm_model_options_initialize(&options);
  options.max_seq_len = (iree_host_size_t)FLAG_max_seq_len;
  options.block_seq_stride = (iree_host_size_t)FLAG_block_seq_stride;
  options.min_prefill_blocks = (iree_host_size_t)FLAG_min_prefill_blocks;
  options.prefill_function = iree_make_cstring_view(FLAG_prefill_function);
  options.decode_function = iree_make_cstring_view(FLAG_decode_function);

  iree_tooling_module_list_t module_list;
  iree_tooling_module_list_initialize(&module_list);
  IREE_RETURN_IF_ERROR(iree_tooling_load_modules_from_flags(
                           instance, host_allocator, &module_list),
                       "loading modules");
  iree_status_t status = iree_ok_status();
  if (module_list.count == 0) {
    status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "no module given; use --module=model.vmfb");
  }
  if (iree_status_is_ok(status)) {
    status = iree_status_annotate_f(
        iree_tooling_create_context_from_flags(
            instance, module_list.count, module_list.values,
            /*default_device_uri=*/iree_string_view_empty(), host_allocator,
            &runtime->context, &runtime->device_list,
            &runtime->device_allocator, &runtime->replay_recorder),
        "creating VM context");
  }
  iree_hal_device_t *lead_device = NULL;
  if (iree_status_is_ok(status)) {
    lead_device = iree_run_llm_lead_device(runtime);
    if (!lead_device) {
      status = iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                                "the module does not use a HAL device");
    }
  }
  iree_vm_module_t *main_module =
      module_list.count > 0 ? module_list.values[module_list.count - 1] : NULL;
  if (iree_status_is_ok(status) && FLAG_vision_function[0] != '\0') {
    status = iree_status_annotate_f(
        iree_llm_vision_create(runtime->context, lead_device, main_module,
                               iree_make_cstring_view(FLAG_vision_function),
                               host_allocator, &runtime->vision),
        "--vision_function=%s", FLAG_vision_function);
  }
  if (iree_status_is_ok(status)) {
    status = iree_llm_model_create(runtime->context, lead_device, main_module,
                                   &options, host_allocator, out_model);
  }
  iree_tooling_module_list_reset(&module_list);
  return status;
}

static iree_status_t
iree_run_llm_load_tokenizer(iree_allocator_t host_allocator,
                            iree_tokenizer_t **out_tokenizer) {
  if (FLAG_tokenizer[0] == '\0') {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "missing --tokenizer=tokenizer.json");
  }
  return iree_llm_tokenizer_load_file(iree_make_cstring_view(FLAG_tokenizer),
                                      host_allocator, out_tokenizer);
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
  iree_status_t status =
      iree_llm_tokenizer_encode(tokenizer, text, FLAG_add_special_tokens,
                                host_allocator, out_tokens, out_count);
  if (iree_status_is_ok(status) && *out_count == 0) {
    status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "the prompt has no tokens; pass --prompt");
  }
  iree_io_file_contents_free(contents);
  return status;
}

// At most this many --stop_token_id flags.
#define IREE_RUN_LLM_MAX_STOP_TOKENS 16

// Parses --stop_token_id, defaulting to the tokenizer's EOS token, into
// |out_values|; none with --ignore_eos.
static iree_status_t
iree_run_llm_parse_stop_tokens(const iree_tokenizer_t *tokenizer,
                               int64_t *out_values,
                               iree_host_size_t *out_count) {
  *out_count = 0;
  if (FLAG_ignore_eos)
    return iree_ok_status();
  const iree_flag_string_list_t list = FLAG_stop_token_id_list();
  if (list.count > IREE_RUN_LLM_MAX_STOP_TOKENS) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "at most %d --stop_token_id flags",
                            IREE_RUN_LLM_MAX_STOP_TOKENS);
  }
  for (iree_host_size_t i = 0; i < list.count; ++i) {
    int64_t value = 0;
    if (!iree_string_view_atoi_int64(list.values[i], &value)) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "--stop_token_id='%.*s' is not an integer",
                              (int)list.values[i].size, list.values[i].data);
    }
    out_values[(*out_count)++] = value;
  }
  if (*out_count == 0) {
    int64_t eos = iree_llm_tokenizer_eos_token(tokenizer);
    if (eos >= 0)
      out_values[(*out_count)++] = eos;
  }
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Prefill inputs
//===----------------------------------------------------------------------===//

// At most this many --prefill_input flags.
#define IREE_RUN_LLM_MAX_PREFILL_INPUTS 32

// Appends every value of |values| to |list|.
static iree_status_t iree_run_llm_append_all(iree_vm_list_t *values,
                                             iree_vm_list_t *list) {
  for (iree_host_size_t i = 0; i < iree_vm_list_size(values); ++i) {
    iree_vm_variant_t value = iree_vm_variant_empty();
    IREE_RETURN_IF_ERROR(iree_vm_list_get_variant_assign(values, i, &value));
    IREE_RETURN_IF_ERROR(iree_vm_list_push_variant_retain(list, &value));
  }
  return iree_ok_status();
}

// Copies the --vision_function |outputs| to the host and writes them to the
// --vision_output targets. A shallow clone is copied, so |outputs| keep their
// device buffers for prefill.
static iree_status_t
iree_run_llm_write_vision_outputs(iree_hal_device_t *device,
                                  iree_vm_list_t *outputs,
                                  iree_allocator_t host_allocator) {
  iree_vm_list_t *copy = NULL;
  IREE_RETURN_IF_ERROR(iree_vm_list_clone(outputs, host_allocator, &copy));
  const iree_hal_buffer_params_t host_params = {
      .usage = IREE_HAL_BUFFER_USAGE_TRANSFER | IREE_HAL_BUFFER_USAGE_MAPPING,
      .access = IREE_HAL_MEMORY_ACCESS_ALL,
      .type =
          IREE_HAL_MEMORY_TYPE_HOST_LOCAL | IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE,
      .queue_affinity = IREE_HAL_QUEUE_AFFINITY_ANY,
  };
  iree_status_t status = iree_status_annotate(
      iree_tooling_transfer_variants(copy, device,
                                     iree_hal_device_allocator(device),
                                     host_params, /*wait_fence=*/NULL,
                                     /*signal_fence=*/NULL),
      IREE_SV("copying the --vision_function results to the host"));
  if (iree_status_is_ok(status)) {
    status = iree_status_annotate(
        iree_tooling_write_variants(copy, FLAG_vision_output_list(),
                                    /*max_element_count=*/0,
                                    /*default_stream=*/NULL, host_allocator),
        IREE_SV("writing --vision_output"));
  }
  iree_vm_list_release(copy);
  return status;
}

// Runs --vision_function on the --vision_input values. Sets |out_results| to
// its results, as they are: on the device, for prefill to take. Writes them to
// the --vision_output targets, if any. Sets |out_duration| to the time the
// call took, without that.
static iree_status_t iree_run_llm_run_vision(iree_run_llm_runtime_t *runtime,
                                             iree_hal_device_t *device,
                                             iree_allocator_t host_allocator,
                                             iree_vm_list_t **out_results,
                                             iree_duration_t *out_duration) {
  *out_results = NULL;
  iree_string_view_t arguments_cconv = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(
      iree_llm_vision_arguments_cconv(runtime->vision, &arguments_cconv));

  iree_vm_list_t *inputs = NULL;
  IREE_RETURN_IF_ERROR(iree_tooling_parse_variants(
                           arguments_cconv, FLAG_vision_input_list(), device,
                           runtime->device_allocator, host_allocator, &inputs),
                       "parsing --vision_input");
  iree_vm_list_t *outputs = NULL;
  iree_status_t status = iree_status_annotate_f(
      iree_llm_vision_run(runtime->vision, inputs, &outputs, out_duration),
      "running --vision_function=%s", FLAG_vision_function);
  if (iree_status_is_ok(status) && FLAG_vision_output_list().count > 0)
    status = iree_run_llm_write_vision_outputs(device, outputs, host_allocator);
  if (iree_status_is_ok(status)) {
    *out_results = outputs;
  } else {
    iree_vm_list_release(outputs);
  }
  iree_vm_list_release(inputs);
  return status;
}

// Parses the --prefill_input values and appends them to |list|.
static iree_status_t iree_run_llm_append_prefill_inputs(
    iree_run_llm_runtime_t *runtime, iree_hal_device_t *device,
    iree_allocator_t host_allocator, iree_vm_list_t *list) {
  const iree_flag_string_list_t specs = FLAG_prefill_input_list();
  if (specs.count > IREE_RUN_LLM_MAX_PREFILL_INPUTS) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "at most %d --prefill_input flags",
                            IREE_RUN_LLM_MAX_PREFILL_INPUTS);
  }
  // Every value is a reference type (a tensor), as --input parses it.
  char cconv_storage[IREE_RUN_LLM_MAX_PREFILL_INPUTS];
  memset(cconv_storage, 'r', specs.count);
  iree_vm_list_t *values = NULL;
  IREE_RETURN_IF_ERROR(iree_tooling_parse_variants(
                           iree_make_string_view(cconv_storage, specs.count),
                           specs, device, runtime->device_allocator,
                           host_allocator, &values),
                       "parsing --prefill_input");
  iree_status_t status = iree_run_llm_append_all(values, list);
  iree_vm_list_release(values);
  return status;
}

//===----------------------------------------------------------------------===//
// Generation
//===----------------------------------------------------------------------===//

static void iree_run_llm_print_ids(const char *label, const int64_t *tokens,
                                   iree_host_size_t count) {
  fprintf(stderr, "%s:", label);
  for (iree_host_size_t i = 0; i < count; ++i) {
    fprintf(stderr, "%s%" PRId64, i ? "," : " ", tokens[i]);
  }
  fputc('\n', stderr);
}

static iree_status_t iree_run_llm_write_text(void *user_data,
                                             iree_string_view_t text) {
  fwrite(text.data, 1, text.size, stdout);
  return iree_ok_status();
}

// Receives the generated tokens: records them and prints their text.
typedef struct iree_run_llm_output_t {
  iree_llm_detokenizer_t detokenizer;
  int64_t *tokens;
  iree_host_size_t count;
  iree_host_size_t capacity;
} iree_run_llm_output_t;

static iree_status_t iree_run_llm_on_token(void *user_data, int64_t token,
                                           bool *out_stop) {
  iree_run_llm_output_t *output = (iree_run_llm_output_t *)user_data;
  if (output->count < output->capacity)
    output->tokens[output->count++] = token;
  iree_status_t status = iree_llm_detokenizer_feed(&output->detokenizer, token);
  fflush(stdout);
  return status;
}

// Copies |logits| from |device| to the host, as iree-run-module does with its
// outputs, and writes them to the .npy file --prefill_logits_output.
static iree_status_t
iree_run_llm_write_logits(iree_hal_device_t *device,
                          iree_hal_buffer_view_t *logits,
                          iree_allocator_t host_allocator) {
  iree_vm_list_t *list = NULL;
  IREE_RETURN_IF_ERROR(iree_vm_list_create(iree_vm_make_undefined_type_def(), 1,
                                           host_allocator, &list));
  iree_vm_ref_t ref = iree_hal_buffer_view_retain_ref(logits);
  iree_status_t status = iree_vm_list_push_ref_move(list, &ref);
  if (iree_status_is_ok(status)) {
    const iree_hal_buffer_params_t host_params = {
        .usage = IREE_HAL_BUFFER_USAGE_TRANSFER | IREE_HAL_BUFFER_USAGE_MAPPING,
        .access = IREE_HAL_MEMORY_ACCESS_ALL,
        .type = IREE_HAL_MEMORY_TYPE_HOST_LOCAL |
                IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE,
        .queue_affinity = IREE_HAL_QUEUE_AFFINITY_ANY,
    };
    status = iree_status_annotate(
        iree_tooling_transfer_variants(
            list, device, iree_hal_device_allocator(device), host_params,
            /*wait_fence=*/NULL, /*signal_fence=*/NULL),
        IREE_SV("copying prefill's logits to the host"));
  }
  iree_io_stream_t *stream = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_status_annotate_f(
        iree_io_stdio_stream_open(
            IREE_IO_STDIO_STREAM_MODE_WRITE | IREE_IO_STDIO_STREAM_MODE_DISCARD,
            iree_make_cstring_view(FLAG_prefill_logits_output), host_allocator,
            &stream),
        "opening '%s'", FLAG_prefill_logits_output);
  }
  if (iree_status_is_ok(status)) {
    status = iree_status_annotate_f(
        iree_numpy_npy_save_ndarray(
            stream, IREE_NUMPY_NPY_SAVE_OPTION_DEFAULT,
            iree_vm_list_get_buffer_view_assign(list, 0), host_allocator),
        "writing prefill's logits to '%s'", FLAG_prefill_logits_output);
  }
  iree_io_stream_release(stream);
  iree_vm_list_release(list);
  return status;
}

// |vision_ms| is negative without --vision_function.
static void iree_run_llm_print_stats(double encode_ms, double vision_ms,
                                     const iree_llm_generate_stats_t *stats) {
  const double prefill_ms = stats->prefill_duration / 1e6;
  const double decode_ms = stats->decode_duration / 1e6;
  const iree_host_size_t steps = stats->decode_steps;
  fputs("\n--- iree-run-llm ---\n", stderr);
  if (vision_ms >= 0)
    fprintf(stderr, "vision:  %.2f ms\n", vision_ms);
  fprintf(stderr,
          "prompt:  %zu tokens (padded to %zu), tokenized in %.2f ms\n"
          "prefill: %.2f ms (%.1f tokens/s)\n"
          "decode:  %zu steps in %.2f ms (%.2f ms/token, %.2f tokens/s)\n"
          "generated %zu tokens, stopped: %s\n",
          stats->prompt_length, stats->padded_length, encode_ms, prefill_ms,
          prefill_ms > 0 ? stats->prompt_length / (prefill_ms / 1e3) : 0.0,
          steps, decode_ms, steps ? decode_ms / steps : 0.0,
          decode_ms > 0 ? steps / (decode_ms / 1e3) : 0.0,
          stats->generated_count,
          iree_llm_stop_reason_string(stats->stop_reason));
}

// |device| is the model's lead device, which prefill's logits are on.
static iree_status_t iree_run_llm_generate(iree_llm_model_t *model,
                                           iree_run_llm_runtime_t *runtime,
                                           iree_hal_device_t *device,
                                           const iree_tokenizer_t *tokenizer,
                                           iree_allocator_t host_allocator) {
  iree_llm_generate_params_t params = {
      .max_new_tokens = (iree_host_size_t)FLAG_max_new_tokens,
      .pad_token = FLAG_pad_token_id >= 0
                       ? FLAG_pad_token_id
                       : iree_llm_tokenizer_pad_token(tokenizer),
  };
  int64_t stop_tokens[IREE_RUN_LLM_MAX_STOP_TOKENS];
  IREE_RETURN_IF_ERROR(iree_run_llm_parse_stop_tokens(
      tokenizer, stop_tokens, &params.stop_token_count));
  params.stop_tokens = stop_tokens;

  const iree_time_t encode_start = iree_time_now();
  int64_t *prompt = NULL;
  iree_host_size_t prompt_length = 0;
  IREE_RETURN_IF_ERROR(iree_run_llm_encode_prompt(tokenizer, host_allocator,
                                                  &prompt, &prompt_length));
  const iree_time_t encode_end = iree_time_now();
  if (FLAG_print_token_ids) {
    iree_run_llm_print_ids("prompt token ids", prompt, prompt_length);
  }

  iree_run_llm_output_t output;
  memset(&output, 0, sizeof(output));
  output.capacity = params.max_new_tokens;
  iree_status_t status = iree_allocator_malloc_array(
      host_allocator, iree_max(output.capacity, 1), sizeof(*output.tokens),
      (void **)&output.tokens);
  if (iree_status_is_ok(status)) {
    const iree_llm_text_callback_t write_text = {iree_run_llm_write_text, NULL};
    status = iree_llm_detokenizer_initialize(
        tokenizer, /*skip_special_tokens=*/!FLAG_print_special_tokens,
        write_text, host_allocator, &output.detokenizer);
  }

  // What prefill takes after input_length: the --vision_function results
  // (image features), then the --prefill_input values; NULL if none. The
  // vision function runs here, before the timed generation.
  iree_vm_list_t *prefill_inputs = NULL;
  iree_duration_t vision_duration = -1;
  if (iree_status_is_ok(status) && runtime->vision) {
    status = iree_run_llm_run_vision(runtime, device, host_allocator,
                                     &prefill_inputs, &vision_duration);
  }
  if (iree_status_is_ok(status) && FLAG_prefill_input_list().count > 0) {
    if (!prefill_inputs) {
      status = iree_vm_list_create(iree_vm_make_undefined_type_def(),
                                   FLAG_prefill_input_list().count,
                                   host_allocator, &prefill_inputs);
    }
    if (iree_status_is_ok(status)) {
      status = iree_run_llm_append_prefill_inputs(
          runtime, device, host_allocator, prefill_inputs);
    }
  }
  params.prefill_inputs = prefill_inputs;

  iree_llm_generate_stats_t stats;
  memset(&stats, 0, sizeof(stats));
  iree_hal_buffer_view_t *prefill_logits = NULL;
  if (iree_status_is_ok(status)) {
    const iree_llm_token_callback_t on_token = {iree_run_llm_on_token, &output};
    status = iree_llm_generate(
        model, prompt, prompt_length, &params, on_token, &stats,
        FLAG_prefill_logits_output[0] != '\0' ? &prefill_logits : NULL);
  }
  if (iree_status_is_ok(status)) {
    status = iree_llm_detokenizer_finish(&output.detokenizer);
    fputc('\n', stdout);
    fflush(stdout);
  }

  if (iree_status_is_ok(status) && FLAG_print_token_ids) {
    iree_run_llm_print_ids("generated token ids", output.tokens, output.count);
  }
  if (iree_status_is_ok(status) && prefill_logits) {
    status = iree_run_llm_write_logits(device, prefill_logits, host_allocator);
  }
  if (iree_status_is_ok(status) && FLAG_print_stats) {
    iree_run_llm_print_stats((encode_end - encode_start) / 1e6,
                             vision_duration >= 0 ? vision_duration / 1e6 : -1,
                             &stats);
  }

  iree_vm_list_release(prefill_inputs);
  iree_hal_buffer_view_release(prefill_logits);
  iree_llm_detokenizer_deinitialize(&output.detokenizer);
  iree_allocator_free(host_allocator, output.tokens);
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
      "the shapes the module was exported with.\n"
      "\n"
      "A VLM takes its image features with --prefill_input, or runs its\n"
      "vision encoder on the device first with --vision_function:\n"
      "  iree-run-llm --device=hexagon --module=model.vmfb \\\n"
      "      --vision_function=vision \\\n"
      "      --vision_input=@image.npy --tokenizer=tokenizer.json \\\n"
      "      --prompt_file=prompt.txt\n");
  iree_flags_parse_checked(IREE_FLAGS_PARSE_MODE_DEFAULT, &argc, &argv);

  iree_allocator_t host_allocator = iree_allocator_system();
  iree_tokenizer_t *tokenizer = NULL;
  iree_status_t status =
      iree_run_llm_load_tokenizer(host_allocator, &tokenizer);

  iree_vm_instance_t *instance = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_tooling_create_instance(host_allocator, &instance);
  }
  iree_run_llm_runtime_t runtime;
  memset(&runtime, 0, sizeof(runtime));
  iree_llm_model_t *model = NULL;
  if (iree_status_is_ok(status)) {
    status =
        iree_run_llm_create_model(instance, host_allocator, &runtime, &model);
  }
  if (iree_status_is_ok(status)) {
    status = iree_run_llm_generate(model, &runtime,
                                   iree_run_llm_lead_device(&runtime),
                                   tokenizer, host_allocator);
  }

  iree_llm_model_free(model);
  iree_run_llm_runtime_deinitialize(&runtime);
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

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
// Further prompts continue the sequence, as further turns of a conversation:
// each is prefilled after the tokens so far, and generated from in turn.
//
// Flags that set how to generate, e.g. --max_new_tokens, apply to the prompts
// after them on the command line, so that each prompt can have its own; an
// output for a prompt, --prefill_logits_output, applies to the next one only.
//
// Example:
//   iree-run-llm --device=hexagon --module=model.vmfb \
//       --tokenizer=tokenizer.json \
//       --max_new_tokens=32 --prompt="The capital of France is" \
//       --max_new_tokens=16 --prompt=" The capital of Germany is"

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
#include "iree/tooling/function_util.h"
#include "iree/tooling/numpy_io.h"
#include "iree/vm/api.h"
#include "llm/engine.h"
#include "llm/tokenizer.h"

//===----------------------------------------------------------------------===//
// Flags
//===----------------------------------------------------------------------===//

IREE_FLAG(string, tokenizer, "", "Path to a HuggingFace tokenizer.json.");
IREE_FLAG(bool, add_special_tokens, true,
          "Add the tokenizer's special tokens (e.g. BOS) to the first prompt, "
          "as HuggingFace's tokenizer(prompt) does by default. Further "
          "prompts never get them.");
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
IREE_FLAG(string, prefill_function, IREE_LLM_DEFAULT_PREFILL_FUNCTION,
          "Name of the exported prefill function.");
IREE_FLAG(string, decode_function, IREE_LLM_DEFAULT_DECODE_FUNCTION,
          "Name of the exported decode function.");

// The prompts, from --prompt and --prompt_file in command line order, each
// with the flags before it: the --max_new_tokens setting, which also applies to
// the prompts after it, and its own --prefill_logits_output.

// Default of --max_new_tokens.
#define IREE_RUN_LLM_DEFAULT_MAX_NEW_TOKENS 64

typedef struct iree_run_llm_prompt_t {
  // The text of a --prompt, or the path of a --prompt_file.
  iree_string_view_t value;
  bool is_file;
  // The --max_new_tokens before it.
  iree_host_size_t max_new_tokens;
  // The --prefill_logits_output right before it, if any; empty otherwise.
  iree_string_view_t prefill_logits_output;
} iree_run_llm_prompt_t;

typedef struct iree_run_llm_prompts_t {
  // The settings for the next prompt.
  iree_host_size_t max_new_tokens;
  // The --prefill_logits_output for the next prompt only, if any.
  iree_string_view_t prefill_logits_output;
  // Whether a --max_new_tokens came after the last prompt, which no prompt
  // uses then; an error, as is a --prefill_logits_output there.
  bool max_new_tokens_unused;
  iree_host_size_t count;
  iree_host_size_t capacity;
  iree_run_llm_prompt_t *values;
} iree_run_llm_prompts_t;

static iree_run_llm_prompts_t iree_run_llm_prompts = {
    .max_new_tokens = IREE_RUN_LLM_DEFAULT_MAX_NEW_TOKENS,
};

static iree_status_t
iree_run_llm_parse_max_new_tokens(iree_string_view_t flag_name, void *storage,
                                  iree_string_view_t value) {
  iree_run_llm_prompts_t *prompts = (iree_run_llm_prompts_t *)storage;
  int32_t max_new_tokens = 0;
  if (!iree_string_view_atoi_int32(value, &max_new_tokens) ||
      max_new_tokens < 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "--max_new_tokens='%.*s' is not a non-negative "
                            "integer",
                            (int)value.size, value.data);
  }
  prompts->max_new_tokens = (iree_host_size_t)max_new_tokens;
  prompts->max_new_tokens_unused = true;
  return iree_ok_status();
}

static void iree_run_llm_print_max_new_tokens(iree_string_view_t flag_name,
                                              void *storage, FILE *file) {
  const iree_run_llm_prompts_t *prompts = (iree_run_llm_prompts_t *)storage;
  fprintf(file, "--%.*s=%zu\n", (int)flag_name.size, flag_name.data,
          prompts->max_new_tokens);
}

IREE_FLAG_CALLBACK(iree_run_llm_parse_max_new_tokens,
                   iree_run_llm_print_max_new_tokens, &iree_run_llm_prompts,
                   max_new_tokens,
                   "Maximum number of tokens to generate after each prompt "
                   "that follows on the command line, up to the next "
                   "--max_new_tokens (default 64).");

static iree_status_t iree_run_llm_parse_prefill_logits_output(
    iree_string_view_t flag_name, void *storage, iree_string_view_t value) {
  iree_run_llm_prompts_t *prompts = (iree_run_llm_prompts_t *)storage;
  if (!iree_string_view_is_empty(prompts->prefill_logits_output)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "--prefill_logits_output='%.*s' follows another "
                            "one for the same prompt",
                            (int)value.size, value.data);
  }
  prompts->prefill_logits_output = value;
  return iree_ok_status();
}

static void
iree_run_llm_print_prefill_logits_output(iree_string_view_t flag_name,
                                         void *storage, FILE *file) {
  const iree_run_llm_prompts_t *prompts = (iree_run_llm_prompts_t *)storage;
  bool printed = false;
  for (iree_host_size_t i = 0; i < prompts->count; ++i) {
    const iree_string_view_t path = prompts->values[i].prefill_logits_output;
    if (iree_string_view_is_empty(path))
      continue;
    fprintf(file, "--%.*s=%.*s\n", (int)flag_name.size, flag_name.data,
            (int)path.size, path.data);
    printed = true;
  }
  if (!printed) {
    fprintf(file, "# --%.*s=...\n", (int)flag_name.size, flag_name.data);
  }
}

IREE_FLAG_CALLBACK(iree_run_llm_parse_prefill_logits_output,
                   iree_run_llm_print_prefill_logits_output,
                   &iree_run_llm_prompts, prefill_logits_output,
                   "File to write prefill's logits of the prompt that follows "
                   "on the command line to, as .npy: [1, P, vocab] for all P "
                   "padded positions, which for a further prompt start with "
                   "the token generated before it. Applies to that one "
                   "prompt only.");

// Appends a prompt with the current settings.
static iree_status_t iree_run_llm_append_prompt(iree_run_llm_prompts_t *prompts,
                                                iree_string_view_t value,
                                                bool is_file) {
  if (prompts->count == prompts->capacity) {
    IREE_RETURN_IF_ERROR(iree_allocator_grow_array(
        iree_allocator_system(), /*minimum_capacity=*/4,
        sizeof(prompts->values[0]), &prompts->capacity,
        (void **)&prompts->values));
  }
  prompts->values[prompts->count++] = (iree_run_llm_prompt_t){
      .value = value,
      .is_file = is_file,
      .max_new_tokens = prompts->max_new_tokens,
      .prefill_logits_output = prompts->prefill_logits_output,
  };
  prompts->prefill_logits_output = iree_string_view_empty();
  prompts->max_new_tokens_unused = false;
  return iree_ok_status();
}

static iree_status_t iree_run_llm_parse_prompt(iree_string_view_t flag_name,
                                               void *storage,
                                               iree_string_view_t value) {
  return iree_run_llm_append_prompt((iree_run_llm_prompts_t *)storage, value,
                                    /*is_file=*/false);
}

static iree_status_t
iree_run_llm_parse_prompt_file(iree_string_view_t flag_name, void *storage,
                               iree_string_view_t value) {
  return iree_run_llm_append_prompt((iree_run_llm_prompts_t *)storage, value,
                                    /*is_file=*/true);
}

// Prints the --prompt or the --prompt_file flags, as |is_file| says, or a
// placeholder if there are none, as IREE's list flags do.
static void iree_run_llm_print_prompts(iree_string_view_t flag_name,
                                       const iree_run_llm_prompts_t *prompts,
                                       bool is_file, FILE *file) {
  bool printed = false;
  for (iree_host_size_t i = 0; i < prompts->count; ++i) {
    const iree_run_llm_prompt_t *prompt = &prompts->values[i];
    if (prompt->is_file != is_file)
      continue;
    fprintf(file, "--%.*s=%.*s\n", (int)flag_name.size, flag_name.data,
            (int)prompt->value.size, prompt->value.data);
    printed = true;
  }
  if (!printed) {
    fprintf(file, "# --%.*s=...\n", (int)flag_name.size, flag_name.data);
  }
}

static void iree_run_llm_print_prompt(iree_string_view_t flag_name,
                                      void *storage, FILE *file) {
  iree_run_llm_print_prompts(flag_name, (iree_run_llm_prompts_t *)storage,
                             /*is_file=*/false, file);
}

static void iree_run_llm_print_prompt_file(iree_string_view_t flag_name,
                                           void *storage, FILE *file) {
  iree_run_llm_print_prompts(flag_name, (iree_run_llm_prompts_t *)storage,
                             /*is_file=*/true, file);
}

IREE_FLAG_CALLBACK(iree_run_llm_parse_prompt, iree_run_llm_print_prompt,
                   &iree_run_llm_prompts, prompt,
                   "Prompt text, used as is: apply any chat template "
                   "beforehand. Repeatable: each further prompt continues the "
                   "sequence after the tokens generated so far, as a further "
                   "turn of a conversation.");
IREE_FLAG_CALLBACK(iree_run_llm_parse_prompt_file,
                   iree_run_llm_print_prompt_file, &iree_run_llm_prompts,
                   prompt_file,
                   "File to read a prompt from, instead of giving its text "
                   "with --prompt; both can be mixed, in any order.");

//===----------------------------------------------------------------------===//
// Setup from flags
//===----------------------------------------------------------------------===//

// What IREE's tooling creates from the --module= and --device= flags, and the
// model runs in. Outlives the model.
typedef struct iree_run_llm_runtime_t {
  iree_vm_context_t *context;
  iree_hal_device_list_t *device_list;
  // The lead device's allocator, which identifies the lead device.
  iree_hal_allocator_t *device_allocator;
  // Set with --device_replay_output.
  iree_hal_replay_recorder_t *replay_recorder;
} iree_run_llm_runtime_t;

static void iree_run_llm_runtime_deinitialize(iree_run_llm_runtime_t *runtime) {
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
      FLAG_max_seq_len <= 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "--block_seq_stride, --min_prefill_blocks and "
                            "--max_seq_len must be positive");
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
  if (iree_status_is_ok(status)) {
    status = iree_llm_model_create(runtime->context, lead_device,
                                   module_list.values[module_list.count - 1],
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

// Checks the prompts from --prompt and --prompt_file.
static iree_status_t iree_run_llm_check_prompts(void) {
  if (iree_run_llm_prompts.count == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "no prompt given; pass --prompt or --prompt_file");
  }
  if (iree_run_llm_prompts.max_new_tokens_unused) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "--max_new_tokens after the last prompt applies to "
                            "no prompt; put it before the prompts it is for");
  }
  if (!iree_string_view_is_empty(iree_run_llm_prompts.prefill_logits_output)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "--prefill_logits_output after the last prompt "
                            "applies to no prompt; put it before the prompt "
                            "it is for");
  }
  return iree_ok_status();
}

// Tokenizes |prompt|, number |index| from 0; only the first gets the
// tokenizer's special tokens, with --add_special_tokens. The caller frees
// |out_tokens| with |host_allocator|.
static iree_status_t iree_run_llm_encode_prompt(
    const iree_tokenizer_t *tokenizer, const iree_run_llm_prompt_t *prompt,
    iree_host_size_t index, iree_allocator_t host_allocator,
    int64_t **out_tokens, iree_host_size_t *out_count) {
  *out_tokens = NULL;
  *out_count = 0;

  iree_io_file_contents_t *contents = NULL;
  iree_string_view_t text = prompt->value;
  if (prompt->is_file) {
    IREE_RETURN_IF_ERROR(
        iree_io_file_contents_read(prompt->value, host_allocator, &contents),
        "reading '%.*s'", (int)prompt->value.size, prompt->value.data);
    text = iree_make_string_view((const char *)contents->const_buffer.data,
                                 contents->const_buffer.data_length);
  }
  iree_status_t status = iree_llm_tokenizer_encode(
      tokenizer, text, FLAG_add_special_tokens && index == 0, host_allocator,
      out_tokens, out_count);
  if (iree_status_is_ok(status) && *out_count == 0) {
    status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "prompt %zu has no tokens", index + 1);
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
// outputs, and writes them to the .npy file |path|.
static iree_status_t iree_run_llm_write_logits(
    iree_hal_device_t *device, iree_hal_buffer_view_t *logits,
    iree_string_view_t path, iree_allocator_t host_allocator) {
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
        iree_io_stdio_stream_open(IREE_IO_STDIO_STREAM_MODE_WRITE |
                                      IREE_IO_STDIO_STREAM_MODE_DISCARD,
                                  path, host_allocator, &stream),
        "opening '%.*s'", (int)path.size, path.data);
  }
  if (iree_status_is_ok(status)) {
    status = iree_status_annotate_f(
        iree_numpy_npy_save_ndarray(
            stream, IREE_NUMPY_NPY_SAVE_OPTION_DEFAULT,
            iree_vm_list_get_buffer_view_assign(list, 0), host_allocator),
        "writing prefill's logits to '%.*s'", (int)path.size, path.data);
  }
  iree_io_stream_release(stream);
  iree_vm_list_release(list);
  return status;
}

static void iree_run_llm_print_stats(iree_host_size_t index,
                                     iree_host_size_t count, double encode_ms,
                                     const iree_llm_generate_stats_t *stats) {
  const double prefill_ms = stats->prefill_duration / 1e6;
  const double decode_ms = stats->decode_duration / 1e6;
  const iree_host_size_t steps = stats->decode_steps;
  if (count > 1) {
    fprintf(stderr, "\n--- iree-run-llm: prompt %zu of %zu ---\n", index + 1,
            count);
  } else {
    fprintf(stderr, "\n--- iree-run-llm ---\n");
  }
  if (stats->start_position > 0) {
    fprintf(stderr,
            "prompt:  %zu tokens (prefilled after the last generated token, "
            "at positions %zu-%zu, padded to %zu), tokenized in %.2f ms\n",
            stats->prompt_length, stats->start_position,
            stats->start_position + stats->prefill_length - 1,
            stats->padded_length, encode_ms);
  } else {
    fprintf(stderr,
            "prompt:  %zu tokens (padded to %zu), tokenized in %.2f ms\n",
            stats->prompt_length, stats->padded_length, encode_ms);
  }
  fprintf(stderr,
          "prefill: %.2f ms (%.1f tokens/s)\n"
          "decode:  %zu steps in %.2f ms (%.2f ms/token, %.2f tokens/s)\n"
          "generated %zu tokens, stopped: %s\n",
          prefill_ms,
          prefill_ms > 0 ? stats->prefill_length / (prefill_ms / 1e3) : 0.0,
          steps, decode_ms, steps ? decode_ms / steps : 0.0,
          decode_ms > 0 ? steps / (decode_ms / 1e3) : 0.0,
          stats->generated_count,
          iree_llm_stop_reason_string(stats->stop_reason));
}

// Generates from prompt |index| of |count|, continuing the model's sequence.
// |device| is the model's lead device, which prefill's logits are on.
static iree_status_t
iree_run_llm_generate_turn(iree_llm_model_t *model, iree_hal_device_t *device,
                           const iree_tokenizer_t *tokenizer,
                           iree_llm_generate_params_t params,
                           iree_host_size_t index, iree_host_size_t count,
                           iree_allocator_t host_allocator) {
  const iree_run_llm_prompt_t *prompt_flag =
      &iree_run_llm_prompts.values[index];
  params.max_new_tokens = prompt_flag->max_new_tokens;
  const iree_string_view_t logits_output = prompt_flag->prefill_logits_output;

  const iree_time_t encode_start = iree_time_now();
  int64_t *prompt = NULL;
  iree_host_size_t prompt_length = 0;
  IREE_RETURN_IF_ERROR(iree_run_llm_encode_prompt(
      tokenizer, prompt_flag, index, host_allocator, &prompt, &prompt_length));
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

  iree_llm_generate_stats_t stats;
  memset(&stats, 0, sizeof(stats));
  iree_hal_buffer_view_t *prefill_logits = NULL;
  if (iree_status_is_ok(status)) {
    const iree_llm_token_callback_t on_token = {iree_run_llm_on_token, &output};
    status = iree_llm_generate(
        model, prompt, prompt_length, &params, on_token, &stats,
        iree_string_view_is_empty(logits_output) ? NULL : &prefill_logits);
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
    status = iree_run_llm_write_logits(device, prefill_logits, logits_output,
                                       host_allocator);
  }
  if (iree_status_is_ok(status) && FLAG_print_stats) {
    iree_run_llm_print_stats(index, count, (encode_end - encode_start) / 1e6,
                             &stats);
  }
  if (!iree_status_is_ok(status) && count > 1) {
    status = iree_status_annotate_f(status, "at prompt %zu of %zu", index + 1,
                                    count);
  }

  iree_hal_buffer_view_release(prefill_logits);
  iree_llm_detokenizer_deinitialize(&output.detokenizer);
  iree_allocator_free(host_allocator, output.tokens);
  iree_allocator_free(host_allocator, prompt);
  return status;
}

// Generates from each prompt in turn, as one sequence.
static iree_status_t iree_run_llm_generate(iree_llm_model_t *model,
                                           iree_hal_device_t *device,
                                           const iree_tokenizer_t *tokenizer,
                                           iree_allocator_t host_allocator) {
  IREE_RETURN_IF_ERROR(iree_run_llm_check_prompts());
  const iree_host_size_t count = iree_run_llm_prompts.count;
  iree_llm_generate_params_t params = {
      .pad_token = FLAG_pad_token_id >= 0
                       ? FLAG_pad_token_id
                       : iree_llm_tokenizer_pad_token(tokenizer),
  };
  int64_t stop_tokens[IREE_RUN_LLM_MAX_STOP_TOKENS];
  IREE_RETURN_IF_ERROR(iree_run_llm_parse_stop_tokens(
      tokenizer, stop_tokens, &params.stop_token_count));
  params.stop_tokens = stop_tokens;

  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0; i < count && iree_status_is_ok(status); ++i) {
    status = iree_run_llm_generate_turn(model, device, tokenizer, params, i,
                                        count, host_allocator);
  }
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
      "Each further --prompt continues the sequence, as a further turn of a\n"
      "conversation. --max_new_tokens applies to the prompts after it,\n"
      "--prefill_logits_output to the next prompt.\n"
      "\n"
      "Usage:\n"
      "  iree-run-llm --device=hexagon --module=model.vmfb \\\n"
      "      --tokenizer=tokenizer.json \\\n"
      "      --max_new_tokens=32 --prompt=\"Once upon a time\"\n"
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
  iree_run_llm_runtime_t runtime;
  memset(&runtime, 0, sizeof(runtime));
  iree_llm_model_t *model = NULL;
  if (iree_status_is_ok(status)) {
    status =
        iree_run_llm_create_model(instance, host_allocator, &runtime, &model);
  }
  if (iree_status_is_ok(status)) {
    status = iree_run_llm_generate(model, iree_run_llm_lead_device(&runtime),
                                   tokenizer, host_allocator);
  }

  iree_llm_model_free(model);
  iree_run_llm_runtime_deinitialize(&runtime);
  iree_vm_instance_release(instance);
  if (tokenizer)
    iree_tokenizer_free(tokenizer);
  iree_allocator_free(iree_allocator_system(), iree_run_llm_prompts.values);

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

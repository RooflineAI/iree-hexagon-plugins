// Copyright 2026 RooflineAI GmbH
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// LLM inference engine: runs a compiled causal LM token by token.
//
// The module is expected to export the two functions of
// integration_tests/models/kv_cache.py's KvCacheLM, which keeps its KV cache
// in module globals shared by both:
//
//   prefill(input_ids: i64[1, P], cache_position: i64[P],
//           input_length: i64[1])
//       -> (next_token: i64[1, 1], logits: [1, P, vocab])
//     Runs the prompt, padded to P, a multiple of block_seq_stride of at
//     least min_prefill_blocks blocks. input_length is the index of the last
//     real token, whose argmax is next_token.
//   decode(input_ids: i64[1, 1], cache_position: i64[pos + 1])
//       -> next_token: i64[1, 1]
//     Runs one token at position pos against the cache.
//
// prefill may take more arguments after input_length, which the caller passes
// as values (e.g. a VLM's image features, see
// integration_tests/models/qwen3_vl.py). The engine passes them through as
// they are; what they mean is between the caller and the module.
//
// Both run in one VM context, so prefill writes to a cache and the same one
// is used by decode uses. Sampling is greedy and is performed inside the
// module.
//
// The engine is supposed to be re-usable in different tools (iree-run-llm
// and e.g. some server). This means for instance that it does not parse any
// command line flags. Instead, everything the engine needs is passed in form
// of C data structures.
// That includes the VM context and its HAL device(s): the caller creates them,
// e.g. iree-run-llm with IREE's tooling from the --device= family of flags, and
// the engine only runs the module's functions in them.
//
// Thread safety: the KV cache lives in the module globals of the VM context,
// so a context holds one cache and runs one sequence at a time; a prefill
// starts a new sequence and discards the previous one. Callers that share a
// model across threads must serialize all calls on it, and must not create two
// models on one context or invoke its prefill/decode functions themselves.
// Models on separate contexts are independent.

#ifndef IREE_HEXAGON_PLUGINS_TOOLS_LLM_ENGINE_H_
#define IREE_HEXAGON_PLUGINS_TOOLS_LLM_ENGINE_H_

#include <stdbool.h>
#include <stdint.h>

#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/vm/api.h"

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

//===----------------------------------------------------------------------===//
// Model
//===----------------------------------------------------------------------===//

// KvCacheLM's defaults, as constants for e.g. command line flag defaults.
#define IREE_LLM_DEFAULT_MAX_SEQ_LEN 128
#define IREE_LLM_DEFAULT_BLOCK_SEQ_STRIDE 32
#define IREE_LLM_DEFAULT_MIN_PREFILL_BLOCKS 2
#define IREE_LLM_DEFAULT_PREFILL_FUNCTION "prefill"
#define IREE_LLM_DEFAULT_DECODE_FUNCTION "decode"

// How the module was exported; must match KvCacheLM's parameters.
typedef struct iree_llm_model_options_t {
  // Context length: the padded prompt and every decode position fit in it.
  iree_host_size_t max_seq_len;
  // The prefill sequence length is a multiple of this.
  iree_host_size_t block_seq_stride;
  // Minimum number of block_seq_stride blocks prefill accepts.
  iree_host_size_t min_prefill_blocks;
  // Names of the exported functions. Only used during creation.
  iree_string_view_t prefill_function;
  iree_string_view_t decode_function;
} iree_llm_model_options_t;

// Initializer of iree_llm_model_options_t with KvCacheLM's defaults, also for
// static storage:
//   iree_llm_model_options_t options = IREE_LLM_MODEL_OPTIONS_DEFAULTS;
#define IREE_LLM_MODEL_OPTIONS_DEFAULTS                                        \
  {                                                                            \
      .max_seq_len = IREE_LLM_DEFAULT_MAX_SEQ_LEN,                             \
      .block_seq_stride = IREE_LLM_DEFAULT_BLOCK_SEQ_STRIDE,                   \
      .min_prefill_blocks = IREE_LLM_DEFAULT_MIN_PREFILL_BLOCKS,               \
      .prefill_function = IREE_SVL(IREE_LLM_DEFAULT_PREFILL_FUNCTION),         \
      .decode_function = IREE_SVL(IREE_LLM_DEFAULT_DECODE_FUNCTION),           \
  }

// Sets |out_options| to IREE_LLM_MODEL_OPTIONS_DEFAULTS.
void iree_llm_model_options_initialize(iree_llm_model_options_t *out_options);

typedef struct iree_llm_model_t iree_llm_model_t;

// Creates a model that runs the prefill and decode functions of |main_module|
// in |context|, which must contain |main_module| and the modules it depends
// on (e.g. the HAL module).
//
// |lead_device| is the device the engine stages inputs on, with its
// allocator, and reads the generated tokens back from; it must be one of the
// context's HAL devices, and the module's functions must accept buffers from
// its allocator. With one device, that is simply the device.
//
// The model retains |context|, |lead_device| and |main_module|; the caller
// keeps its own references.
iree_status_t iree_llm_model_create(iree_vm_context_t *context,
                                    iree_hal_device_t *lead_device,
                                    iree_vm_module_t *main_module,
                                    const iree_llm_model_options_t *options,
                                    iree_allocator_t host_allocator,
                                    iree_llm_model_t **out_model);

// Frees |model|. NULL is a no-op.
void iree_llm_model_free(iree_llm_model_t *model);

const iree_llm_model_options_t *
iree_llm_model_options(const iree_llm_model_t *model);

// Returns the prefill sequence length a prompt of |prompt_length| tokens is
// padded to. It may exceed max_seq_len, in which case prefill rejects it.
iree_host_size_t
iree_llm_model_padded_prompt_length(const iree_llm_model_t *model,
                                    iree_host_size_t prompt_length);

// Returns the position the next decode runs at: the number of tokens in the
// KV cache. 0 before the first prefill.
iree_host_size_t iree_llm_model_position(const iree_llm_model_t *model);

// Starts a new sequence: runs prefill on |prompt|, padded with |pad_token|,
// and returns the token following it. Padding is never attended to, so any
// valid token ID works as |pad_token|.
//
// |extra_inputs|, if not NULL, holds the values prefill takes after
// input_length, in order; they must be usable on the lead device, e.g. the
// results of another function in the same context.
//
// If |out_logits| is not NULL, it receives prefill's logits, [1, P, vocab] for
// all P padded positions, as the module returned them: in device memory, which
// the caller may have to copy to the host to read them (e.g. with
// iree_tooling_transfer_variants). The caller releases them. Otherwise they
// are dropped.
iree_status_t
iree_llm_model_prefill(iree_llm_model_t *model, const int64_t *prompt,
                       iree_host_size_t prompt_length, int64_t pad_token,
                       iree_vm_list_t *extra_inputs, int64_t *out_token,
                       iree_hal_buffer_view_t **out_logits);

// Runs decode on |token| at iree_llm_model_position and returns the token
// following it. Fails with OUT_OF_RANGE once the cache holds max_seq_len
// tokens, and with FAILED_PRECONDITION before a prefill.
iree_status_t iree_llm_model_decode(iree_llm_model_t *model, int64_t token,
                                    int64_t *out_token);

//===----------------------------------------------------------------------===//
// Generation
//===----------------------------------------------------------------------===//

typedef enum iree_llm_stop_reason_e {
  // max_new_tokens tokens were generated.
  IREE_LLM_STOP_REASON_MAX_NEW_TOKENS = 0,
  // The model produced one of the stop tokens.
  IREE_LLM_STOP_REASON_STOP_TOKEN,
  // The KV cache is full.
  IREE_LLM_STOP_REASON_MAX_SEQ_LEN,
  // The token callback asked to stop.
  IREE_LLM_STOP_REASON_CALLBACK,
} iree_llm_stop_reason_t;

// Returns a short description of |reason|, e.g. "stop token".
const char *iree_llm_stop_reason_string(iree_llm_stop_reason_t reason);

typedef struct iree_llm_generate_params_t {
  iree_host_size_t max_new_tokens;
  // See iree_llm_model_prefill.
  int64_t pad_token;
  // Tokens that end generation; they are not passed to the token callback.
  iree_host_size_t stop_token_count;
  const int64_t *stop_tokens;
  // See iree_llm_model_prefill's |extra_inputs|; NULL for none.
  iree_vm_list_t *prefill_inputs;
} iree_llm_generate_params_t;

// Called with each generated token, in order. Setting |*out_stop| ends
// generation without an error (e.g. a client went away); returning an error
// aborts it with that error.
typedef iree_status_t (*iree_llm_token_callback_fn_t)(void *user_data,
                                                      int64_t token,
                                                      bool *out_stop);
typedef struct iree_llm_token_callback_t {
  iree_llm_token_callback_fn_t fn;
  void *user_data;
} iree_llm_token_callback_t;

typedef struct iree_llm_generate_stats_t {
  iree_host_size_t prompt_length;
  iree_host_size_t padded_length;
  // Tokens passed to the callback; the first comes from prefill.
  iree_host_size_t generated_count;
  // Decode invocations, including one that returned a stop token.
  iree_host_size_t decode_steps;
  // Time spent in iree_llm_model_prefill, and summed over all decode_steps in
  // iree_llm_model_decode; the token callback is in neither.
  iree_duration_t prefill_duration;
  iree_duration_t decode_duration;
  iree_llm_stop_reason_t stop_reason;
} iree_llm_generate_stats_t;

// Runs |prompt| through prefill, then decode until a stop condition, passing
// each generated token to |callback|. |out_stats| is filled in also when an
// error is returned, as far as generation got. With max_new_tokens == 0 the
// prompt is only validated.
//
// If |out_prefill_logits| is not NULL, it receives prefill's logits as
// iree_llm_model_prefill's |out_logits| does, or NULL if prefill did not run
// or failed; the caller releases them, also when an error is returned.
iree_status_t iree_llm_generate(iree_llm_model_t *model, const int64_t *prompt,
                                iree_host_size_t prompt_length,
                                const iree_llm_generate_params_t *params,
                                iree_llm_token_callback_t callback,
                                iree_llm_generate_stats_t *out_stats,
                                iree_hal_buffer_view_t **out_prefill_logits);

#ifdef __cplusplus
} // extern "C"
#endif // __cplusplus

#endif // IREE_HEXAGON_PLUGINS_TOOLS_LLM_ENGINE_H_

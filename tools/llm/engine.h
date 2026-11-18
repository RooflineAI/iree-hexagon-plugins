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
//           input_length: i64[1]) -> (next_token: i64[1, 1], logits)
//     Runs the prompt, padded to P, a multiple of block_seq_stride of at
//     least min_prefill_blocks blocks. input_length is the index of the last
//     real token, whose argmax is next_token.
//   decode(input_ids: i64[1, 1], cache_position: i64[pos + 1])
//       -> next_token: i64[1, 1]
//     Runs one token at position pos against the cache.
//
// Both run in one VM context, so the cache prefill writes is the one decode
// reads. Sampling is greedy, inside the module.
//
// The engine reads no flags of its own: everything is passed in, so the CLI
// tool (iree-run-llm) and other frontends (e.g. a server) can share it. The HAL
// device is still created by IREE's tooling from its --device= family of flags.
//
// Thread safety: an iree_llm_model_t holds one KV cache and so runs one
// sequence at a time; a prefill starts a new sequence and discards the
// previous one. Callers that share a model across threads must serialize all
// calls on it. Separate models (separate contexts) are independent.

#ifndef IREE_HEXAGON_PLUGINS_TOOLS_LLM_ENGINE_H_
#define IREE_HEXAGON_PLUGINS_TOOLS_LLM_ENGINE_H_

#include <stdbool.h>
#include <stdint.h>

#include "iree/base/api.h"
#include "iree/tokenizer/tokenizer.h"
#include "iree/vm/api.h"

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

//===----------------------------------------------------------------------===//
// Model
//===----------------------------------------------------------------------===//

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

// Sets KvCacheLM's defaults: 128 / 32 / 2, "prefill" and "decode".
void iree_llm_model_options_initialize(iree_llm_model_options_t *out_options);

typedef struct iree_llm_model_t iree_llm_model_t;

// Creates one VM context for |modules| on the device IREE's tooling flags
// select, and looks up the prefill and decode functions in the last module.
// The context retains the modules; the caller keeps its own references.
iree_status_t iree_llm_model_create(iree_vm_instance_t *instance,
                                    iree_host_size_t module_count,
                                    iree_vm_module_t **modules,
                                    const iree_llm_model_options_t *options,
                                    iree_allocator_t host_allocator,
                                    iree_llm_model_t **out_model);

// Frees |model|, closing a HAL replay capture if one was requested. NULL is
// a no-op.
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
iree_status_t iree_llm_model_prefill(iree_llm_model_t *model,
                                     const int64_t *prompt,
                                     iree_host_size_t prompt_length,
                                     int64_t pad_token, int64_t *out_token);

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
  iree_duration_t prefill_duration;
  iree_duration_t decode_duration;
  iree_llm_stop_reason_t stop_reason;
} iree_llm_generate_stats_t;

// Runs |prompt| through prefill, then decode until a stop condition, passing
// each generated token to |callback|. |out_stats| is filled in also when an
// error is returned, as far as generation got. With max_new_tokens == 0 the
// prompt is only validated.
iree_status_t iree_llm_generate(iree_llm_model_t *model, const int64_t *prompt,
                                iree_host_size_t prompt_length,
                                const iree_llm_generate_params_t *params,
                                iree_llm_token_callback_t callback,
                                iree_llm_generate_stats_t *out_stats);

//===----------------------------------------------------------------------===//
// Tokenizer
//===----------------------------------------------------------------------===//

// Loads a HuggingFace tokenizer.json from |path|. Free it with
// iree_tokenizer_free.
iree_status_t iree_llm_tokenizer_load_file(iree_string_view_t path,
                                           iree_allocator_t host_allocator,
                                           iree_tokenizer_t **out_tokenizer);

// Tokenizes |text| from the start of an input. With |add_special_tokens| the
// tokenizer's special tokens (e.g. BOS) are added, as HuggingFace's
// tokenizer(text) does by default. The caller frees |*out_tokens| with
// |host_allocator|; it is NULL when |*out_count| is 0.
iree_status_t iree_llm_tokenizer_encode(const iree_tokenizer_t *tokenizer,
                                        iree_string_view_t text,
                                        bool add_special_tokens,
                                        iree_allocator_t host_allocator,
                                        int64_t **out_tokens,
                                        iree_host_size_t *out_count);

// Returns the tokenizer's EOS token, or -1 if it has none.
int64_t iree_llm_tokenizer_eos_token(const iree_tokenizer_t *tokenizer);

// Returns the tokenizer's pad token, or 0 if it has none.
int64_t iree_llm_tokenizer_pad_token(const iree_tokenizer_t *tokenizer);

// Receives a piece of detokenized text. |text| is only valid during the call.
typedef iree_status_t (*iree_llm_text_callback_fn_t)(void *user_data,
                                                     iree_string_view_t text);
typedef struct iree_llm_text_callback_t {
  iree_llm_text_callback_fn_t fn;
  void *user_data;
} iree_llm_text_callback_t;

// Turns a stream of tokens into text, one token at a time. Byte-level tokens
// that only form a character together are held back until it is complete.
typedef struct iree_llm_detokenizer_t {
  iree_allocator_t host_allocator;
  iree_llm_text_callback_t callback;
  void *state_storage;
  iree_tokenizer_decode_state_t *state;
  char text[IREE_TOKENIZER_DECODE_OUTPUT_RECOMMENDED_SIZE];
} iree_llm_detokenizer_t;

// Starts a detokenized stream that |tokenizer| must outlive.
// |skip_special_tokens| leaves out special tokens (BOS/EOS/...).
iree_status_t iree_llm_detokenizer_initialize(
    const iree_tokenizer_t *tokenizer, bool skip_special_tokens,
    iree_llm_text_callback_t callback, iree_allocator_t host_allocator,
    iree_llm_detokenizer_t *out_detokenizer);

// Feeds |token|, passing the text it completes, if any, to the callback.
iree_status_t iree_llm_detokenizer_feed(iree_llm_detokenizer_t *detokenizer,
                                        int64_t token);

// Ends the stream, passing the text still held back, if any, to the callback.
iree_status_t iree_llm_detokenizer_finish(iree_llm_detokenizer_t *detokenizer);

// Releases the stream's state. Safe on a zeroed or failed-to-initialize
// detokenizer.
void iree_llm_detokenizer_deinitialize(iree_llm_detokenizer_t *detokenizer);

#ifdef __cplusplus
} // extern "C"
#endif // __cplusplus

#endif // IREE_HEXAGON_PLUGINS_TOOLS_LLM_ENGINE_H_

// Copyright 2026 RooflineAI GmbH
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Text side of the LLM inference engine (llm/engine.h): turns a prompt into
// token IDs and generated token IDs back into text, with IREE's tokenizer
// from a HuggingFace tokenizer.json. The engine itself only handles token IDs,
// so frontends that tokenize elsewhere do not need this library.

#ifndef IREE_HEXAGON_PLUGINS_TOOLS_LLM_TOKENIZER_H_
#define IREE_HEXAGON_PLUGINS_TOOLS_LLM_TOKENIZER_H_

#include <stdbool.h>
#include <stdint.h>

#include "iree/base/api.h"
#include "iree/tokenizer/tokenizer.h"

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

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

#endif // IREE_HEXAGON_PLUGINS_TOOLS_LLM_TOKENIZER_H_

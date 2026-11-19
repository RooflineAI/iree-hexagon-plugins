// Copyright 2026 RooflineAI GmbH
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "llm/tokenizer.h"

#include <inttypes.h>
#include <string.h>

#include "iree/io/file_contents.h"
#include "iree/tokenizer/format/huggingface/tokenizer_json.h"
#include "iree/tokenizer/vocab/vocab.h"

iree_status_t iree_llm_tokenizer_load_file(iree_string_view_t path,
                                           iree_allocator_t host_allocator,
                                           iree_tokenizer_t **out_tokenizer) {
  *out_tokenizer = NULL;
  iree_io_file_contents_t *contents = NULL;
  IREE_RETURN_IF_ERROR(
      iree_io_file_contents_read(path, host_allocator, &contents),
      "reading '%.*s'", (int)path.size, path.data);
  iree_status_t status = iree_status_annotate_f(
      iree_tokenizer_from_huggingface_json(
          iree_make_string_view((const char *)contents->const_buffer.data,
                                contents->const_buffer.data_length),
          host_allocator, out_tokenizer),
      "loading tokenizer '%.*s'", (int)path.size, path.data);
  iree_io_file_contents_free(contents);
  return status;
}

iree_status_t iree_llm_tokenizer_encode(const iree_tokenizer_t *tokenizer,
                                        iree_string_view_t text,
                                        bool add_special_tokens,
                                        iree_allocator_t host_allocator,
                                        int64_t **out_tokens,
                                        iree_host_size_t *out_count) {
  *out_tokens = NULL;
  *out_count = 0;

  iree_tokenizer_encode_flags_t flags =
      IREE_TOKENIZER_ENCODE_FLAG_AT_INPUT_START;
  if (add_special_tokens) {
    flags |= IREE_TOKENIZER_ENCODE_FLAG_ADD_SPECIAL_TOKENS;
  }
  // No token is shorter than a byte, and add room for the special tokens.
  iree_host_size_t capacity =
      text.size + iree_tokenizer_max_special_token_count(tokenizer) + 1;
  iree_tokenizer_token_id_t *ids = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc_array(
      host_allocator, capacity, sizeof(*ids), (void **)&ids));
  iree_host_size_t count = 0;
  iree_status_t status = iree_tokenizer_encode(
      tokenizer, text, flags,
      iree_tokenizer_make_token_output(ids, NULL, NULL, capacity),
      host_allocator, &count);
  int64_t *tokens = NULL;
  if (iree_status_is_ok(status) && count > 0) {
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
  return status;
}

int64_t iree_llm_tokenizer_eos_token(const iree_tokenizer_t *tokenizer) {
  int32_t eos =
      iree_tokenizer_vocab_special_ids(iree_tokenizer_vocab(tokenizer)).eos;
  return eos >= 0 ? eos : -1;
}

int64_t iree_llm_tokenizer_pad_token(const iree_tokenizer_t *tokenizer) {
  int32_t pad =
      iree_tokenizer_vocab_special_ids(iree_tokenizer_vocab(tokenizer)).pad;
  return pad >= 0 ? pad : 0;
}

iree_status_t iree_llm_detokenizer_initialize(
    const iree_tokenizer_t *tokenizer, bool skip_special_tokens,
    iree_llm_text_callback_t callback, iree_allocator_t host_allocator,
    iree_llm_detokenizer_t *out_detokenizer) {
  memset(out_detokenizer, 0, sizeof(*out_detokenizer));
  out_detokenizer->host_allocator = host_allocator;
  out_detokenizer->callback = callback;
  iree_host_size_t state_size = 0;
  IREE_RETURN_IF_ERROR(
      iree_tokenizer_decode_state_calculate_size(tokenizer, &state_size));
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(host_allocator, state_size,
                                             &out_detokenizer->state_storage));
  iree_tokenizer_decode_flags_t flags =
      skip_special_tokens ? IREE_TOKENIZER_DECODE_FLAG_SKIP_SPECIAL_TOKENS
                          : IREE_TOKENIZER_DECODE_FLAG_NONE;
  iree_status_t status = iree_tokenizer_decode_state_initialize(
      tokenizer, flags,
      iree_make_byte_span(out_detokenizer->state_storage, state_size),
      &out_detokenizer->state);
  if (!iree_status_is_ok(status)) {
    iree_allocator_free(host_allocator, out_detokenizer->state_storage);
    out_detokenizer->state_storage = NULL;
  }
  return status;
}

static iree_status_t
iree_llm_detokenizer_emit(iree_llm_detokenizer_t *detokenizer,
                          iree_host_size_t length) {
  if (length == 0)
    return iree_ok_status();
  return detokenizer->callback.fn(
      detokenizer->callback.user_data,
      iree_make_string_view(detokenizer->text, length));
}

iree_status_t iree_llm_detokenizer_feed(iree_llm_detokenizer_t *detokenizer,
                                        int64_t token) {
  iree_tokenizer_token_id_t id = (iree_tokenizer_token_id_t)token;
  iree_tokenizer_token_id_list_t tokens =
      iree_tokenizer_make_token_id_list(&id, 1);
  while (tokens.count > 0) {
    iree_host_size_t consumed = 0, length = 0;
    IREE_RETURN_IF_ERROR(iree_tokenizer_decode_state_feed(
        detokenizer->state, tokens,
        iree_make_mutable_string_view(detokenizer->text,
                                      sizeof(detokenizer->text)),
        &consumed, &length));
    if (consumed == 0 && length == 0) {
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "token %" PRId64 " does not fit the text buffer",
                              token);
    }
    IREE_RETURN_IF_ERROR(iree_llm_detokenizer_emit(detokenizer, length));
    tokens.values += consumed;
    tokens.count -= consumed;
  }
  return iree_ok_status();
}

iree_status_t iree_llm_detokenizer_finish(iree_llm_detokenizer_t *detokenizer) {
  iree_host_size_t length = 0;
  IREE_RETURN_IF_ERROR(iree_tokenizer_decode_state_finalize(
      detokenizer->state,
      iree_make_mutable_string_view(detokenizer->text,
                                    sizeof(detokenizer->text)),
      &length));
  return iree_llm_detokenizer_emit(detokenizer, length);
}

void iree_llm_detokenizer_deinitialize(iree_llm_detokenizer_t *detokenizer) {
  if (detokenizer->state)
    iree_tokenizer_decode_state_deinitialize(detokenizer->state);
  iree_allocator_free(detokenizer->host_allocator, detokenizer->state_storage);
  memset(detokenizer, 0, sizeof(*detokenizer));
}

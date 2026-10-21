// Copyright 2026 RooflineAI GmbH
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Turns off Android's heap pointer tagging for the Tracy server tools.
// Tracy's server stores heap pointers in 48 bits (server/TracyShortPtr.hpp),
// dropping the tag in the top byte, and bionic's free() aborts on an untagged
// pointer ("Pointer tag for 0x... was truncated"). Runs as a constructor, so
// before the tools allocate anything or start a thread.

#include <malloc.h>

__attribute__((constructor)) static void disable_heap_pointer_tagging(void) {
  mallopt(M_BIONIC_SET_HEAP_TAGGING_LEVEL, M_HEAP_TAGGING_LEVEL_NONE);
}

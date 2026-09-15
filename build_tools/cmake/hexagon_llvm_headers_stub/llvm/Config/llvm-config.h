// Minimal stand-in for LLVM's generated llvm/Config/llvm-config.h, for
// consumers that only need a few header-only LLVM ADT/Support utilities
// (StringMap, ArrayRef, MemAlloc, ...) without configuring or linking real
// LLVM libraries. Only the macros those specific headers use are defined.

#ifndef LLVM_CONFIG_H
#define LLVM_CONFIG_H

#define LLVM_VERSION_MAJOR 23
#define LLVM_VERSION_MINOR 0
#define LLVM_VERSION_PATCH 0
#define LLVM_VERSION_STRING "23.0.0git"

#define LLVM_DEFAULT_TARGET_TRIPLE ""

#define LLVM_ENABLE_THREADS 1

#endif

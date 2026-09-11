// Minimal stand-in for LLVM's generated llvm/Config/abi-breaking.h; see
// llvm-config.h in this same directory. LLVM_DISABLE_ABI_BREAKING_CHECKS_ENFORCING
// is passed as a compile definition alongside this header, so the real
// header's link-time mismatch checks (meant for linking against a real
// libSupport) are skipped entirely -- consumers here only use header-only
// ADT/Support utilities.

#ifndef LLVM_ABI_BREAKING_CHECKS_H
#define LLVM_ABI_BREAKING_CHECKS_H

#include "llvm/Config/llvm-config.h"

#define LLVM_ENABLE_ABI_BREAKING_CHECKS 0
#define LLVM_ENABLE_REVERSE_ITERATION 0

#endif

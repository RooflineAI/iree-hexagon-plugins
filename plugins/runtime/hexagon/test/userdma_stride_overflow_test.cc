// Copyright 2026 RooflineAI GmbH
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "qcom_hexagon_backend/bin/runtime/UserDMA/UserDMA.h"

#include <cstdint>
#include <cstdio>
#include <cstring>

namespace {

constexpr uint32_t kWidth = 16;
constexpr uint32_t kHeight = 2;
constexpr uint32_t kSourceStride = 65568;
constexpr uint32_t kDestinationStride = 16;

alignas(128) uint8_t source[kSourceStride + kWidth];
alignas(128) uint8_t destination[kHeight * kDestinationStride];

int fail(const char *message) {
  std::printf("FAIL: %s\n", message);
  return 1;
}

} // namespace

int main(int argc, char **argv) {
  (void)argc;
  (void)argv;

  std::memset(source, 0, sizeof(source));
  std::memset(destination, 0, sizeof(destination));
  for (uint32_t column = 0; column < kWidth; ++column) {
    source[column] = static_cast<uint8_t>(0x10u + column);
    source[kSourceStride + column] = static_cast<uint8_t>(0x80u + column);
  }

  hexagon::userdma::DMAStatus status = hexagon::userdma::DMAFailure;
  hexagon::userdma::UserDMA dma(/*fifo_len=*/2);
  uint32_t token = dma.copy2D(
      source, hexagon::userdma::DDR, destination, hexagon::userdma::DDR, kWidth,
      kHeight, kSourceStride, kDestinationStride,
      /*bypassCacheSrc=*/false, /*bypassCacheDst=*/false,
      /*isOrdered=*/false, /*cacheAllocationPolicy=*/0, &status);
  if (status != hexagon::userdma::DMASuccess)
    return fail("copy2D returned failure");

  dma.wait(token);
  if (std::memcmp(destination, source, kWidth) != 0)
    return fail("first row does not match");
  if (std::memcmp(destination + kDestinationStride, source + kSourceStride,
                  kWidth) != 0)
    return fail("second row does not match");

  std::printf("PASS: overflowing UserDMA source stride\n");
  return 0;
}

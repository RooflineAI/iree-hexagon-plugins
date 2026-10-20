# Copyright 2026 RooflineAI GmbH
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Hand-written, because not supported by the bazel_to_cmake generator.
#
# rsync for the phone, the CMake counterpart of BUILD.bazel in this directory:
# the upstream release, built with its own configure and make by the plain NDK
# compiler. build_tools/cmake/build_and_package.sh picks up `build/rsync` and
# `src/COPYING` from this directory's binary dir.

include(ExternalProject)

# Keep in lockstep with the "rsync" http_archive in MODULE.bazel.
set(RSYNC_VERSION 3.5.1)
set(RSYNC_SHA256 c55f9c9dc10fb8bec397b399a0fdded53cc9a2d8e30891bb0d63724d25c37bef)

# Keep in lockstep with _CONFIGURE_OPTIONS in BUILD.bazel.
set(RSYNC_CONFIGURE_OPTIONS
  --host=aarch64-linux-android
  # The bundled zlib and popt, so the binary needs nothing but libc.
  --with-included-zlib
  --with-included-popt
  # Optional features that need libraries the NDK does not have, and that a
  # plain file transfer does not use.
  --disable-md2man
  --disable-openssl
  --disable-xxhash
  --disable-zstd
  --disable-lz4
  --disable-idn
  --disable-iconv
  --disable-locale
  --disable-acl-support
  --disable-xattr-support
)

# The compiler this tree builds everything else with, targeting the API level
# that android.toolchain.cmake derives from ANDROID_PLATFORM.
set(_RSYNC_CC "${CMAKE_C_COMPILER} --target=${CMAKE_C_COMPILER_TARGET}")

ExternalProject_Add(iree_hexagon_rsync
  URL "https://download.samba.org/pub/rsync/src/rsync-${RSYNC_VERSION}.tar.gz"
  URL_HASH "SHA256=${RSYNC_SHA256}"
  # Keep the archive's timestamps: rsync's configure re-runs autoconf when
  # configure.ac looks newer than configure.sh.
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE
  SOURCE_DIR "${CMAKE_CURRENT_BINARY_DIR}/src"
  BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/build"
  CONFIGURE_COMMAND
    ${CMAKE_COMMAND} -E env "CC=${_RSYNC_CC}"
    <SOURCE_DIR>/configure ${RSYNC_CONFIGURE_OPTIONS}
  BUILD_COMMAND make -j8 rsync
  INSTALL_COMMAND ""
  BUILD_BYPRODUCTS "${CMAKE_CURRENT_BINARY_DIR}/build/rsync"
)

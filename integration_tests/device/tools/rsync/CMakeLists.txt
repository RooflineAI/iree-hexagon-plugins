# Copyright 2026 RooflineAI GmbH
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Hand-written, because not supported by the bazel_to_cmake generator.
#
# rsync for the phone, with zstd and xxHash, the CMake counterpart of
# BUILD.bazel in this directory (see there for why): the upstream releases,
# built with the plain NDK compiler. build_tools/cmake/build_and_package.sh
# picks up `build/rsync` and the licenses under `src/` from this directory's
# binary dir.

include(ExternalProject)

# Keep the versions in lockstep with the rsync, rsync_zstd and rsync_xxhash
# http_archives in MODULE.bazel.
set(_RSYNC_URL "https://github.com/RsyncProject/rsync/releases/download/v3.5.1/rsync-3.5.1.tar.gz")
set(_RSYNC_SHA256 c55f9c9dc10fb8bec397b399a0fdded53cc9a2d8e30891bb0d63724d25c37bef)
set(_ZSTD_URL "https://github.com/facebook/zstd/releases/download/v1.5.7/zstd-1.5.7.tar.gz")
set(_ZSTD_SHA256 eb33e51f49a15e023950cd7825ca74a4a2b43db8354825ac24fc1b7ee09e6fa3)
set(_XXHASH_URL "https://github.com/Cyan4973/xxHash/archive/refs/tags/v0.8.3.tar.gz")
set(_XXHASH_SHA256 aae608dfe8213dfd05d909a57718ef82f30722c392344583d3f39050c7f29a80)

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
  --disable-lz4
  --disable-idn
  --disable-iconv
  --disable-locale
  --disable-acl-support
  --disable-xattr-support
)

# The compiler this tree builds everything else with, targeting the API level
# that android.toolchain.cmake derives from ANDROID_PLATFORM.
set(_CC "${CMAKE_C_COMPILER} --target=${CMAKE_C_COMPILER_TARGET}")

set(_SRC_DIR "${CMAKE_CURRENT_BINARY_DIR}/src")

# zstd: the static library only, single threaded, without the dictionary
# builder and legacy formats rsync does not use.
ExternalProject_Add(iree_hexagon_rsync_zstd
  URL "${_ZSTD_URL}"
  URL_HASH "SHA256=${_ZSTD_SHA256}"
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE
  SOURCE_DIR "${_SRC_DIR}/zstd"
  CONFIGURE_COMMAND ""
  BUILD_COMMAND
    make -C <SOURCE_DIR>/lib -j8 libzstd.a
    "CC=${_CC}" "AR=${CMAKE_AR}"
    ZSTD_LIB_DICTBUILDER=0 ZSTD_LIB_DEPRECATED=0 ZSTD_LEGACY_SUPPORT=0
  INSTALL_COMMAND ""
  BUILD_BYPRODUCTS "${_SRC_DIR}/zstd/lib/libzstd.a"
)

# xxHash: its one source file.
ExternalProject_Add(iree_hexagon_rsync_xxhash
  URL "${_XXHASH_URL}"
  URL_HASH "SHA256=${_XXHASH_SHA256}"
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE
  SOURCE_DIR "${_SRC_DIR}/xxhash"
  BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/xxhash"
  CONFIGURE_COMMAND ""
  BUILD_COMMAND
    ${CMAKE_C_COMPILER} --target=${CMAKE_C_COMPILER_TARGET}
    -O2 -c <SOURCE_DIR>/xxhash.c -o <BINARY_DIR>/xxhash.o
  COMMAND ${CMAKE_AR} rcs <BINARY_DIR>/libxxhash.a <BINARY_DIR>/xxhash.o
  INSTALL_COMMAND ""
  BUILD_BYPRODUCTS "${CMAKE_CURRENT_BINARY_DIR}/xxhash/libxxhash.a"
)

# rsync, out of tree. Its configure aborts if zstd or xxHash is unusable, as
# both are enabled by default, and the config.h check guards against that
# changing.
ExternalProject_Add(iree_hexagon_rsync
  URL "${_RSYNC_URL}"
  URL_HASH "SHA256=${_RSYNC_SHA256}"
  # Keep the archive's timestamps: rsync's configure re-runs autoconf when
  # configure.ac looks newer than configure.sh.
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE
  SOURCE_DIR "${_SRC_DIR}/rsync"
  BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/build"
  DEPENDS iree_hexagon_rsync_zstd iree_hexagon_rsync_xxhash
  CONFIGURE_COMMAND
    ${CMAKE_COMMAND} -E env
    "CC=${_CC}"
    "CPPFLAGS=-I${_SRC_DIR}/zstd/lib -I${_SRC_DIR}/xxhash"
    "LDFLAGS=-L${_SRC_DIR}/zstd/lib -L${CMAKE_CURRENT_BINARY_DIR}/xxhash"
    <SOURCE_DIR>/configure ${RSYNC_CONFIGURE_OPTIONS}
  BUILD_COMMAND grep -q "^#define SUPPORT_ZSTD 1" config.h
  COMMAND grep -q "^#define SUPPORT_XXHASH 1" config.h
  COMMAND make -j8 rsync
  INSTALL_COMMAND ""
  BUILD_BYPRODUCTS "${CMAKE_CURRENT_BINARY_DIR}/build/rsync"
)

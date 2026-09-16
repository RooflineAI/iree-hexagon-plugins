// Minimal stand-in for <alloca.h>, which the Hexagon SDK's toolchain does
// not ship (it has no libc alloca() declaration anywhere in its include
// tree), unlike glibc/bionic/musl. hexagon-clang still recognizes alloca()
// as a builtin, so this just spells that out the way the real header would.

#ifndef ALLOCA_H
#define ALLOCA_H

#define alloca(size) __builtin_alloca(size)

#endif

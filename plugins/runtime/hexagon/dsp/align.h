// Copyright 2025 RooflineAI GmbH

#ifndef HEXAGON_DSP_ALIGN_H
#define HEXAGON_DSP_ALIGN_H

/**
 * Align a variable to a bigger multiple of bytes.
 * Hexagon version of iree_alignas(),
 * iree/base/alignment.h, which provides iree_alignas(), does not support
 * Hexagon
 */
#define HEXAGON_ALIGNAS(x) __attribute__((__aligned__(x)))

/**
 * Get alignment of type.
 * Hexagon version of iree_alignof(),
 * iree/base/alignment.h, which provides iree_alignas(), does not support
 * Hexagon
 */
#define HEXAGON_ALIGNOF(T) __alignof__(T)

/**
 * Type alignment wrapper, to ensure a minimum alignment.
 */
#define HEXAGON_BIG_ALIGNOF(T) (HEXAGON_ALIGNOF(T) > 8 ? HEXAGON_ALIGNOF(T) : 8)

/**
 * Align a size S for a certain type T, i.e., find the next non-smaller multiple
 * of the size of the type.
 */
#define HEXAGON_ALIGN_SIZE_FOR_TYPE(S, T)                                      \
  (((S) + (HEXAGON_BIG_ALIGNOF(T) - 1)) & ~(HEXAGON_BIG_ALIGNOF(T) - 1))

#endif // #ifndef HEXAGON_DSP_ALIGN_H

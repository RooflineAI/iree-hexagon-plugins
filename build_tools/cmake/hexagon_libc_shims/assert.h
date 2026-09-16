// Supplement for <assert.h>: the Hexagon SDK bundles a pre-C11 Dinkumware
// assert.h that predates static_assert/_Static_assert entirely. Pull in the
// real one for assert()/NDEBUG handling via include_next, then add what a
// C11 <assert.h> is additionally supposed to provide.

#ifndef IREE_HEXAGON_SHIM_ASSERT_H
#define IREE_HEXAGON_SHIM_ASSERT_H

#include_next <assert.h>

#ifndef static_assert
#define static_assert _Static_assert
#endif

#endif

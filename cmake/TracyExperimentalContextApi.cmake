# Passed as CMAKE_PROJECT_INCLUDE, so this runs right after IREE's top-level
# project() call -- early enough that IREE_TRACING_EXPERIMENTAL_CONTEXT_API
# is visible everywhere, including IREE's own cmake-configure-time checks,
# not just when compiling individual files.
#
# add_compile_definitions() only adds to the current directory scope's
# definitions; unlike overriding CMAKE_C_FLAGS/CMAKE_CXX_FLAGS directly, it
# can't clobber whatever flags the active toolchain file (Hexagon's or the
# Android NDK's) already seeded there.
add_compile_definitions(IREE_TRACING_EXPERIMENTAL_CONTEXT_API=1)

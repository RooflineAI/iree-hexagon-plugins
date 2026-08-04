// Copyright 2026 RooflineAI GmbH

#ifndef HEXAGON_DSP_RT_RUNTIME_STATE_H
#define HEXAGON_DSP_RT_RUNTIME_STATE_H

#include "profiler.h"
#include "runtime_state_fwd_decl.h"

/**
 * @brief The runtime state data structure contains pointers that are required
 *        to access the runtime, its state and its functionality from inside
 *        dispatches.
 *
 * The dispatch state has been extended for Hexagon to contain the
 * pointer to the runtime state structure.
 * See hexagon_dsp_extended_dispatch_state_t in dispatch_state.h.
 */
struct hexagon_rt_state_s {
  hexagon_rt_prof_context_t *prof_context;
}; // typedef-ed to hexagon_rt_state_t in runtime_state_fwd_decl.h

#endif // #ifndef HEXAGON_DSP_RT_RUNTIME_STATE_H

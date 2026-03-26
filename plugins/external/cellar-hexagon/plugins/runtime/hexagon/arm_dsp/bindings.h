// Copyright 2025 RooflineAI GmbH

#ifndef IREE_HAL_DRIVERS_HEXAGON_ARM_DSP_BINDINGS_H_
#define IREE_HAL_DRIVERS_HEXAGON_ARM_DSP_BINDINGS_H_

#include <stdint.h>

/// a buffer binding for a command buffer execution
typedef struct hexagon_rt_arm_dsp_binding_s {
  int32_t fd; ///< file descriptor of RPCmem allocation
  uint64_t offset;
  uint64_t length;
} __attribute__((packed)) hexagon_rt_arm_dsp_binding_t;

/// header of serialized binding table
typedef struct hexagon_rt_arm_dsp_binding_tab_s {
  uint32_t num_entries; ///< number of entries
  // followed by num_entries hexagon_rt_arm_dsp_binding_t
} __attribute__((packed)) hexagon_rt_arm_dsp_binding_tab_t;

#endif // #ifndef IREE_HAL_DRIVERS_HEXAGON_ARM_DSP_BINDINGS_H_

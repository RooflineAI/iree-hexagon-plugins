// Copyright 2025 RooflineAI GmbH

#ifndef IREE_HAL_DRIVERS_HEXAGON_ARM_DSP_CMD_BUF_H_
#define IREE_HAL_DRIVERS_HEXAGON_ARM_DSP_CMD_BUF_H_

#include <stdint.h>

/// a binding for a dispatch command
typedef struct hexagon_rt_arm_dsp_binding_s {
  uint32_t slot;
  int64_t buffer_handle;
  uint64_t offset;
  uint64_t length;
} __attribute__((packed)) hexagon_rt_arm_dsp_binding_t;

/// type of command buffer entry
/// note: type does not have fixed width, use
//  hexagon_rt_arm_dsp_cmd_type_store_t for strong the values in structs
typedef enum hexagon_rt_arm_dsp_cmd_type_enum {
  HEXAGON_RT_ARM_DSP_CMD_DISPATCH,
  HEXAGON_RT_ARM_DSP_CMD_BARRIER,
} hexagon_rt_arm_dsp_cmd_type_enum_t;
/// fixed width type used for storing enum values in struct,
/// used to store constants from hexagon_rt_arm_dsp_cmd_type_enum_t
typedef uint8_t hexagon_rt_arm_dsp_cmd_type_store_t;

/// header of serialized command (i.e., command buffer entry)
typedef struct hexagon_rt_arm_dsp_cmd_base_s {
  hexagon_rt_arm_dsp_cmd_type_store_t cmd_type;
  // followed by more data depending on cmd_type
} __attribute__((packed)) hexagon_rt_arm_dsp_cmd_base_t;

/// dispatch command, "derived" from hexagon_rt_arm_dsp_cmd_base_t
typedef struct hexagon_rt_arm_dsp_cmd_dispatch_s {
  hexagon_rt_arm_dsp_cmd_base_t base; ///< keep this the first entry
  int64_t executable_handle;
  int32_t entry_point;
  uint32_t num_bindings;
  // followed by num_bindings hexagon_rt_arm_dsp_binding_t
  // TODO: add fields
} __attribute__((packed)) hexagon_rt_arm_dsp_cmd_dispatch_t;

/// barrier command, "derived" from hexagon_rt_arm_dsp_cmd_base_t
typedef struct hexagon_rt_arm_dsp_cmd_barrier_s {
  hexagon_rt_arm_dsp_cmd_base_t base; ///< keep this the first entry
} __attribute__((packed)) hexagon_rt_arm_dsp_cmd_barrier_t;

/// header of serialized command buffer
typedef struct hexagon_rt_arm_dsp_cmd_buf_s {
  uint32_t num_entries; ///< number of entries
  // followed by num_entries hexagon_rt_arm_dsp_cmd_base_t
} __attribute__((packed)) hexagon_rt_arm_dsp_cmd_buf_t;

#endif // #ifndef IREE_HAL_DRIVERS_HEXAGON_ARM_DSP_CMD_BUF_H_

// Copyright 2025 RooflineAI GmbH

#ifndef IREE_HAL_DRIVERS_HEXAGON_ARM_DSP_CMD_BUF_H_
#define IREE_HAL_DRIVERS_HEXAGON_ARM_DSP_CMD_BUF_H_

#include <stdint.h>

/// a constant for dispatch
/// (Even though this is just a 32 bit unsigned integer, it is a struct in
///  order to mark it as packed, i.e., unaligned.)
/// Note: Constants are passed to _command_buffer_dispatch() as byte span
/// (see dispatch function in iree_hal_command_buffer_vtable_t in
/// third-party/iree/runtime/src/iree/hal/command_buffer.h:1100), but
/// to the kernel code inside an executable library as an array of uint32_t
/// values (see constants pointer in iree_hal_executable_dispatch_state_v0_t in
/// third-party/iree/runtime/src/iree/hal/local/executable_library.h:303).
/// The conversion includes checks (size multiple of 4, number of constants
/// not larger than UINT16_MAX) that can fail. It is therefore beneficial to
/// do the conversion as early as possible, which means the data type is already
/// uint32_t in serialized data being passed from ARM host to DSP.
typedef struct hexagon_rt_arm_dsp_con_s {
  uint32_t value;
} __attribute__((packed)) hexagon_rt_arm_dsp_con_t;

/// reference to a buffer for dispatch
typedef struct hexagon_rt_arm_dsp_buf_ref_s {
  uint32_t slot; ///< dispatch time binding slot, used if fd is -1
  int32_t fd;    ///< file descriptor of RPCmem allocation, -1 if unused
  uint64_t offset;
  uint64_t length;
} __attribute__((packed)) hexagon_rt_arm_dsp_buf_ref_t;

/// type of command buffer entry
/// note: type does not have fixed width, use
//  hexagon_rt_arm_dsp_cmd_type_store_t for strong the values in structs
typedef enum hexagon_rt_arm_dsp_cmd_type_enum {
  HEXAGON_RT_ARM_DSP_CMD_DISPATCH,
  HEXAGON_RT_ARM_DSP_CMD_BARRIER,
  HEXAGON_RT_ARM_DSP_CMD_COPY,
  HEXAGON_RT_ARM_DSP_CMD_FILL,
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
  /// DSP handle to executable that contains the function to call
  int64_t executable_handle;
  /// number of exported function to call from executable
  uint32_t export_ordinal;
  /// workgroup size - divided into x * y * z (z uses smaller type in IREE)
  //@{
  uint32_t workgroup_size_x;
  uint32_t workgroup_size_y;
  uint16_t workgroup_size_z;
  //@}
  /// workgroup count - divided into x * y * z (z uses smaller type in IREE)
  //@{
  uint32_t workgroup_count_x;
  uint32_t workgroup_count_y;
  uint16_t workgroup_count_z;
  //@}
  /// number of constant values to pass to the kernel
  uint16_t constant_count;
  /// number of bindings (i.e. fixed buffers or placeholders)
  uint32_t num_bindings;
  // followed by constant_count constants of hexagon_rt_arm_dsp_con_t each
  // followed by num_bindings hexagon_rt_arm_dsp_buf_ref_t
} __attribute__((packed)) hexagon_rt_arm_dsp_cmd_dispatch_t;

/// barrier command, "derived" from hexagon_rt_arm_dsp_cmd_base_t
typedef struct hexagon_rt_arm_dsp_cmd_barrier_s {
  hexagon_rt_arm_dsp_cmd_base_t base; ///< keep this the first entry
} __attribute__((packed)) hexagon_rt_arm_dsp_cmd_barrier_t;

/// copy command, "derived" from hexagon_rt_arm_dsp_cmd_base_t
typedef struct hexagon_rt_arm_dsp_cmd_copy_s {
  hexagon_rt_arm_dsp_cmd_base_t base; ///< keep this the first entry
  hexagon_rt_arm_dsp_buf_ref_t src;
  hexagon_rt_arm_dsp_buf_ref_t trgt;
} __attribute__((packed)) hexagon_rt_arm_dsp_cmd_copy_t;

/// fill command, "derived" from hexagon_rt_arm_dsp_cmd_base_t
typedef struct hexagon_rt_arm_dsp_cmd_fill_s {
  hexagon_rt_arm_dsp_cmd_base_t base; ///< keep this the first entry
  uint8_t pattern_length;
  uint8_t pattern[4]; // max pattern length according to comments in IREE is 4
  hexagon_rt_arm_dsp_buf_ref_t trgt;
} __attribute__((packed)) hexagon_rt_arm_dsp_cmd_fill_t;

/// header of serialized command buffer
typedef struct hexagon_rt_arm_dsp_cmd_buf_s {
  uint32_t num_entries; ///< number of entries
  // followed by num_entries "derived" from hexagon_rt_arm_dsp_cmd_base_t
} __attribute__((packed)) hexagon_rt_arm_dsp_cmd_buf_t;

#endif // #ifndef IREE_HAL_DRIVERS_HEXAGON_ARM_DSP_CMD_BUF_H_

// Copyright 2025 RooflineAI GmbH

#ifndef IREE_HAL_DRIVERS_HEXAGON_COMMAND_BUFFER_TYPES_H_
#define IREE_HAL_DRIVERS_HEXAGON_COMMAND_BUFFER_TYPES_H_

#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/hal/utils/resource_set.h"
#include "rpc_types.h"

//===----------------------------------------------------------------------===//
// iree_hal_hexagon_command_buffer_t
//===----------------------------------------------------------------------===//

typedef enum iree_hal_hexagon_command_type_e {
  IREE_HAL_HEXAGON_COMMAND_DISPATCH,
  IREE_HAL_HEXAGON_COMMAND_BARRIER,
  IREE_HAL_HEXAGON_COMMAND_COPY,
} iree_hal_hexagon_command_type_t;

/// "base class" of commands
typedef struct iree_hal_hexagon_command_base_s {
  /// neighbors in doubly linked list
  //@{
  struct iree_hal_hexagon_command_base_s *prev;
  struct iree_hal_hexagon_command_base_s *next;
  //@}
  /// type of "derived" command
  iree_hal_hexagon_command_type_t cmd_type;
  /// derived command data follows here
} iree_hal_hexagon_command_base_t;

/// dispatch command, "derived" from iree_hal_hexagon_command_base_t
typedef struct iree_hal_hexagon_command_dispatch_s {
  iree_hal_hexagon_command_base_t base; ///< keep this the first entry
  /// executable that contains the function to call, not owned
  iree_hal_executable_t *executable;
  /// RPC executable that contains the function to call, not owned
  rpc_executable_handle_t rpc_executable_handle;
  /// number of exported function to call from executable
  iree_hal_executable_export_ordinal_t export_ordinal;
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
  /// constant values to pass to the kernel
  /// Note: Constants are passed to _command_buffer_dispatch() as byte span
  /// (see dispatch function in iree_hal_command_buffer_vtable_t in
  /// third-party/iree/runtime/src/iree/hal/command_buffer.h:1100), but
  /// to the kernel code inside an executable library as an array of uint32_t
  /// values (see constants pointer in iree_hal_executable_dispatch_state_v0_t
  //  in third-party/iree/runtime/src/iree/hal/local/executable_library.h:303).
  /// The conversion includes checks (size multiple of 4, number of constants
  /// not larger than UINT16_MAX) that can fail. It is therefore beneficial to
  /// do the conversion as early as possible, which means the data type is
  /// already uint32_t when stored in the command buffer (i.e. here).
  //@{
  uint16_t constant_count;
  uint32_t *constants;
  //@}
  /// bindings (i.e. fixed buffers or placeholders), pointers inside points to
  /// memory in the same block as this structure
  iree_hal_buffer_ref_list_t bindings;
  /// ... buffer for constant_count uint32_t is appended after this struct
  /// ... buffer for binding.values is appended after the constant buffer
} iree_hal_hexagon_command_dispatch_t;

/// barrier command, "derived" from iree_hal_hexagon_command_base_t
typedef struct iree_hal_hexagon_command_barrier_s {
  iree_hal_hexagon_command_base_t base; ///< keep this the first entry
} iree_hal_hexagon_command_barrier_t;

/// buffer copy command, "derived" from iree_hal_hexagon_command_base_t
typedef struct iree_hal_hexagon_command_copy_s {
  iree_hal_hexagon_command_base_t base; ///< keep this the first entry
  iree_hal_buffer_ref_t src;
  iree_hal_buffer_ref_t dest;
} iree_hal_hexagon_command_copy_t;

typedef struct iree_hal_hexagon_command_buffer_t {
  iree_hal_command_buffer_t base;
  iree_allocator_t host_allocator;
  rpc_session_handle_t rpc_session_handle; // not owned, owner: device
  /// doubly linked list (non-circular) of entries
  //@{
  iree_hal_hexagon_command_base_t *first_entry;
  iree_hal_hexagon_command_base_t *last_entry;
  //@}
  /// owned resources, that's buffers used in direct buffer references (actual
  /// buffers being referenced at recording time) in transfers and dispatches
  iree_hal_resource_set_t *resource_set;
  /// command buffer on DSP side (valid if finalized, otherwise 0)
  rpc_command_buffer_handle_t rpc_command_buffer_handle;
} iree_hal_hexagon_command_buffer_t;

#endif // IREE_HAL_DRIVERS_HEXAGON_COMMAND_BUFFER_TYPES_H_

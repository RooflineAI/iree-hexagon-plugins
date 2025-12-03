// Copyright 2025 RooflineAI GmbH

#ifndef IREE_HAL_DRIVERS_HEXAGON_COMMAND_BUFFER_TYPES_H_
#define IREE_HAL_DRIVERS_HEXAGON_COMMAND_BUFFER_TYPES_H_

#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "rpc_types.h"

//===----------------------------------------------------------------------===//
// iree_hal_hexagon_command_buffer_t
//===----------------------------------------------------------------------===//

typedef enum iree_hal_hexagon_command_type_e {
  IREE_HAL_HEXAGON_COMMAND_DISPATCH,
  IREE_HAL_HEXAGON_COMMAND_BARRIER,
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
  /// bindings (i.e. fixed buffers or placeholders), pointers inside points to
  /// memory in the same block as this structure
  iree_hal_buffer_ref_list_t bindings;
  /// ... buffer for binding.values is appended after this struct
} iree_hal_hexagon_command_dispatch_t;

/// barrier command, "derived" from iree_hal_hexagon_command_base_t
typedef struct iree_hal_hexagon_command_barrier_s {
  iree_hal_hexagon_command_base_t base; ///< keep this the first entry
} iree_hal_hexagon_command_barrier_t;

typedef struct iree_hal_hexagon_command_buffer_t {
  iree_hal_command_buffer_t base;
  iree_allocator_t host_allocator;
  rpc_session_handle_t rpc_session_handle; // not owned, owner: device
  /// doubly linked list (non-circular) of entries
  //@{
  iree_hal_hexagon_command_base_t *first_entry;
  iree_hal_hexagon_command_base_t *last_entry;
  //@}
  /// command buffer on DSP side (valid if finalized, otherwise 0)
  rpc_command_buffer_handle_t rpc_command_buffer_handle;
} iree_hal_hexagon_command_buffer_t;

#endif // IREE_HAL_DRIVERS_HEXAGON_COMMAND_BUFFER_TYPES_H_

// Copyright 2025 RooflineAI GmbH

#ifndef IREE_HAL_DRIVERS_HEXAGON_SERIALIZE_COMMAND_BUFFER_TYPES_H_
#define IREE_HAL_DRIVERS_HEXAGON_SERIALIZE_COMMAND_BUFFER_TYPES_H_

#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/hal/utils/resource_set.h"
#include "rpc_types.h"

//===----------------------------------------------------------------------===//
// iree_hal_hexagon_command_buffer_t
//===----------------------------------------------------------------------===//

/// Function pointer type for translating a buffer to the DSP virtual address.
/// It is used to pass the translation function to the serialization functions
/// in order to decouple unit tests from the actual HAL.
/// @param[in] buffer pointer to IREE buffer
/// @param[out] out_dsp_vaddr filled with virtual address of buffer's memory on
///                           DSP
/// @return IREE status
typedef iree_status_t (*iree_hal_hexagon_buffer_to_dsp_vaddr_t)(
    iree_hal_buffer_t *buffer, rpc_dsp_vaddr_t *out_dsp_vaddr);

/// an entry in the command buffer
typedef struct iree_hal_hexagon_command_buffer_entry_s {
  /// next entry in linked list
  struct iree_hal_hexagon_command_buffer_entry_s *next;
  /// size of command data
  iree_host_size_t size;
  /// command data (beginning with hexagon_rt_arm_dsp_cmd_base_t) follows here,
  /// allocated within same memory block
} iree_hal_hexagon_command_buffer_entry_t;

typedef struct iree_hal_hexagon_command_buffer_t {
  iree_hal_command_buffer_t base;
  iree_allocator_t host_allocator;
  rpc_session_handle_t rpc_session_handle; // not owned, owner: device
  /// linked list of entries
  //@{
  iree_hal_hexagon_command_buffer_entry_t *first_entry;
  iree_hal_hexagon_command_buffer_entry_t *last_entry;
  //@}
  /// owned resources, that's buffers used in direct buffer references (actual
  /// buffers being referenced at recording time) in transfers and dispatches
  iree_hal_resource_set_t *resource_set;
  /// command buffer on DSP side (valid if finalized, otherwise 0)
  rpc_command_buffer_handle_t rpc_command_buffer_handle;
} iree_hal_hexagon_command_buffer_t;

#endif // IREE_HAL_DRIVERS_HEXAGON_SERIALIZE_COMMAND_BUFFER_TYPES_H_

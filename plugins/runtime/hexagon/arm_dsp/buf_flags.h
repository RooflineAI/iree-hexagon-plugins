// Copyright 2026 RooflineAI GmbH
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_HEXAGON_ARM_DSP_BUF_FLAGS_H_
#define IREE_HAL_DRIVERS_HEXAGON_ARM_DSP_BUF_FLAGS_H_

/// Flags of a buffer reference (hexagon_rt_arm_dsp_buf_ref_t) or a binding
/// table entry (hexagon_rt_arm_dsp_binding_t): what the host may do with the
/// allocation between executions, so the DSP can skip cache maintenance the
/// host does not need. 0 is always safe. For a KV cache, for example, they
/// tell the DSP if it can keep it hot.

/// The host never writes the allocation: it cannot map it, or the contents are
/// immutable. The DSP does not need to invalidate it before an execution.
#define HEXAGON_RT_ARM_DSP_BUF_FLAG_HOST_NO_WRITE (1u << 0)
/// The host never reads the allocation: it cannot map it. The DSP does not need
/// to flush it after an execution.
#define HEXAGON_RT_ARM_DSP_BUF_FLAG_HOST_NO_READ (1u << 1)

#endif // IREE_HAL_DRIVERS_HEXAGON_ARM_DSP_BUF_FLAGS_H_

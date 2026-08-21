# Copyright 2026 RooflineAI GmbH
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Workspace-specific external HAL driver configuration.

Loaded by @iree//runtime/src/iree/hal/drivers:BUILD.bazel (patched in
patches/iree/ to support this root-provided override) via "@//..." - only
resolvable because IREE is referenced through the "@iree" local_repository
alias (root's own repo mapping), not the "iree_core" bzlmod module. See
MODULE.bazel's comment on the "@iree" alias for why.
"""

def get_external_hal_drivers():
    """Returns the list of external HAL drivers for this workspace.

    Returns:
        List of struct(target=..., register_fn=..., [enabled_by=...]) for
        each external driver. "enabled_by" is a config_setting label; the
        driver is always included if omitted.
    """
    return [
        struct(
            target = "@//plugins/runtime/hexagon/registration:DriverModuleLib",
            register_fn = "iree_hal_hexagon_driver_module_register",
            enabled_by = "@//plugins/runtime/hexagon:hexagon_runtime_enabled",
        ),
    ]

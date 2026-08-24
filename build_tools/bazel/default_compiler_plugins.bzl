# Copyright 2026 RooflineAI GmbH
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Default compiler plugin registry for iree-hexagon-plugins Bazel builds.

IREE's own compiler/plugins/BUILD.bazel loads this file from the root
workspace via @//build_tools/bazel:default_compiler_plugins.bzl (see
third-party/iree/build_tools/bazel/default_compiler_plugins.bzl).
This is the extension point where this repo's own plugins are registered.
"""

def iree_default_compiler_plugin_ids():
    """Returns plugin IDs enabled by plain Bazel invocation."""
    return [
        "input_stablehlo",
        "input_torch",
        "input_tosa",
        "hal_target_llvm_cpu",
        "hal_target_local",
        "hal_target_vmvx",
        "hexagon",
        "example",
        "simple_io_sample",
    ]

def iree_default_compiler_plugins():
    """Returns all known compiler plugin registration targets.

    All stock IREE entries below deliberately use "@iree//..." (the
    local_repository alias, not the "iree_core" bzlmod module) - this dict is
    consumed from within "@iree//compiler/plugins/BUILD.bazel" itself, and
    every plugin registration target linked into the same binary must be
    resolved through the same repo instance to share C++ type identity with
    the rest of the statically-linked compiler (see MODULE.bazel's comment on
    the "@iree" alias for why "iree_core" can't be used here at all).
    """
    return {
        # Input plugins.
        "input_stablehlo": "@iree//compiler/plugins/input/StableHLO:registration",
        "input_tosa": "@iree//compiler/plugins/input/TOSA:registration",
        "input_torch": "@iree//compiler/plugins/input/Torch:registration",
        # Target plugins.
        "hal_target_cuda": "@iree//compiler/plugins/target/CUDA",
        "hal_target_llvm_cpu": "@iree//compiler/plugins/target/LLVMCPU",
        "hal_target_local": "@iree//compiler/plugins/target/Local",
        "hal_target_metal_spirv": "@iree//compiler/plugins/target/MetalSPIRV",
        "hal_target_rocm": "@iree//compiler/plugins/target/ROCM",
        "hal_target_vmvx": "@iree//compiler/plugins/target/VMVX",
        "hal_target_vulkan_spirv": "@iree//compiler/plugins/target/VulkanSPIRV",
        "hal_target_webgpu_spirv": "@iree//compiler/plugins/target/WebGPUSPIRV",
        # This repo's own plugins. The key must match the plugin id used in
        # the corresponding PluginRegistration.cpp's
        # `iree_register_compiler_plugin_<id>` entry point. "@//..." (rather
        # than bare "//...") is required here since this label is resolved
        # from within "@iree"'s repo mapping, where bare "//" would mean
        # "@iree" itself, not this root module.
        "hexagon": "@//plugins/compiler/hexagon:registration",
        # Sample plugins.
        "example": "@iree//samples/compiler_plugins/example:registration",
        "simple_io_sample": "@iree//samples/compiler_plugins/simple_io_sample:registration",
    }

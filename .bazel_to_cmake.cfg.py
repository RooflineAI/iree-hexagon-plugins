# Copyright 2026 RooflineAI GmbH
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import os
import re
import shlex

import bazel_to_cmake_converter
import bazel_to_cmake_targets

DEFAULT_ROOT_DIRS = [
    "plugins",
    "integration_tests/device/tools",
    "build_tools/bazel/overlays/hexagon_mlir",
]

REPO_MAP = {
    # This repo's own root module is "iree_hexagon_plugins" (see MODULE.bazel),
    # so its own targets use bare "//..." labels already handled by the
    # default (package-relative / repo-relative) conversion path.
    #
    # The IREE submodule is consumed as module "iree_core" but aliased to the
    # repo name "iree" (MODULE.bazel), so its labels look like "@iree//...".
    # bazel_to_cmake_targets.TargetConverter always looks for IREE's own
    # targets under the repo name given by REPO_MAP["@iree_core"].
    "@iree_core": "@iree",
}


class CustomBuildFileFunctions(bazel_to_cmake_converter.BuildFileFunctions):
    # The hexagon-mlir overlay (build_tools/bazel/overlays/hexagon_mlir/
    # qcom_hexagon_backend) mostly holds Bazel build glue; most of the
    # C/C++ it "contains" is really the real hexagon-mlir submodule, reached
    # at Bazel-build time via a symlink-forest repository rule
    # (hexagon_mlir_configure.bzl: every overlay BUILD.bazel file is
    # symlinked into a synthetic @hexagon-mlir repo alongside every upstream
    # file not shadowed by the overlay). CMake has no equivalent of that
    # repository indirection, so srcs/hdrs entries that aren't physically
    # present under the overlay directory are redirected to their real
    # location in the submodule at the same path relative to
    # qcom_hexagon_backend/.
    _OVERLAY_ROOT_REL = "build_tools/bazel/overlays/hexagon_mlir/qcom_hexagon_backend"
    _SUBMODULE_ROOT_REL = "third-party/hexagon-mlir/qcom_hexagon_backend"

    def _redirect_overlay_paths(self, paths):
        if not paths or not isinstance(paths, list):
            return paths
        overlay_root_abs = os.path.join(
            self._repo_root, *self._OVERLAY_ROOT_REL.split("/")
        )
        # self._build_dir is absolute when the tool is invoked with
        # --recursive_dir (os.walk() over an absolute root), but is passed
        # through as-is (repo-root-relative, per the tool's own documented
        # usage) when given as a plain positional directory argument -- so it
        # cannot be compared against overlay_root_abs without normalizing.
        build_dir = self._build_dir
        if not os.path.isabs(build_dir):
            build_dir = os.path.join(self._repo_root, build_dir)
        if not (
            build_dir == overlay_root_abs
            or build_dir.startswith(overlay_root_abs + os.sep)
        ):
            return paths
        submodule_root_abs = os.path.join(
            self._repo_root, *self._SUBMODULE_ROOT_REL.split("/")
        )
        rel_dir = os.path.relpath(build_dir, overlay_root_abs)
        resolved = []
        for p in paths:
            # Target references (":foo", "//foo:bar", "@repo//...") and
            # CMake-generator-expression strings emitted by glob() are left
            # untouched -- only plain relative filenames are redirectable.
            if not isinstance(p, str) or p.startswith((":", "@", "$")) or "//" in p:
                resolved.append(p)
                continue
            if os.path.isfile(os.path.join(build_dir, p)):
                resolved.append(p)
                continue
            submodule_file = os.path.normpath(
                os.path.join(submodule_root_abs, rel_dir, p)
            )
            if not os.path.isfile(submodule_file):
                raise FileNotFoundError(
                    f"Overlay source '{p}' not found under {build_dir} nor at "
                    f"{submodule_file}"
                )
            # Emit a path relative to the CMakeLists.txt being generated
            # (iree_cc_library() resolves relative SRCS/HDRS against
            # CMAKE_CURRENT_SOURCE_DIR, same as plain in-tree sources) rather
            # than an absolute path baked in at generation time -- the
            # latter hardcodes this machine's checkout location into a
            # git-tracked file, breaking for every other clone.
            resolved.append(
                os.path.relpath(submodule_file, build_dir).replace(os.sep, "/")
            )
        return resolved

    def _includes_for_strip_include_prefix(self, strip_include_prefix):
        # The base converter's cc_library() accepts (and silently drops, via
        # **kwargs) strip_include_prefix -- it has no translation for it at
        # all. The CMake equivalent is an INCLUDES entry for the same
        # directory Bazel strips to, expressed relative to the CMakeLists.txt
        # being generated (see _convert_includes_block, which wraps each
        # entry in a $<BUILD_INTERFACE:${CMAKE_CURRENT_{SOURCE,BINARY}_DIR}/
        # ...> pair).
        if not strip_include_prefix:
            return None
        build_dir = self._build_dir
        if not os.path.isabs(build_dir):
            build_dir = os.path.join(self._repo_root, build_dir)
        if strip_include_prefix.startswith("/"):
            # Leading "/": workspace(repo-root)-relative.
            prefix_abs = os.path.join(
                self._repo_root, *strip_include_prefix.lstrip("/").split("/")
            )
        else:
            # No leading "/": Bazel resolves it relative to the *package*
            # (this BUILD file's own directory), not the repo root -- e.g.
            # strip_include_prefix = "." means "this directory is already
            # the include root", which must stay a "."/"." INCLUDES pair,
            # not 7 "../"s up to the repo root.
            prefix_abs = os.path.join(build_dir, *strip_include_prefix.split("/"))
        rel = os.path.relpath(prefix_abs, build_dir).replace(os.sep, "/")
        return [rel]

    def _normalize_label(self, src):
        # The base implementation treats any leading "/" as (the start of) a
        # Bazel "//"-rooted label and lstrip()s it unconditionally, mangling
        # genuine absolute filesystem paths (a single leading "/", never
        # valid Bazel label syntax) -- which is exactly what
        # _redirect_overlay_paths() injects for submodule-only sources.
        # Pass those through untouched.
        if src.startswith("/") and not src.startswith("//"):
            return src
        return super()._normalize_label(src)

    def _split_select_defines(self, defines):
        # The base converter only supports select()-based `deps` (via
        # _convert_platform_select_deps), not `defines` -- but this repo's
        # tracy-profiler defines (both the ARM-host hexagon library and the
        # DSP-side hexagon_dsp_skel binary) are select()-based. Rather than
        # fighting the base defines_block's unconditional string-quoting
        # (which would mangle a "${VAR}"-style multi-value reference), pull
        # any ConditionSelect out so the caller can emit the target
        # unconditionally and follow it with a guarded
        # target_compile_definitions() for the select()-only part.
        if isinstance(defines, bazel_to_cmake_converter.ConditionSelect):
            return None, defines
        return defines, None

    def _emit_extra_defines(
        self,
        name,
        extra_defines,
        target_compatible_with,
        alwayslink=False,
        header_only=False,
    ):
        if extra_defines is None:
            return
        # The target this refers to only exists inside its own
        # target_compatible_with guard (e.g. hexagon_dsp_skel only exists
        # when IREE_HEXAGON_MLIR_DSP_BUILD is set) -- wrap with the exact
        # same guard so this doesn't reference an undefined target in a
        # configure where that guard is false (iree_add_all_subdirs()
        # descends into every subdirectory regardless of which toolchain
        # tree is being configured).
        #
        # target_compile_definitions() is a bare CMake builtin, so it needs
        # the real, package-name (not package-namespace "::" alias) target
        # name -- add_custom_command's add_dependencies() et al happily
        # accept an ALIAS target, but target_compile_definitions() flatly
        # rejects one ("target_compile_definitions can not be used on an
        # ALIAS target"), and ${_PACKAGE_NS}::name is always an ALIAS
        # (iree_cc_library.cmake's iree_add_alias_library() call) pointing
        # at ${_PACKAGE_NAME}_name.
        #
        # Which keyword applies to that real target further depends on how
        # iree_cc_library.cmake actually built it:
        # - ALWAYSLINK: ${_PACKAGE_NAME}_name itself is an INTERFACE target
        #   propagating objects; the real compiled objects (and thus the
        #   only place a PRIVATE define can apply) live in its ".objects"
        #   twin instead.
        # - header-only (no srcs): ${_PACKAGE_NAME}_name IS an INTERFACE
        #   target directly (no ".objects" twin at all) -- only the
        #   INTERFACE keyword is legal on it.
        # - otherwise: a normal STATIC/SHARED target -- PRIVATE applies
        #   directly.
        self._emit_platform_guard_begin(target_compatible_with)
        self._converter.body += "iree_package_name(_PACKAGE_NAME)\n"
        if alwayslink:
            self._converter.body += (
                f"# ALWAYSLINK: ${{_PACKAGE_NAME}}_{name} itself is an INTERFACE\n"
                "# target; the real compiled objects live in its \".objects\" twin.\n"
            )
            target_name = "${_PACKAGE_NAME}_" + name + ".objects"
            keyword = "PRIVATE"
        elif header_only:
            target_name = "${_PACKAGE_NAME}_" + name
            keyword = "INTERFACE"
        else:
            target_name = "${_PACKAGE_NAME}_" + name
            keyword = "PRIVATE"
        for label, values in extra_defines.conditions.items():
            if label == "//conditions:default" or not values:
                continue
            cond = self._convert_select_condition(label)
            defs = " ".join(values)
            self._converter.body += (
                f"if({cond})\n"
                f"  target_compile_definitions({target_name} {keyword} {defs})\n"
                f"endif()\n\n"
            )
        self._emit_platform_guard_end(target_compatible_with)

    # Unlike every packaging/platform-alias target (already no-ops above),
    # neither //plugins/runtime/hexagon:hexagon nor
    # //plugins/runtime/hexagon/registration:DriverModuleLib (which depends on
    # it) carries any target_compatible_with in Bazel -- both are only ever
    # actually built under an explicit --platforms=//platform:aarch64_android
    # transition (nothing in this repo ever requests them any other way).
    # CMake's add_subdirectory()+iree_cc_library() is eager, though: every
    # configure tree that pulls in this repo (including the plain
    # host/compiler tree) would otherwise try to define (and `ninja all`
    # would try to link) these ARM-only libraries, which transitively need
    # the Android-arch-specific hexagon_sdk::cdsprpc_android_aarch64 import.
    # Guard with a CMake-only option with no Bazel-side equivalent (see root
    # CMakeLists.txt).
    _ARM_ONLY_LIBRARIES = {
        ("plugins/runtime/hexagon", "hexagon"),
        ("plugins/runtime/hexagon/registration", "DriverModuleLib"),
    }

    def _is_arm_hexagon_top_library(self, name):
        build_dir = self._build_dir
        if not os.path.isabs(build_dir):
            build_dir = os.path.join(self._repo_root, build_dir)
        build_dir = os.path.normpath(build_dir)
        for rel_dir, lib_name in self._ARM_ONLY_LIBRARIES:
            if name != lib_name:
                continue
            expected = os.path.normpath(
                os.path.join(self._repo_root, *rel_dir.split("/"))
            )
            if build_dir == expected:
                return True
        return False

    def cc_library(
        self,
        name=None,
        hdrs=None,
        srcs=None,
        textual_hdrs=None,
        defines=None,
        target_compatible_with=None,
        alwayslink=None,
        includes=None,
        strip_include_prefix=None,
        **kwargs,
    ):
        android_only = self._is_arm_hexagon_top_library(name)
        if android_only:
            self._converter.body += "if(IREE_HEXAGON_ANDROID_BUILD)\n"
        defines, extra_defines = self._split_select_defines(defines)
        extra_includes = self._includes_for_strip_include_prefix(strip_include_prefix)
        if extra_includes:
            includes = list(includes or []) + extra_includes
        super().cc_library(
            name=name,
            hdrs=self._redirect_overlay_paths(hdrs),
            srcs=self._redirect_overlay_paths(srcs),
            textual_hdrs=self._redirect_overlay_paths(textual_hdrs),
            defines=defines,
            target_compatible_with=target_compatible_with,
            alwayslink=alwayslink,
            includes=includes,
            **kwargs,
        )
        if android_only:
            self._converter.body += "endif()\n\n"
            self._converter.body += "if(IREE_HEXAGON_ANDROID_BUILD)\n"
        self._emit_extra_defines(
            name,
            extra_defines,
            target_compatible_with,
            alwayslink=bool(alwayslink),
            header_only=not bool(srcs),
        )
        if android_only:
            self._converter.body += "endif()\n\n"

    # plugins/runtime/hexagon/interface/ uses a custom QAIC-codegen rule
    # (hexagon_interface) sliced via Bazel OutputGroupInfo into per-group
    # filegroups -- both are no-op in this converter (see hexagon_interface()
    # and pkg_files-family below) since that whole directory's CMakeLists.txt
    # is hand-written instead (the *_stub.c/*_skel.c outputs are real
    # compilable sources, not just header-only ordering deps, which the
    # generic filegroup-as-stamp-file conversion can't express). The one
    # consumer outside that directory -- hexagon_dsp_skel's cc_binary here --
    # needs its srcs entry redirected to the real generated file path (see
    # plugins/runtime/hexagon/interface/CMakeLists.txt for where it's
    # produced) plus an explicit build-order dependency on the custom target
    # that generates it.
    _HEXAGON_DSP_SKEL_C_LABEL = "//plugins/runtime/hexagon/interface:hexagon_dsp_skel_c"
    _HEXAGON_DSP_SKEL_C_PATH = (
        "${CMAKE_CURRENT_BINARY_DIR}/../interface/hexagon_dsp/hexagon_dsp_skel.c"
    )

    def cc_binary(
        self,
        name=None,
        srcs=None,
        defines=None,
        target_compatible_with=None,
        linkshared=None,
        **kwargs,
    ):
        defines, extra_defines = self._split_select_defines(defines)
        needs_interface_gen_dep = bool(srcs) and self._HEXAGON_DSP_SKEL_C_LABEL in srcs
        if needs_interface_gen_dep:
            srcs = [
                self._HEXAGON_DSP_SKEL_C_PATH if s == self._HEXAGON_DSP_SKEL_C_LABEL else s
                for s in srcs
            ]
        srcs = self._redirect_overlay_paths(srcs)
        if linkshared:
            self._emit_shared_cc_binary(
                name=name,
                srcs=srcs,
                defines=defines,
                extra_defines=extra_defines,
                target_compatible_with=target_compatible_with,
                **kwargs,
            )
        else:
            super().cc_binary(
                name=name,
                srcs=srcs,
                defines=defines,
                target_compatible_with=target_compatible_with,
                **kwargs,
            )
            self._emit_extra_defines(name, extra_defines, target_compatible_with)
        if needs_interface_gen_dep:
            # add_dependencies() rejects an ALIAS target outright ("Cannot
            # add target-level dependencies to alias target ...") -- unlike
            # target_compile_definitions() et al, which happily accept one
            # (see _emit_extra_defines()). Needs the real, package-name
            # (not package-namespace/alias) target name.
            self._emit_platform_guard_begin(target_compatible_with)
            self._converter.body += (
                f"iree_package_name(_PACKAGE_NAME)\n"
                f"add_dependencies(${{_PACKAGE_NAME}}_{name} hexagon_dsp_interface_gen)\n\n"
            )
            self._emit_platform_guard_end(target_compatible_with)

    def _emit_shared_cc_binary(
        self,
        name,
        srcs=None,
        copts=None,
        deps=None,
        defines=None,
        extra_defines=None,
        includes=None,
        target_compatible_with=None,
        **kwargs,
    ):
        # The base converter's cc_binary() (bazel_to_cmake_converter.py) has
        # no concept of Bazel's linkshared=True at all: it accepts (and
        # silently drops, via **kwargs) the flag and always emits a plain
        # iree_cc_binary(), never a shared object. CMake's iree_cc_library()
        # macro is what actually knows how to opt a single target into
        # SHARED while the rest of the build stays STATIC (see
        # iree_cc_library.cmake and its SHARED keyword) -- hexagon_dsp_skel
        # (the only linkshared=True cc_binary in this repo) needs exactly
        # that, so translate it to a SHARED iree_cc_library() instead of
        # silently producing a non-shared iree_cc_binary() that leaves the
        # DSP-side FastRPC .so entirely unbuilt in its expected shared form.
        if self._should_skip_target(**kwargs):
            return
        name_block = self._convert_string_arg_block("NAME", name, quote=False)
        srcs_block = self._convert_srcs_block(srcs)
        copts_block = self._convert_string_list_block("COPTS", copts, sort=False)
        deps_block, platform_deps_block = self._convert_platform_select_deps(name, deps)
        defines_block = self._convert_string_list_block("DEFINES", defines)
        includes_block = self._convert_includes_block(includes)

        self._emit_platform_guard_begin(target_compatible_with)
        if platform_deps_block:
            self._converter.body += platform_deps_block
        self._converter.body += (
            f"iree_cc_library(\n"
            f"{name_block}"
            f"{srcs_block}"
            f"{copts_block}"
            f"{deps_block}"
            f"{defines_block}"
            f"{includes_block}"
            f"  SHARED\n)\n\n"
        )
        # iree_cc_library() names the CMake target (and, by default, its
        # output file) after the full package path, unlike a bare Bazel
        # binary name -- restore the plain name so packaging steps that
        # expect exactly "lib<name>.so" (matching what Bazel's cc_binary
        # would have produced) keep working.
        self._converter.body += (
            f"iree_package_name(_PACKAGE_NAME)\n"
            f"set_target_properties(${{_PACKAGE_NAME}}_{name} PROPERTIES\n"
            f'  OUTPUT_NAME "{name}"\n'
            f")\n"
        )
        self._emit_platform_guard_end(target_compatible_with)

        if extra_defines is not None:
            self._emit_platform_guard_begin(target_compatible_with)
            self._converter.body += "iree_package_name(_PACKAGE_NAME)\n"
            for label, values in extra_defines.conditions.items():
                if label == "//conditions:default" or not values:
                    continue
                cond = self._convert_select_condition(label)
                defs = " ".join(values)
                # cond already ANDs in IREE_ENABLE_RUNTIME_TRACING for the
                # tracy label (see _convert_platform_condition) -- without
                # it, iree_tracing_context_t (used unconditionally once
                # IREE_HAL_HEXAGON_ENABLE_PROFILER is defined) would be
                # referenced without ever being declared in a
                # tracing-disabled configure.
                self._converter.body += (
                    f"if({cond})\n"
                    f"  target_compile_definitions(${{_PACKAGE_NAME}}_{name} PRIVATE {defs})\n"
                    f"endif()\n\n"
                )
            self._emit_platform_guard_end(target_compatible_with)

    def filegroup(self, name, srcs, **kwargs):
        super().filegroup(name, self._redirect_overlay_paths(srcs), **kwargs)

    def iree_lit_test(
        self,
        name,
        cfg,
        test_file=None,
        tools=None,
        data=None,
        timeout=None,
        tags=None,
        target_compatible_with=None,
        **kwargs,
    ):
        # Only iree_lit_test_suite has an existing handler; this repo also
        # has 3 standalone iree_lit_test targets (the readelf-dependent,
        # Linux-only tests). Maps to CMake's iree_lit_test() the same way
        # iree_lit_test_suite() maps to iree_lit_test_suite() -- both defined
        # in build_tools/cmake/iree_lit_test.cmake.
        if self._should_skip_target(tags=tags, **kwargs):
            return
        name_block = self._convert_string_arg_block("NAME", name, quote=False)
        test_file_block = self._convert_string_arg_block("TEST_FILE", test_file)
        tools_block = self._convert_target_list_block("TOOLS", tools)
        data_block = self._convert_target_list_block("DATA", data)
        labels_block = self._convert_string_list_block("LABELS", tags)
        timeout_block = self._convert_timeout_arg_block("TIMEOUT", timeout)

        self._emit_platform_guard_begin(target_compatible_with)
        self._converter.body += (
            f"iree_lit_test(\n"
            f"{name_block}"
            f"{test_file_block}"
            f"{tools_block}"
            f"{data_block}"
            f"{labels_block}"
            f"{timeout_block}"
            f")\n\n"
        )
        self._emit_platform_guard_end(target_compatible_with)

    # The flatcc driver binary has no plain-library CMake target under the
    # name the base target map gives it (bazel_to_cmake_targets.py maps
    # "@com_github_dvidelabs_flatcc//:flatcc" to the "flatcc" *library*
    # target, not the "iree-flatcc-cli" tool binary that this genrule's
    # $(location)/tools actually need) -- and, unlike an ordinary library
    # dep, it needs cross-compile-aware host-tool resolution, mirroring
    # IREE core's own build_tools/cmake/flatbuffer_c_library.cmake.
    _FLATCC_TOOL_LABEL = "@com_github_dvidelabs_flatcc//:flatcc"
    _FLATCC_TOOL_VAR = "IREE_FLATCC_TOOL_BINARY"

    def genrule(self, name, srcs=None, outs=None, cmd=None, tools=None, **kwargs):
        # No existing bazel_to_cmake handler for plain genrule (IREE itself
        # only ever uses its own higher-level macros). This repo has exactly
        # one: flatcc schema codegen in plugins/runtime/hexagon/schemas. Only
        # the two Make-variable substitutions that genrule actually uses
        # ($(location <label>), $(RULEDIR)) are implemented -- this is not a
        # general Bazel genrule interpreter.
        srcs = srcs or []
        tools = tools or []
        uses_flatcc = self._FLATCC_TOOL_LABEL in tools

        def _location(label):
            if label in srcs:
                return f"${{CMAKE_CURRENT_SOURCE_DIR}}/{label}"
            if label == self._FLATCC_TOOL_LABEL:
                return "${" + self._FLATCC_TOOL_VAR + "}"
            return self._targets.convert_target(label)[0]

        resolved_cmd = cmd.replace("$(RULEDIR)", "${CMAKE_CURRENT_BINARY_DIR}")
        resolved_cmd = re.sub(
            r"\$\(location ([^)]+)\)", lambda m: _location(m.group(1)), resolved_cmd
        )
        outs_block = "\n".join(f'    "${{CMAKE_CURRENT_BINARY_DIR}}/{o}"' for o in outs)
        srcs_block = "\n".join(
            f'    "${{CMAKE_CURRENT_SOURCE_DIR}}/{s}"' for s in srcs
        )
        tool_targets = [
            "${" + self._FLATCC_TOOL_VAR + "}" if t == self._FLATCC_TOOL_LABEL
            else self._targets.convert_target(t)[0]
            for t in tools
        ]
        tools_block = "\n".join(f"    {t}" for t in tool_targets)

        flatcc_setup = ""
        if uses_flatcc:
            # Cross-compile-aware host-tool resolution (mirrors
            # flatbuffer_c_library.cmake in IREE core): natively, the plain
            # target name is used directly as a COMMAND token below, which
            # CMake resolves to that executable target's build output at
            # generate time; when cross-compiling, IREE_HOST_BIN_DIR points
            # at the already-built host tools tree instead.
            flatcc_setup = (
                f"set({self._FLATCC_TOOL_VAR} iree-flatcc-cli)\n"
                f"if(IREE_HOST_BIN_DIR)\n"
                f'  set({self._FLATCC_TOOL_VAR} "${{IREE_HOST_BIN_DIR}}/iree-flatcc-cli")\n'
                f"endif()\n"
            )
            # The tool must be its own COMMAND token (not embedded in a
            # single shell string) for CMake's target-name-to-output-path
            # substitution to apply in the native (non-cross-compiling)
            # case -- so split into argv instead of the generic
            # `sh -c <whole cmd>` used below. Plain CMake double-quoting
            # (not shlex/shell quoting -- there is no shell here) is enough:
            # none of these argv tokens contain embedded double quotes.
            command_block = "\n".join(
                f'    "{arg}"' for arg in shlex.split(resolved_cmd)
            )
            command = f"  COMMAND\n{command_block}\n"
        else:
            command = f"  COMMAND sh -c {shlex.quote(resolved_cmd)}\n"

        self._converter.body += (
            f"{flatcc_setup}"
            f"add_custom_command(\n"
            f"  OUTPUT\n{outs_block}\n"
            f"{command}"
            f"  DEPENDS\n{srcs_block}\n{tools_block}\n"
            f"  VERBATIM\n"
            f")\n"
            f"add_custom_target({name}_gen DEPENDS\n{outs_block}\n)\n\n"
        )

    def hexagon_plugin_library(self, deps=None, hdr_deps=None, **kwargs):
        # Bazel-side, hexagon_plugin_library (build_tools/bazel/defs.bzl) wraps
        # iree_cc_library forcing linkstatic+alwayslink, and threads hdr_deps
        # through cc_headers_only so they're compile-time-only (satisfying
        # Bazel's layering_check without re-linking those deps' objects).
        # CMake has no equivalent layering enforcement and no per-target
        # static/shared linkage knob (that's a global CMAKE_BUILD_TYPE-ish
        # concern), so this simplifies to a plain ALWAYSLINK cc_library with
        # hdr_deps folded into deps.
        all_deps = list(deps or []) + list(hdr_deps or [])
        self.cc_library(deps=all_deps, alwayslink=True, **kwargs)

    # runtime_lib (build_tools/bazel/overlays/hexagon_mlir/qcom_hexagon_backend/
    # bin/runtime) is the one hexagon_mlir_overlay_library() target also built
    # in the DSP-only CMake tree (cmake/HexagonToolchain.cmake,
    # IREE_BUILD_COMPILER=OFF), where LLVM/MLIR itself is never configured at
    # all. Its hdr_deps=["@llvm-project//mlir:LLVMSupportHeaders"] (a handful
    # of llvm::StringMap/ADT includes used by multithreading/StringMap.cpp
    # etc.) has no real CMake target to link against there -- every other
    # hexagon_mlir_overlay_library() target only builds in the host/compiler
    # tree (IREE_BUILD_COMPILER=ON), where LLVM is already configured and the
    # generic hdr_deps-folded-into-deps handling below resolves fine. Swap in
    # the copts/include-path equivalent instead of a link dependency on a
    # target that doesn't exist in that tree (build_tools/cmake/
    # hexagon_llvm_headers_stub stands in for the handful of ADT headers
    # actually needed).
    _RUNTIME_LIB_NO_LLVM_TREE_COPTS = [
        "-Wno-unused-variable",
        "-DLLVM_DISABLE_ABI_BREAKING_CHECKS_ENFORCING=1",
        "-flax-vector-conversions",
    ]
    _RUNTIME_LIB_NO_LLVM_TREE_INCLUDES = [
        "../../../../../../../third-party/hexagon-mlir/qcom_hexagon_backend/bin/runtime/include",
        "../../../../../../../build_tools/cmake/hexagon_llvm_headers_stub",
        "../../../../../../../third-party/iree/third_party/llvm-project/llvm/include",
    ]

    def hexagon_mlir_overlay_library(
        self, name=None, deps=None, hdr_deps=None, copts=None, includes=None, **kwargs
    ):
        is_runtime_lib_no_llvm_tree = name == "runtime_lib" and hdr_deps
        if is_runtime_lib_no_llvm_tree:
            all_deps = list(deps or [])
            copts = list(copts or []) + self._RUNTIME_LIB_NO_LLVM_TREE_COPTS
            includes = list(includes or []) + self._RUNTIME_LIB_NO_LLVM_TREE_INCLUDES
            # Pure CMake install/export config with no Bazel-side equivalent
            # (Bazel has no concept of an install export set or component) --
            # runtime_lib is depended on by hexagon_rt_library and
            # hexagon_dsp_skel outside its own package, so it needs to be
            # installed/exported like any other PUBLIC library, in the same
            # "Runtime" export set as the rest of this repo's runtime
            # libraries. Must precede the iree_cc_library() call itself
            # (these variables are read at ADD-time, not appended
            # afterward), so this can't live below the preserved-content
            # marker.
            self._converter.body += (
                "# runtime_lib is a PUBLIC library depended on by "
                "hexagon_rt_library and\n"
                "# hexagon_dsp_skel\n"
                'set(IREE_INSTALL_LIBRARY_TARGETS_DEFAULT_EXPORT_SET "Runtime")\n'
                'set(IREE_INSTALL_LIBRARY_TARGETS_DEFAULT_COMPONENT "IREEDevLibraries-Runtime")\n'
            )
        else:
            all_deps = list(deps or []) + list(hdr_deps or [])
        self.cc_library(
            name=name, deps=all_deps, copts=copts, includes=includes, **kwargs
        )

    def pkg_files(self, *args, **kwargs):
        # rules_pkg packaging + platform_aliases.bzl transitions have no
        # CMake-macro equivalent to autogenerate from -- the actual zip
        # packaging (hexagon_runtime_aarch64_android(_tracy).zip,
        # device_tools_aarch64_android.zip) is hand-written directly
        # (cmake/PackageHexagonRuntime.cmake, invoked from content below the
        # BAZEL_TO_CMAKE_PRESERVES marker in these same BUILD.bazel-derived
        # files), mirroring the file lists in the underlying BUILD.bazel by
        # hand rather than by macro conversion.
        pass

    def pkg_zip(self, *args, **kwargs):
        pass

    def copy_file(self, *args, **kwargs):
        pass

    def aarch64_android_platform_alias(self, *args, **kwargs):
        pass

    def aarch64_android_tracy_platform_alias(self, *args, **kwargs):
        pass

    def hexagon_platform_alias(self, *args, **kwargs):
        pass

    def td_library(self, *args, **kwargs):
        # Mirrors iree_td_library: no CMake equivalent needed, tablegen
        # targets declare their own .td sources/deps directly.
        pass

    def cc_import(self, name, static_library=None, shared_library=None, **kwargs):
        # No existing bazel_to_cmake handler for cc_import (IREE doesn't need
        # one internally). Bazel's cc_import(static_library=<label>) wraps a
        # prebuilt archive/shared object as a linkable target; the CMake
        # equivalent is an IMPORTED library target. The wrapped library lives
        # in an external SDK repo (@hexagon_sdk) fetched/exposed by
        # cmake/HexagonSDK.cmake (see plan Phase 2), which is expected to
        # define a same-named IMPORTED target directly -- so here we just
        # need an ALIAS from the plugin's local package-qualified name to
        # that externally-defined imported target.
        cmake_name = self._targets.convert_target(f":{name}")[0]
        if static_library is not None:
            aliased = self._targets.convert_target(static_library)[0]
        elif shared_library is not None:
            aliased = self._targets.convert_target(shared_library)[0]
        else:
            raise NotImplementedError(
                f"cc_import({name}) needs static_library or shared_library"
            )
        # convert_target() resolves a package-relative ":name" reference to
        # the literal placeholder "::name" (see _convert_to_cmake_path) --
        # correct only inside an iree_cc_library()/iree_cc_binary() DEPS
        # list, where the macro itself calls iree_package_ns() internally to
        # resolve the "::" prefix at CMake-configure time. This emits a bare
        # add_library() instead, which has no such resolution, so it needs
        # the same iree_package_ns(_PACKAGE_NS) bootstrap plus the expanded
        # "${_PACKAGE_NS}::name" spelled out explicitly.
        if cmake_name.startswith("::"):
            self._converter.body += "iree_package_ns(_PACKAGE_NS)\n"
            cmake_name = "${_PACKAGE_NS}" + cmake_name
        self._converter.body += (
            f"add_library({cmake_name} ALIAS {aliased})\n\n"
        )

    def glob(self, include, exclude=None, exclude_directories=1):
        # The base glob() refuses "**" patterns outright (see its comment: no
        # in-tree uses at the time it was written). The hexagon-mlir overlay
        # does have one (Conversion/**/*.h, forwarding every *ToLLVM
        # conversion pass's public headers) -- Bazel's glob() has package-
        # boundary-crossing subtleties for "**", but this overlay tree has no
        # nested Bazel packages under Conversion/, so plain recursion is
        # equivalent. Reimplement using file(GLOB_RECURSE ...) for any "**"
        # pattern, delegating non-recursive patterns to the base behavior.
        if not any("**" in p for p in include) and not any(
            "**" in p for p in (exclude or [])
        ):
            return super().glob(include, exclude=exclude, exclude_directories=exclude_directories)
        if exclude_directories != 1:
            self._convert_unimplemented_function("glob", "with exclude_directories")
        exclude = exclude or []

        def _glob_var(pattern):
            return "_GLOB_" + (
                pattern.replace("*", "X").replace(".", "_").replace("/", "_").upper()
            )

        def _emit(pattern, var, mode):
            self._converter.body += (
                f"file({mode} {var} LIST_DIRECTORIES false"
                f" RELATIVE {self._expand_cmake_var('CMAKE_CURRENT_SOURCE_DIR')}"
                f" CONFIGURE_DEPENDS {pattern})\n"
            )

        glob_vars = []
        for pattern in include:
            var = _glob_var(pattern)
            glob_vars.append(var)
            _emit(pattern, var, "GLOB_RECURSE" if "**" in pattern else "GLOB")
        for pattern in exclude:
            exclude_var = _glob_var(pattern)
            _emit(pattern, exclude_var, "GLOB_RECURSE" if "**" in pattern else "GLOB")
            for glob_var in glob_vars:
                self._converter.body += (
                    f"list(REMOVE_ITEM {glob_var} "
                    f"{self._expand_cmake_var(exclude_var)})\n"
                )
        return [self._expand_cmake_var(var) for var in glob_vars]

    def gentbl_cc_library(self, tbl_outs=None, td_file=None, td_srcs=None, **kwargs):
        # This repo's @llvm-project pin uses the newer tblgen.bzl convention
        # where tbl_outs is a dict {"Out.h.inc": ["-flag", ...], ...}; the
        # bazel_to_cmake converter still expects the older
        # [(["-flag", ...], "Out.h.inc"), ...] list-of-tuples form. Normalize.
        if isinstance(tbl_outs, dict):
            tbl_outs = [(flags, out) for out, flags in tbl_outs.items()]
        # td_file/td_srcs need the same overlay->submodule redirection as
        # cc_library's hdrs/srcs (see _redirect_overlay_paths): the hexagon-
        # mlir overlay's .td sources physically live in the submodule too.
        if td_file is not None:
            (td_file,) = self._redirect_overlay_paths([td_file])
        td_srcs = self._redirect_overlay_paths(td_srcs)
        super().gentbl_cc_library(
            tbl_outs=tbl_outs, td_file=td_file, td_srcs=td_srcs, **kwargs
        )

    def _convert_platform_condition(self, constraint_label):
        # Custom constraint_values defined in //constraints (no @platforms
        # equivalent for the Hexagon DSP CPU/QuRT OS). Both are only ever
        # true together (target_compatible_with lists both), and both are
        # only true when configured with cmake/HexagonToolchain.cmake, so
        # both map to the same guard variable that toolchain file sets.
        if constraint_label.endswith(
            ("//constraints:cpu_hexagon", "//constraints:os_qurt")
        ):
            return "IREE_HEXAGON_MLIR_DSP_BUILD"
        # Tracy-enabled config_setting. IREE_TRACING_PROVIDER is only
        # meaningful when IREE_ENABLE_RUNTIME_TRACING is on -- mirrors how
        # IREE's own generated runtime/src/iree/base/tracing/CMakeLists.txt
        # nests its "tracy" branch inside `if(IREE_ENABLE_RUNTIME_TRACING)`;
        # without the AND, a IREE_TRACING_PROVIDER=tracy + tracing-disabled
        # config would still define IREE_HAL_HEXAGON_ENABLE_PROFILER and
        # reference iree_tracing_context_t, which is only actually declared
        # once IREE_ENABLE_RUNTIME_TRACING pulls in iree/base/tracing/tracy.h.
        if constraint_label.endswith(
            "runtime/src/iree/base/tracing:_tracy_enable"
        ):
            return 'IREE_ENABLE_RUNTIME_TRACING AND IREE_TRACING_PROVIDER STREQUAL "tracy"'
        # target_compatible_with = select({"@android_ndk_detect//:android_ndk_available":
        # [], "//conditions:default": ["@platforms//:incompatible"]}) gates
        # Bazel targets in a single unified build graph that spans every
        # platform at once. This repo's CMake port instead only ever
        # add_subdirectory()s these targets from the dedicated Android
        # cross-build tree (Phase 4) in the first place, so the guard is
        # unconditionally true there and irrelevant elsewhere.
        if constraint_label.endswith("android_ndk_detect//:android_ndk_available"):
            return "TRUE"
        # Gates a packaging-only copy_file() (already a no-op, see
        # pkg_files/pkg_zip/copy_file above); same rationale as
        # android_ndk_available.
        if constraint_label.endswith(
            "plugins/runtime/hexagon:hexagon_runtime_enabled"
        ):
            return "TRUE"
        # Bazel --define=hexagon_playground_profiler=true opt-in flag with no
        # CMake equivalent yet; map to a dedicated cache option (declared in
        # plugins/iree_runtime_plugin.cmake), defaulting to the same
        # effective default (disabled).
        if constraint_label.endswith(":hexagon_playground_profiler_enabled"):
            return "IREE_HEXAGON_PLAYGROUND_PROFILER"
        return super()._convert_platform_condition(constraint_label)

    def hexagon_interface(self, *args, **kwargs):
        # Custom rule (build_tools/bazel/hexagon_interface.bzl) invoking the
        # Hexagon SDK's `qaic` RPC-stub generator, consumed via Bazel
        # OutputGroupInfo-sliced filegroup() references (header/stub/skel)
        # that have no bazel_to_cmake equivalent either. This whole directory
        # (plugins/runtime/hexagon/interface/) is hand-written instead (Phase
        # 3: QAIC add_custom_command + plain cc_library targets), excluded
        # from autogeneration entirely (no autogeneration header) -- this
        # no-op only exists so the recursive conversion walk doesn't crash
        # while processing sibling directories in the same pass.
        pass


class CustomTargetConverter(bazel_to_cmake_targets.TargetConverter):
    # Package root for the hexagon-mlir overlay CMake tree (Phase 1): a
    # hand-written build_tools/bazel/overlays/hexagon_mlir/CMakeLists.txt
    # calls iree_setup_c_src_root(PACKAGE_ROOT_PREFIX "hexagon_mlir") before
    # add_subdirectory(qcom_hexagon_backend), so every target defined at or
    # below qcom_hexagon_backend/ gets namespace "hexagon_mlir[::<relpath>]".
    _HEXAGON_MLIR_PREFIX = "@hexagon-mlir//qcom_hexagon_backend"

    def _initialize(self):
        self._update_target_mappings(
            {
                # The base converter only maps the bare "@llvm-project//lld"
                # package (to library sub-targets like COFF/ELF/MachO), not
                # this specific ":ld.lld" binary sub-target that the plugin's
                # lit tests reference directly as a tool.
                "@llvm-project//lld:ld.lld": ["${IREE_LLD_TARGET}"],
                # MLIR's Bazel overlay (third_party/llvm-project/utils/bazel/
                # llvm-project-overlay/mlir/BUILD.bazel) names this library
                # "ConvertToLLVM" (wrapping
                # lib/Conversion/ConvertToLLVM/ConvertToLLVMPass.cpp), but
                # MLIR's own CMakeLists.txt still calls the actual library
                # MLIRConvertToLLVMPass -- a Bazel/CMake naming mismatch
                # upstream, not something the base converter's generic
                # "MLIR" + name fallback can get right on its own.
                "@llvm-project//mlir:ConvertToLLVM": ["MLIRConvertToLLVMPass"],
                # Hexagon SDK (fetched/exposed by cmake/HexagonSDK.cmake).
                "@hexagon_sdk//:qaic": ["hexagon_sdk::qaic"],
                "@hexagon_sdk//:incs_tree": [],  # consumed via HEXAGON_SDK_INCS_TREE_DIR, not a link dep
                "@hexagon_sdk//:cdsprpc_android_aarch64": [
                    "hexagon_sdk::cdsprpc_android_aarch64"
                ],
                "@hexagon_sdk//:qurt_headers": ["hexagon_sdk::qurt_headers"],
                "@hexagon_sdk//:hexagon_toolchain_bit_headers": [
                    "hexagon_sdk::hexagon_toolchain_bit_headers"
                ],
                "@hexagon_sdk//:qhl_hvx_headers": ["hexagon_sdk::qhl_hvx_headers"],
                "@hexagon_sdk//:rtos/qurt/computev68/lib/libqurt.a": [
                    "hexagon_sdk::qurt_lib"
                ],
                # HexKL (fetched/exposed by cmake/HexKL.cmake).
                "@hexkl//:hexkl_micro_v79": ["hexkl::hexkl_micro_v79"],
            }
        )

    def convert_target(self, target):
        if target.startswith(self._HEXAGON_MLIR_PREFIX):
            return self._convert_hexagon_mlir_target(target)
        return super().convert_target(target)

    def _convert_hexagon_mlir_target(self, target: str):
        # "@hexagon-mlir//qcom_hexagon_backend:Foo" -> "hexagon_mlir::Foo"
        # "@hexagon-mlir//qcom_hexagon_backend/lib/Common:HexagonCommon" ->
        #     "hexagon_mlir::lib::Common::HexagonCommon"
        rest = target[len(self._HEXAGON_MLIR_PREFIX) :]
        path_part, _, name_part = rest.rpartition(":")
        path_part = path_part.strip("/")
        ns = "hexagon_mlir"
        if path_part:
            ns += "::" + path_part.replace("/", "::")
        return [f"{ns}::{name_part}"]

    def _convert_unmatched_target(self, target: str) -> str:
        # _convert_iree_core_target() only special-cases compiler/src/iree/*,
        # runtime/src/iree/* and tools/* and otherwise falls through to here
        # -- e.g. "@iree//compiler/plugins/target/LLVMCPU:LinkerTool". Mirror
        # IREE's own convention (prefix with "iree::", don't prune the path)
        # for those rather than treating them as this repo's own targets.
        iree_core_repo = self._repo_alias("@iree_core")
        if iree_core_repo and target.startswith(f"{iree_core_repo}//"):
            stripped = target[len(iree_core_repo) :]
            return ["iree::" + self._convert_to_cmake_path(stripped)]
        # Otherwise: this repo's own cross-package targets, e.g.
        # "//plugins/runtime/hexagon/arm_dsp:hexagon_arm_dsp".
        return ["iree_hexagon_plugins::" + self._convert_to_cmake_path(target)]

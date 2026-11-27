# Copyright 2026 RooflineAI GmbH
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Import and compile on the host while tests run on the device.

The device stage (push, run, pull) stays serial, but import and compile only
need the host. At the start of the session, the imports and compiles of all
collected tests are queued, in test order, to one worker thread, so the host
work runs ahead of the device work. The worker does one import or compile at a
time.

A thread is enough: the main thread mostly waits on adb, and the import and
iree-compile run in other processes.

The imports run in one long-lived child process, so torch is loaded and warmed
up once per session. For each import the child points its stdout and stderr
(fds 1 and 2) at the model's `import.log`, which catches everything the import
prints, from Python, warnings, progress bars and C++. The first test that reads
the import prints that log.
The child is replaced once its memory grows past _RECYCLE_RSS_BYTES, and after a
crash.

Imports are shared per (model, whether the weights are external) and compiles
per (model, compile case, weights mode), so the run cases of one compile case
share one module. With external weights, the compile job also encodes them
(`stages/encoding.py`) and hashes the result, so that this too overlaps the
device work.
"""

from __future__ import annotations

import dataclasses
import gc
import multiprocessing
import os
import pathlib
import sys
import time
import traceback
from collections.abc import Sequence
from concurrent.futures import Future, ProcessPoolExecutor, ThreadPoolExecutor
from concurrent.futures.process import BrokenProcessPool

import pytest

from integration_tests.modeltree.spec import ModelSpec
from integration_tests.stages.artifacts import ImportedModel, ParameterArchive
from integration_tests.stages.compiling import CompileCase, compile_model
from integration_tests.stages.encoding import encoder_output_flags, parameter_archive

# import does not release some memory back to the os
# memory held by the import process may conflict with the memory needed for iree-compile step
# The import process is replaced when it holds more than this after an import.
_RECYCLE_RSS_BYTES = 4 << 30
# having a fresh import process for every import would need high setup costs e.g. loading torch


class ModelImportError(Exception):
    """The import failed; the message carries its output."""


@dataclasses.dataclass(frozen=True)
class Compiled:
    """A compiled module, with the import it was compiled from."""

    # The inputs and references the run and check need.
    imported: ImportedModel
    vmfb: pathlib.Path
    # iree-compile's time for the module, without the encoding.
    seconds: float
    # What iree-compile printed (iree-compile.log).
    log: str
    # The weights the module loads at runtime, for the `external` weights
    # mode; None otherwise, or when the model has no weights.
    parameters: ParameterArchive | None = None
    # Compiling and running the encoder, and hashing its output.
    encode_seconds: float = 0.0


@dataclasses.dataclass(frozen=True)
class Import:
    model: ImportedModel
    # Everything the import printed, stdout and stderr interleaved.
    log: str
    seconds: float


def _rss_bytes() -> int:
    with open("/proc/self/statm") as statm:
        return int(statm.read().split()[1]) * os.sysconf("SC_PAGE_SIZE")


def _import_in_child(
    name: str, output_dir: pathlib.Path, externalize: bool
) -> tuple[ImportedModel | None, str, int]:
    """Runs in the import child: (the import or None, its log, the child's RSS after importing)."""
    # Imported here, so that only the child loads torch.
    from integration_tests.modeltree import discover
    from integration_tests.stages.importing import import_model

    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "import.log"
    imported = None
    sys.stdout.flush()
    sys.stderr.flush()
    saved = os.dup(1), os.dup(2)
    try:
        with open(log_path, "w") as log:
            os.dup2(log.fileno(), 1)
            os.dup2(log.fileno(), 2)
            try:
                (spec,) = discover.select(names=[name])
                imported = import_model(spec, output_dir, externalize=externalize)
            except Exception:
                traceback.print_exc()
            finally:
                sys.stdout.flush()
                sys.stderr.flush()
    finally:
        os.dup2(saved[0], 1)
        os.dup2(saved[1], 2)
        os.close(saved[0])
        os.close(saved[1])

    gc.collect()
    return imported, log_path.read_text(errors="replace"), _rss_bytes()


class Prefetcher:
    def __init__(
        self,
        items: Sequence[pytest.Item],
        iree_compile: pathlib.Path,
        iree_encode_parameters: pathlib.Path,
        lld: pathlib.Path | None,
        tmp_path_factory: pytest.TempPathFactory,
    ) -> None:
        self._iree_compile = iree_compile
        self._iree_encode_parameters = iree_encode_parameters
        self._lld = lld
        self._tmp_path_factory = tmp_path_factory
        # the worker thread.
        # Only 1 worker is allowed:
        # - Memory: two LLM imports or compiles at once can exhaust the host
        # - _compile relies on the queue order: a compile is queued after its
        #   import, so with one worker the import is done when it starts.
        # - _import_pool is created and replaced by the worker without a lock.
        self._executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="prefetch"
        )
        # The import child; created and replaced by the worker thread only.
        self._import_pool: ProcessPoolExecutor | None = None
        # imports already queued, by (model, external weights)
        self._imports: dict[tuple[str, bool], Future[Import]] = {}
        # Imports whose output a test has printed already.
        self._import_logged: set[tuple[str, bool]] = set()
        # Compiles whose output a test has printed already.
        self._compile_logged: set[tuple[str, str, str]] = set()
        # models already compiled, by (model, compile case, weights mode)
        self._compiles: dict[tuple[str, str, str], Future[Compiled]] = {}
        # Queue every test, in test order. The first import overlaps with the
        # session setup (deploying RT to the device). Queue is worked sequentially.
        for item in items:
            self._prefetch(item)

    def compiled(
        self, spec: ModelSpec, compile_case: CompileCase, weights: str = "embedded"
    ) -> Compiled:
        """Look up the compile that __init__ queued and wait for it; returns the
        compiled module with its import.

        Raises CompilationError, or ModelImportError if the import failed.
        Raises KeyError, if this compile case was not queued
        Prints the import's output the first time a test asks for this model,
        and iree-compile's output, the first time a test asks for this compile.
        """
        key = (spec.name, compile_case.name, weights)
        compiled = self._compiles[key]
        try:
            # raise compile or import error on failure
            result = compiled.result()
        finally:
            # print import log, even if compile raised
            import_key = (spec.name, weights == "external")
            if import_key not in self._import_logged:
                self._import_logged.add(import_key)
                # we know its done: the compile waited for it.
                imported = self._imports[import_key]
                if imported.exception() is None:
                    done = imported.result()
                    print(
                        f"{spec.name}: imported ({weights} weights) in "
                        f"{done.seconds:.2f}s, output:"
                    )
                    print(done.log)
        # Only on success: print compile time and compile log (although probably empty)
        # a failed compile's output is in its CompilationError.
        if key not in self._compile_logged:
            self._compile_logged.add(key)
            if result.log.strip():
                print(
                    f"{spec.name} [{compile_case.name}, {weights} weights]: "
                    "iree-compile output:"
                )
                print(result.log)
        return result

    def close(self) -> None:
        # Cancels everything still queued, e.g. after -x
        # shutdown needs to wait for the currently running job
        self._executor.shutdown(wait=True, cancel_futures=True)
        if self._import_pool is not None:
            self._import_pool.shutdown(wait=True)

    def _prefetch(self, item: pytest.Item) -> None:
        callspec = getattr(item, "callspec", None)
        if callspec is None or item.get_closest_marker("skip") is not None:
            return
        spec = callspec.params.get("model_spec")
        compile_case = callspec.params.get("compile_case")
        weights = callspec.params.get("weights", "embedded")
        if spec is not None and compile_case is not None:
            self._submit_compile(spec, compile_case, weights)

    # The two _submit_* queue their work unless it is queued already, so tests
    # sharing a model share its import, and tests sharing a (model, compile
    # case) share its compile.
    def _submit_import(self, spec: ModelSpec, externalize: bool) -> Future[Import]:
        key = (spec.name, externalize)
        imported = self._imports.get(key)
        if imported is None:
            suffix = "_external" if externalize else ""
            imported = self._executor.submit(
                self._import,
                spec,
                externalize,
                self._tmp_path_factory.mktemp(spec.name.replace("/", "_") + suffix),
            )
            self._imports[key] = imported
        return imported

    def _submit_compile(
        self, spec: ModelSpec, compile_case: CompileCase, weights: str
    ) -> Future[Compiled]:
        key = (spec.name, compile_case.name, weights)
        compiled = self._compiles.get(key)
        if compiled is None:
            # get or queue the relevant import
            imported = self._submit_import(spec, weights == "external")
            suffix = "_external" if weights == "external" else ""
            compiled = self._executor.submit(
                self._compile,
                spec,
                compile_case,
                weights,
                imported,
                self._tmp_path_factory.mktemp(
                    f"{spec.name.replace('/', '_')}_{compile_case.name}{suffix}"
                ),
            )
            self._compiles[key] = compiled
        return compiled

    def _import(
        self, spec: ModelSpec, externalize: bool, output_dir: pathlib.Path
    ) -> Import:
        """The import job: runs on the worker thread, queued by _submit_import."""
        if self._import_pool is None:
            self._import_pool = self._new_import_pool()
        start = time.time()
        # manage the import process, detects if its still alive
        try:
            future = self._import_pool.submit(
                _import_in_child, spec.name, output_dir, externalize
            )
        except BrokenProcessPool:
            self._import_pool = self._new_import_pool()  # re-start
            # an error occurring with a fresh import process is checked below as it belongs to this import
            future = self._import_pool.submit(
                _import_in_child, spec.name, output_dir, externalize
            )
        # wait for completion
        try:
            model, log, rss = future.result()
        except BrokenProcessPool as err:
            # The child died during this import.
            # e.g. killed for running out of memory
            self._import_pool = None  # Re-start for next import
            log_path = output_dir / "import.log"
            log = log_path.read_text(errors="replace") if log_path.exists() else ""
            raise ModelImportError(
                f"the import process died while importing {spec.name} ({err}); "
                f"its output so far:\n{log}"
            ) from err
        if rss > _RECYCLE_RSS_BYTES:
            # stop if it has clobbered too much mem
            self._import_pool.shutdown(wait=True)
            self._import_pool = None
        if model is None:
            raise ModelImportError(f"importing {spec.name} failed:\n{log}")
        return Import(model, log, time.time() - start)

    @staticmethod
    def _new_import_pool() -> ProcessPoolExecutor:
        # spawn, not fork: this process has threads.
        return ProcessPoolExecutor(
            max_workers=1, mp_context=multiprocessing.get_context("spawn")
        )

    def _compile(
        self,
        spec: ModelSpec,
        compile_case: CompileCase,
        weights: str,
        imported: Future[Import],
        work_dir: pathlib.Path,
    ) -> Compiled:
        """The compile job: runs on the worker thread, queued by _submit_compile."""
        # Submitted after its import to the single worker, so already done.
        # Raises the import's ModelImportError if it failed.
        model = imported.result().model
        encoder_mlir = work_dir / "encoder.mlirbc"
        # A stale encoder from an earlier compile into the same directory would
        # otherwise pass for this compile's.
        encoder_mlir.unlink(missing_ok=True)
        external = weights == "external"
        start = time.time()
        vmfb = compile_model(
            iree_compile=self._iree_compile,
            model_mlir=model.model_mlir,
            output_vmfb=work_dir / "model.vmfb",
            compile_case=compile_case,
            log_file=work_dir / "iree-compile.log",
            linker=self._lld,
            model_flags=spec.extra_compile_flags,
            extra_flags=tuple(encoder_output_flags(encoder_mlir)) if external else (),
        )
        seconds = time.time() - start
        log = (work_dir / "iree-compile.log").read_text(errors="replace")
        if not external:
            return Compiled(model, vmfb, seconds, log)

        start = time.time()
        if model.weights_archive is None:
            if encoder_mlir.exists():
                raise ModelImportError(
                    f"{spec.name}: the import for external weights wrote no archive"
                )
            parameters = None
        else:
            parameters = parameter_archive(
                iree_compile=self._iree_compile,
                iree_encode_parameters=self._iree_encode_parameters,
                encoder_mlir=encoder_mlir,
                weights_archive=model.weights_archive,
                output_archive=work_dir / "model.irpa",
                linker=self._lld,
            )
        return Compiled(model, vmfb, seconds, log, parameters, time.time() - start)

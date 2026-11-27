# Copyright 2026 RooflineAI GmbH
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Importing a model from the tree to MLIR, plus its reference outputs.

  * torch program -> torch-dialect MLIR: `torch.export.export`, a
    `run_decompositions` with the table built below, then
    `iree.turbine.aot.export`.
  * torch-dialect MLIR -> IREE input: This repo's iree-compile registers
    the `input_torch` plugin (see
    `build_tools/bazel/default_compiler_plugins.bzl`), so
    `--iree-input-type=auto` does it as part of the compile.

The exported entry point is named `main`, which is iree-turbine's default.
A model that `get_model()` returns as a `KvCacheLM` (`models/kv_cache.py`) is
exported as two functions instead, `prefill` and `decode`. It needs a
`generation` block in its model.yaml: instead of a reference per function, the
import records the model's tokenizer, and the prefill logits and tokens an
eager greedy loop generates from the prompt, for iree-run-llm on the device to
reproduce.
"""

from __future__ import annotations

import dataclasses
import importlib.metadata
import importlib.util
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from iree.turbine import aot
from iree.turbine.aot import current_aot_decompositions
from torch._decomp import get_decompositions
from torch._export.utils import (
    _collect_all_valid_cia_ops_for_aten_namespace,
    _get_decomp_for_cia,
)

from integration_tests.models.kv_cache import KvCacheLM
from integration_tests.modeltree.spec import InputSpec, ModelSpec
from integration_tests.stages.artifacts import (
    GenerationReference,
    ImportedModel,
    ModuleCall,
)

_aten = torch.ops.aten

MODEL_MLIR_NAME = "model.mlirbc"

_TORCH_DTYPES: dict[str, torch.dtype] = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "int64": torch.int64,
    "int32": torch.int32,
    "bool": torch.bool,
}

# Ops that must be decomposed before the importer sees them, because torch-mlir
# either cannot lower them or lowers them badly.
DEFAULT_DECOMPOSITIONS = [
    _aten.conv1d,
    _aten.conv1d.default,
    _aten.conv2d,
    _aten.conv2d.default,
    _aten.conv3d,
    _aten.conv3d.default,
    # hardtanh likewise gets decomposed to a clamp.
    _aten.hardtanh,
    _aten.hardtanh.default,
    # Since torch 2.6 linear is no longer decomposed automatically, and
    # torch-mlir's own decomposition turns it into a batch_matmul with batch
    # size 1 rather than a matmul.
    _aten.linear.default,
    _aten.matmul.default,
    _aten.split,
    _aten.split_with_sizes.default,
    _aten.concat.default,
    # Attention: keep the math form rather than the fused CPU kernels, which
    # have no torch-mlir lowering.
    _aten.scaled_dot_product_attention.default,
    _aten._scaled_dot_product_attention_math.default,
    _aten._scaled_dot_product_flash_attention_for_cpu.default,
    _aten._safe_softmax,
    # Attention masks are built from a boolean new_ones ANDed with the causal
    # mask. Left alone, new_ones imports as a torch.vtensor.literal of
    # tensor<i1> and __and__ as torch.aten.__and__.Tensor, neither of which
    # torch-to-iree can legalize.
    _aten.new_ones,
    _aten.new_ones.default,
    # The boolean AND the mask is built with. CompositeImplicitAutograd, so it
    # only decomposes through the synthesis in build_decomposition_table below -
    # torch's own tables have no entry for it.
    _aten.__and__.Tensor,
    _aten._weight_norm.default,
    _aten._weight_norm_interface,
    _aten.diff.default,
    _aten.gru,
    _aten.lstm,
    _aten.multiply.Tensor,
    _aten.norm_except_dim.default,
    _aten.upsample_bicubic2d,
    _aten.upsample_bilinear2d,
    _aten.upsample_nearest2d,
]

# Recorded in the manifest beside every import, so that "which decompositions
# produced this MLIR" is answerable after the fact.
DECOMPOSITION_NAMES = tuple(sorted(str(op) for op in DEFAULT_DECOMPOSITIONS))


def manifest(spec: ModelSpec) -> dict[str, Any]:
    """Everything that determined this MLIR, for the record."""
    return {
        "model": spec.name,
        "dtype": spec.dtype,
        "inputs": [dataclasses.asdict(entry) for entry in spec.inputs],
        "decompositions": list(DECOMPOSITION_NAMES),
        "versions": _tool_versions(),
    }


def _tool_versions() -> dict[str, str]:
    """Versions of everything between the model definition and the MLIR."""
    versions: dict[str, str] = {}
    for distribution in ("torch", "transformers", "iree-turbine"):
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[distribution] = "absent"
    return versions


def build_decomposition_table(ops: list[Any]) -> dict[Any, Any]:
    """A decomposition table that actually covers `ops`.

    Three sources, because none alone is enough:

      * iree-turbine's own default table, which is what `aot.export` would have
        applied on its own.
      * `get_decompositions(ops)` for ops with a registered decomposition
        (`aten.new_ones`, `aten._safe_softmax`);
      * `_get_decomp_for_cia(op)` for CompositeImplicitAutograd ops, which have
        no registry entry but do have an implementation the exporter can be told
        to inline (`aten.conv2d.default`, `aten.__and__.Tensor`).
    """
    table = dict(current_aot_decompositions())
    table.update(get_decompositions(ops))
    cia_ops = _collect_all_valid_cia_ops_for_aten_namespace()
    for op in ops:
        if op in cia_ops:
            table[op] = _get_decomp_for_cia(op)
    return table


def _load_model_source(spec: ModelSpec) -> Any:
    """The spec's `model.py`, as a Python module."""
    source = spec.model_source
    module_spec = importlib.util.spec_from_file_location(
        f"hexagon_it_model_{spec.name.replace('/', '_').replace('-', '_')}", source
    )
    if module_spec is None or module_spec.loader is None:
        raise ImportError(f"cannot load {source}")
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return module


def load_model(spec: ModelSpec) -> torch.nn.Module:
    """Call `get_model()` from the spec's `model.py`."""
    source = spec.model_source
    module = _load_model_source(spec)
    if not hasattr(module, "get_model"):
        raise ImportError(f"{source} does not define get_model()")
    model = module.get_model()
    model.eval()
    return model


def _resolve_high(input_spec: InputSpec, model: torch.nn.Module) -> int:
    """The exclusive upper bound for an integral generator."""
    if input_spec.high is not None:
        return input_spec.high
    vocab_size = getattr(getattr(model, "config", None), "vocab_size", None)
    if vocab_size is None:
        raise ValueError(
            f"input dtype {input_spec.dtype} needs an upper bound: either set "
            "'high' in model.yaml or use a model whose config has vocab_size"
        )
    return int(vocab_size)


def _load_image(spec: ModelSpec, input_spec: InputSpec) -> torch.Tensor:
    """Decode a PNG beside model.yaml into the spec's NCHW shape, scaled 0..1.

    RGB, HWC->NCHW, uint8/255 - with no ImageNet mean/std normalization.
    It is what their resnet-50 test asserts a label against, so keeping
    it identical keeps the two results comparable. It also means the image
    has to already be the right size; no resizing happens here.
    """
    from PIL import Image

    assert input_spec.image is not None
    path = spec.directory / input_spec.image
    if not path.exists():
        raise FileNotFoundError(f"{spec.name}: no such image {path}")
    array = np.array(Image.open(path).convert("RGB"))
    array = np.expand_dims(array.transpose([2, 0, 1]), axis=0)
    tensor = torch.from_numpy(array.astype(np.float32) / 255.0)
    if tuple(tensor.shape) != tuple(input_spec.shape):
        raise ValueError(
            f"{spec.name}: {path.name} decodes to {tuple(tensor.shape)}, but the "
            f"spec says {tuple(input_spec.shape)}. Resize the image; this "
            "importer deliberately does not."
        )
    return tensor.to(_TORCH_DTYPES[input_spec.dtype])


def label_map(model: torch.nn.Module) -> dict[int, str] | None:
    """The model's own id -> label mapping, if it has one."""
    id2label = getattr(getattr(model, "config", None), "id2label", None)
    if not id2label:
        return None
    return {int(key): str(value) for key, value in id2label.items()}


def generate_inputs(
    spec: ModelSpec, model: torch.nn.Module
) -> tuple[torch.Tensor, ...]:
    """Build the input tensors the spec describes, deterministically."""
    tensors = []
    for input_spec in spec.inputs:
        dtype = _TORCH_DTYPES[input_spec.dtype]
        INTEGRAL = ("int64", "int32", "bool")
        generator = torch.Generator().manual_seed(input_spec.seed)
        if input_spec.generator == "image":
            tensor = _load_image(spec, input_spec)
        elif input_spec.generator == "zeros":
            tensor = torch.zeros(input_spec.shape, dtype=dtype)
        elif input_spec.generator == "ones":
            tensor = torch.ones(input_spec.shape, dtype=dtype)
        elif input_spec.generator == "normal":
            if input_spec.dtype in INTEGRAL:
                raise ValueError(
                    f"{spec.name}: unexpected input generator {input_spec.generator!r} "
                    f"for dtype {input_spec.dtype!r}"
                )
            tensor = torch.randn(
                input_spec.shape, generator=generator, dtype=torch.float32
            ).to(dtype)
        elif input_spec.generator == "uniform":
            if input_spec.dtype in INTEGRAL:
                high = 2 if dtype is torch.bool else _resolve_high(input_spec, model)
                tensor = torch.randint(
                    input_spec.low, high, input_spec.shape, generator=generator
                ).to(dtype)
            else:
                raise ValueError(
                    f"{spec.name}: unexpected input generator {input_spec.generator!r} "
                    f"for dtype {input_spec.dtype!r}"
                )
        else:
            raise ValueError(
                f"{spec.name}: unknown input generator {input_spec.generator!r} "
                f"for dtype {input_spec.dtype!r}; known: "
                f"normal, uniform, ones, zeros, image"
            )

        tensors.append(tensor)

    return tuple(tensors)


def _as_tensor_tuple(outputs: Any) -> tuple[torch.Tensor, ...]:
    if hasattr(outputs, "items"):  # transformers ModelOutput
        outputs = tuple(value for _, value in outputs.items())
    elif not isinstance(outputs, (tuple, list)):
        outputs = (outputs,)
    for output in outputs:
        if not isinstance(output, torch.Tensor):
            raise TypeError(f"model returned a non-tensor output: {type(output)}")
    return tuple(outputs)


def import_model(
    spec: ModelSpec,
    output_dir: Path,
    decompositions: list[Any] | None = None,
) -> ImportedModel:
    """Export the spec's model to MLIR and record its inputs and eager outputs.

    Writes `model.mlirbc`, `input{N}.npy` and `output{N}.npy` into
    `output_dir`.
    The outputs come from running the model eagerly in torch on the same
    inputs.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    model = load_model(spec)
    inputs = generate_inputs(spec, model)
    if isinstance(model, KvCacheLM):
        return _import_kv_cache(spec, output_dir, model, inputs, decompositions)
    if spec.generation is not None:
        raise ValueError(
            f"{spec.name}: 'generation' needs a get_model() that returns a "
            f"KvCacheLM, not a {type(model).__name__}"
        )

    # The eager reference is taken before the export, because aot.export can
    # mutate the module.
    with torch.no_grad():
        outputs = _as_tensor_tuple(model(*inputs))
    output_files = []
    for index, tensor in enumerate(outputs):
        output_file = output_dir / f"output{index}.npy"
        np.save(output_file, tensor.detach().cpu().numpy())
        output_files.append(output_file)

    input_files = []
    for index, tensor in enumerate(inputs):
        input_file = output_dir / f"input{index}.npy"
        np.save(input_file, tensor.detach().cpu().numpy())
        input_files.append(input_file)

    ops = decompositions if decompositions is not None else DEFAULT_DECOMPOSITIONS
    # Non-strict:
    # The strict export traces with Dynamo, which errors out on Python if the graph
    # would not capture faithfully (globals, side effects, non-torch calls),
    # holds another copy of the weights while it runs (duplicating memory
    # usage). The non-strict export runs the Python as is and bakes plain Python
    # values (config numbers, shape arithmetic) into the graph unchecked. If an
    # error happens, a bug will still happen and the verification against the
    # eager reference will show a mismatch (or compilation will fail). In order
    # to limit memory usage in the CI and when executing integration tests, we
    # use a non-strict export, which should be acceptable in these previously
    # manually tested models.
    exported_program = torch.export.export(model, args=inputs, strict=False)
    exported_program = exported_program.run_decompositions(
        build_decomposition_table(ops)
    )
    # aot.export takes the decomposed program from here; iree-turbine still
    # applies its own default table on top, which is where the batch-norm and
    # _to_copy decompositions a conv net needs come from.
    exported = aot.export(exported_program)

    model_mlir = output_dir / MODEL_MLIR_NAME
    exported.save_mlir(model_mlir)

    labels = label_map(model)
    if labels is not None:
        # Recorded so the semantic check can name a class without importing
        # torch or re-downloading the model config.
        (output_dir / "labels.json").write_text(json.dumps(labels, indent=2))

    (output_dir / "import.json").write_text(
        json.dumps(
            {**manifest(spec), "model_mlir_bytes": model_mlir.stat().st_size},
            indent=2,
            default=str,
        )
    )

    return ImportedModel(
        model_mlir=model_mlir,
        calls=[ModuleCall("main", input_files, output_files)],
        labels_file=(output_dir / "labels.json") if labels is not None else None,
    )


def _import_kv_cache(
    spec: ModelSpec,
    output_dir: Path,
    model: KvCacheLM,
    inputs: tuple[torch.Tensor, ...],
    decompositions: list[Any] | None,
) -> ImportedModel:
    """Export `prefill` and `decode` as functions of one module.

    `inputs`, a prompt, only give the example arguments of the export. The
    cache becomes module state, initially zero. The functions are not checked
    one by one but together, by iree-run-llm against the generation reference.
    """
    if spec.generation is None:
        raise ValueError(
            f"{spec.name}: a KvCacheLM needs a 'generation' block, which is how "
            "its prefill and decode are checked"
        )
    (prompt,) = inputs
    length = prompt.shape[-1]
    block = torch.export.Dim("block_dim", min=2)
    seq_len = torch.export.Dim("seq_len", min=1, max=model.max_seq_len)
    prefill_args = (prompt, torch.arange(length), torch.tensor([length - 1]))
    prefill_shapes = {
        "input_ids": {1: model.block_seq_stride * block},
        "cache_position": {0: model.block_seq_stride * block},
        "input_length": {},
    }
    decode_shapes = {"input_ids": {}, "cache_position": {0: seq_len}}

    # The prompt's last token, decoded right after it.
    decode_args = (prompt[:, -1:], torch.arange(length + 1))
    # The eager reference is taken before the export, which can mutate the
    # module.
    generation = _generation_reference(spec, output_dir, model)
    functions = {
        "prefill": (prefill_args, prefill_shapes),
        "decode": (decode_args, decode_shapes),
    }

    builder = aot.FxProgramsBuilder(model)
    for name, (args, dynamic_shapes) in functions.items():
        # Non-strict: dynamo cannot trace through transformers' model code.
        builder.export_program(
            getattr(KvCacheLM, name),
            name=name,
            args=args,
            dynamic_shapes=dynamic_shapes,
            strict=False,
        )
    # export_program applies only iree-turbine's default table; apply the
    # suite's, as the single-function import does.
    ops = decompositions if decompositions is not None else DEFAULT_DECOMPOSITIONS
    table = build_decomposition_table(ops)
    for program in builder.programs.values():
        program.target = program.target.run_decompositions(table)
    exported = aot.export(builder, import_symbolic_shape_expressions=True)

    model_mlir = output_dir / MODEL_MLIR_NAME
    exported.save_mlir(model_mlir)
    (output_dir / "import.json").write_text(
        json.dumps(
            {
                **manifest(spec),
                "functions": list(functions),
                "model_mlir_bytes": model_mlir.stat().st_size,
            },
            indent=2,
            default=str,
        )
    )
    return ImportedModel(
        model_mlir=model_mlir,
        calls=[],
        generation=generation,
    )


# iree-run-llm's defaults for padding the prompt: at least two blocks, padded
# with token 0. Passed to it explicitly, so that the two cannot drift apart.
_MIN_PREFILL_BLOCKS = 2
_PAD_TOKEN_ID = 0


class NearTieError(ValueError):
    """The reference generation has a near-tie: the prompt is unsuitable."""


def _generation_reference(
    spec: ModelSpec, output_dir: Path, model: KvCacheLM
) -> GenerationReference:
    """The files iree-run-llm needs, and the tokens it has to generate.

    The prompt is tokenized with the HuggingFace `tokenizers` library from the
    same tokenizer.json the device gets, so the device's prompt tokens check its
    own tokenizer too. The reference is `KvCacheLM.generate_greedy`: the
    exported functions run eagerly, padded as on the device. Its prefill logits
    go to generation_prefill_logits.npy.

    Raises `NearTieError` if at some step the reference's most likely token
    leads the runner-up by less than the spec's `min_margin`: a rounding
    difference on the device could tip it, and the tokens after it could not be
    compared. The reference is written to generation_reference.json first.
    """
    from tokenizers import Tokenizer

    generation = spec.generation
    assert generation is not None
    module = _load_model_source(spec)
    if not hasattr(module, "get_tokenizer_json"):
        raise ImportError(
            f"{spec.model_source} does not define get_tokenizer_json(), which "
            "'generation' needs"
        )
    tokenizer_json = output_dir / "tokenizer.json"
    tokenizer_json.write_bytes(Path(module.get_tokenizer_json()).read_bytes())
    prompt_file = output_dir / "prompt.txt"
    prompt_file.write_text(generation.prompt)

    tokenizer = Tokenizer.from_file(str(tokenizer_json))
    prompt_ids = tokenizer.encode(generation.prompt).ids
    generated, prefill_logits = model.generate_greedy(
        prompt_ids,
        generation.max_new_tokens,
        min_prefill_blocks=_MIN_PREFILL_BLOCKS,
        pad_token_id=_PAD_TOKEN_ID,
    )
    prefill_logits_file = output_dir / "generation_prefill_logits.npy"
    np.save(prefill_logits_file, prefill_logits.detach().cpu().numpy())
    reference = GenerationReference(
        prompt_file=prompt_file,
        tokenizer_json=tokenizer_json,
        max_new_tokens=generation.max_new_tokens,
        max_seq_len=model.max_seq_len,
        block_seq_stride=model.block_seq_stride,
        min_prefill_blocks=_MIN_PREFILL_BLOCKS,
        pad_token_id=_PAD_TOKEN_ID,
        prompt_ids=tuple(prompt_ids),
        generated=tuple(generated),
        prefill_logits_file=prefill_logits_file,
    )
    (output_dir / "generation_reference.json").write_text(
        json.dumps(
            {
                key: value
                for key, value in dataclasses.asdict(reference).items()
                if key not in ("prompt_file", "tokenizer_json", "prefill_logits_file")
            },
            indent=2,
        )
    )
    for step, info in enumerate(generated):
        if info.margin >= generation.min_margin:
            continue
        shorten = f" or set max_new_tokens to at most {step}" if step > 0 else ""
        raise NearTieError(
            f"{spec.name}: near-tie in the reference generation, please change "
            f"the prompt{shorten}. At step {step}, token {info.selected_id} "
            f"({tokenizer.decode([info.selected_id])!r}) leads token "
            f"{info.runner_up_id} ({tokenizer.decode([info.runner_up_id])!r}) "
            f"by {info.margin:.3f}, less than min_margin={generation.min_margin:g}. "
            f"The margins are in {output_dir / 'generation_reference.json'}."
        )
    return reference

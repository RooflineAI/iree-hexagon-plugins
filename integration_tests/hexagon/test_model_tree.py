# Copyright 2026 RooflineAI GmbH
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Checks on the model tree that need no phone, no torch and no compiler."""

from __future__ import annotations

from integration_tests.hexagon.modeltree import discover
from integration_tests.hexagon.stages.compiling import CASE_NAMES


def test_every_model_yaml_parses() -> None:
    specs = discover.load_all()
    assert specs, "the model tree is empty"
    for spec in specs:
        assert spec.model_source.exists(), f"{spec.name}: no model.py"
        assert spec.inputs, f"{spec.name}: no inputs"
    print(f"model tree: {len(specs)} models in {discover.MODELS_DIR}")


def test_expected_outcomes_name_real_cases() -> None:
    """An outcome keyed on a mistyped case name would silently never apply."""
    for spec in discover.load_all():
        for outcome in spec.expected_outcomes:
            assert outcome.case == "*" or outcome.case in CASE_NAMES, (
                f"{spec.name}: expected_outcomes names case {outcome.case!r}, "
                f"which is not one of {CASE_NAMES}"
            )
            assert outcome.reason, f"{spec.name}: outcome with no reason"


def test_model_names_are_unique() -> None:
    names = [spec.name for spec in discover.load_all()]
    assert len(names) == len(set(names)), f"duplicate model names in {names}"


def test_tag_filtering() -> None:
    """The tag algebra, on the real tree: AND within, OR across."""
    hexagon = {spec.name for spec in discover.select(include=[("hexagon",)])}
    assert hexagon, "no model is tagged 'hexagon'"
    assert not discover.select(include=[("nonexistent-tag",)])
    without_llm = {
        spec.name
        for spec in discover.select(include=[("hexagon",)], exclude=[("llm",)])
    }
    assert without_llm <= hexagon


def test_compile_cases_name_real_cases() -> None:
    """A mistyped case name in compile_cases would skip the model entirely."""
    for spec in discover.load_all():
        for name in spec.compile_cases:
            assert name in CASE_NAMES, (
                f"{spec.name}: compile_cases names {name!r}, which is not one "
                f"of {CASE_NAMES}"
            )


def test_semantic_checks_have_an_image() -> None:
    """A label assertion on random noise would be meaningless, not just wrong."""
    for spec in discover.load_all():
        if spec.semantic_check is None:
            continue
        assert any(entry.generator == "image" for entry in spec.inputs), (
            f"{spec.name}: semantic_check needs an input with generator: image"
        )
        for entry in spec.inputs:
            if entry.image is not None:
                assert (spec.directory / entry.image).exists(), (
                    f"{spec.name}: missing image {entry.image}"
                )

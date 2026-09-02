# Copyright 2026 RooflineAI GmbH
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Known failures as data, checked against the evidence."""

from __future__ import annotations

import dataclasses
import enum
from typing import Any


class Status(enum.StrEnum):
    """Where a (model, compile case) pair ended up."""

    PASSED = "PASSED"
    COMPILE_FAILURE = "COMPILE_FAILURE"
    RUNTIME_FAILURE = "RUNTIME_FAILURE"
    ACCURACY_FAILURE = "ACCURACY_FAILURE"


_EXPECTABLE = tuple(status for status in Status if status is not Status.PASSED)
_EXPECTABLE_NAMES = [str(status) for status in _EXPECTABLE]


class Mismatch(enum.StrEnum):
    """The ways a run can disagree with what the model said would happen."""

    UNEXPECTED_PASS = "unexpected_pass"
    DIFFERENT_ERROR_TYPE = "different_error_type"
    REASON_NOT_FOUND = "reason_not_found"
    NO_XFAIL_REASON = "no_xfail_reason"


class OutcomeError(ValueError):
    pass


def parse_status(raw: Any) -> Status:
    """A `status:` field of a `model.yaml`, with a message naming the choices."""
    try:
        return Status(str(raw))
    except ValueError as err:
        raise OutcomeError(
            f"unknown status {raw!r}; one of {_EXPECTABLE_NAMES}"
        ) from err


@dataclasses.dataclass(frozen=True)
class ExpectedOutcome:
    """One `expected_outcomes` entry of a `model.yaml`."""

    status: Status
    reason: str
    # No `case` means the entry covers every compile case.
    case: str = "*"
    comment: str | None = None

    def __post_init__(self) -> None:
        if self.status is Status.PASSED:
            raise OutcomeError(
                f"case '{self.case}': PASSED is not an expected *outcome*; "
                "remove the entry"
            )
        # An entry with no reason is rejected at load time rather than at run time.
        if not self.reason.strip():
            raise OutcomeError(
                f"case '{self.case}': 'reason' is required and must be a "
                f"substring of the log the failure produces "
                f"({Mismatch.NO_XFAIL_REASON})"
            )
        object.__setattr__(self, "reason", self.reason.strip())


@dataclasses.dataclass(frozen=True)
class Outcome:
    """What actually happened, plus the log that proves it."""

    status: Status
    log: str

    @property
    def passed(self) -> bool:
        return self.status is Status.PASSED


@dataclasses.dataclass(frozen=True)
class MismatchReport:
    kind: Mismatch
    message: str


def check_outcome(expected: ExpectedOutcome, actual: Outcome) -> MismatchReport | None:
    """None if `actual` is the failure `expected` describes, else why not."""
    if actual.passed:
        return MismatchReport(
            Mismatch.UNEXPECTED_PASS,
            f"case '{expected.case}' is recorded as {expected.status} "
            f"({expected.reason!r}) but it passed. If that is a fix, delete the "
            "expected_outcomes entry from model.yaml.",
        )
    if actual.status is not expected.status:
        return MismatchReport(
            Mismatch.DIFFERENT_ERROR_TYPE,
            f"case '{expected.case}' is recorded as {expected.status} but it "
            f"failed as {actual.status}. Log:\n{_tail(actual.log)}",
        )
    if expected.reason not in actual.log:
        return MismatchReport(
            Mismatch.REASON_NOT_FOUND,
            f"case '{expected.case}' failed as {actual.status}, as recorded, "
            f"but for a different reason: {expected.reason!r} does not appear "
            f"in the log. Either the cause changed or the recorded reason is "
            f"stale. Log:\n{_tail(actual.log)}",
        )
    return None


def _tail(log: str, limit: int = 4000) -> str:
    """The end of a log, which is where compiler diagnostics put the point."""
    return log if len(log) <= limit else "...\n" + log[-limit:]

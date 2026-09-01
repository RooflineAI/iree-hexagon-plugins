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


class Mismatch(enum.StrEnum):
    """The ways a run can disagree with what the model said would happen."""

    UNEXPECTED_PASS = "unexpected_pass"
    DIFFERENT_ERROR_TYPE = "different_error_type"
    REASON_NOT_FOUND = "reason_not_found"
    NO_XFAIL_REASON = "no_xfail_reason"


class OutcomeError(ValueError):
    pass


@dataclasses.dataclass(frozen=True)
class ExpectedOutcome:
    """One `expected_outcomes` entry of a `model.yaml`."""

    case: str
    status: Status
    reason: str
    comment: str | None = None

    @classmethod
    def from_yaml(cls, where: str, data: dict[str, Any]) -> ExpectedOutcome:
        unknown = set(data) - {"case", "status", "reason", "comment"}
        if unknown:
            raise OutcomeError(f"{where}: unknown key(s) {sorted(unknown)}")
        if "status" not in data:
            raise OutcomeError(
                f"{where}: 'status' is required; one of {[str(s) for s in _EXPECTABLE]}"
            )
        try:
            status = Status(str(data["status"]))
        except ValueError as err:
            raise OutcomeError(
                f"{where}: unknown status {data['status']!r}; one of "
                f"{[str(s) for s in _EXPECTABLE]}"
            ) from err
        if status is Status.PASSED:
            raise OutcomeError(
                f"{where}: PASSED is not an expected *outcome*; remove the entry"
            )
        # An entry with no reason is exactly the hole this module exists to
        # close, so it is rejected at load time rather than at run time.
        reason = str(data.get("reason", "")).strip()
        if not reason:
            raise OutcomeError(
                f"{where}: 'reason' is required and must be a substring of the "
                f"log the failure produces ({Mismatch.NO_XFAIL_REASON})"
            )
        return cls(
            case=str(data.get("case", "*")),
            status=status,
            reason=reason,
            comment=None if data.get("comment") is None else str(data["comment"]),
        )


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

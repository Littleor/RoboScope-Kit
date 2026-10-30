# Copyright 2026 RoboScope Team
# SPDX-License-Identifier: Apache-2.0

"""Serializable records shared by benchmark controllers and worker processes."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class WorkItem:
    """One independently claimable unit of evaluation work."""

    ordinal: int
    item_id: str
    group_id: str
    payload: dict[str, Any]
    expected_results: int = 1

    def __post_init__(self) -> None:
        if self.ordinal < 0:
            raise ValueError("ordinal must be non-negative")
        if not self.item_id:
            raise ValueError("item_id cannot be empty")
        if not self.group_id:
            raise ValueError("group_id cannot be empty")
        if self.expected_results < 1:
            raise ValueError("expected_results must be positive")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> WorkItem:
        return cls(**payload)


@dataclass(frozen=True)
class EvaluationRecord:
    """One terminal result counted by the benchmark's official protocol."""

    record_id: str
    success: bool
    steps: int
    elapsed_s: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.record_id:
            raise ValueError("record_id cannot be empty")
        if self.steps < 0:
            raise ValueError("steps must be non-negative")
        if self.elapsed_s < 0:
            raise ValueError("elapsed_s must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> EvaluationRecord:
        return cls(**payload)


@dataclass(frozen=True)
class WorkResult:
    """Terminal output for one :class:`WorkItem`."""

    item_id: str
    group_id: str
    worker_id: int
    records: tuple[EvaluationRecord, ...]
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.item_id:
            raise ValueError("item_id cannot be empty")
        if not self.group_id:
            raise ValueError("group_id cannot be empty")
        if self.worker_id < 0:
            raise ValueError("worker_id must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["records"] = [record.to_dict() for record in self.records]
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> WorkResult:
        values = dict(payload)
        values["records"] = tuple(EvaluationRecord.from_dict(record) for record in values["records"])
        return cls(**values)


def validate_terminal_accounting(results: Sequence[WorkResult], expected: int) -> None:
    """Require exactly one terminal record for every expected episode."""

    record_ids = [record.record_id for result in results for record in result.records]
    if len(record_ids) != expected:
        raise RuntimeError(f"expected {expected} terminal records, received {len(record_ids)}")
    if len(record_ids) != len(set(record_ids)):
        raise RuntimeError("duplicate terminal record ids detected")

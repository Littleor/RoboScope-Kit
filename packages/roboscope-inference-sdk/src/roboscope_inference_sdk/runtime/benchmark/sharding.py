# Copyright 2026 RoboScope Team
# SPDX-License-Identifier: Apache-2.0

"""Deterministic episode sharding across GPU instances.

A sharded evaluation runs the same benchmark command on N independent instances.
Every instance generates the identical full work-item list, then keeps only the
items whose list position satisfies ``position % N == shard_index``. Because the
full list is deterministic, all instances agree on the partition without any
coordination, and the merged records are comparable to a single-instance run.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from .schema import WorkItem

DEFAULT_EPISODES_PER_SHARD = 500


def resolve_shard_count(total_records: int, episodes_per_shard: int) -> int:
    """Derive the shard count from the workload size (one shard per fixed episode budget)."""

    if total_records < 1:
        raise ValueError("total_records must be positive")
    if episodes_per_shard < 1:
        raise ValueError("episodes_per_shard must be positive")
    return max(1, -(-total_records // episodes_per_shard))


def select_shard_items(
    items: Sequence[WorkItem],
    shard_index: int,
    shard_count: int,
) -> list[WorkItem]:
    """Keep the deterministic subset of ``items`` owned by one shard."""

    if shard_count < 1:
        raise ValueError("shard_count must be positive")
    if not 0 <= shard_index < shard_count:
        raise ValueError(f"shard_index {shard_index} out of range for shard_count {shard_count}")
    selected = [item for position, item in enumerate(items) if position % shard_count == shard_index]
    if not selected:
        raise ValueError(
            f"shard {shard_index}/{shard_count} selected no work items "
            f"from a {len(items)}-item workload"
        )
    return selected


def shard_info(
    shard_index: int,
    shard_count: int,
    full_total_records: int,
    episodes_per_shard: int,
    explicit_count: bool,
    build: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Describe one shard run for manifest/result provenance.

    ``build`` is the image's SDK version and evaluation kit commit. Shards of one run must share
    it, because another build may partition or score the same workload differently.
    """

    return {
        "index": shard_index,
        "count": shard_count,
        "explicit_count": explicit_count,
        "episodes_per_shard": episodes_per_shard,
        "full_total_records": full_total_records,
        **({"build": dict(build)} if build is not None else {}),
    }


def plan_shard(
    items: Sequence[WorkItem],
    shard_index: int,
    shard_count: int | None,
    episodes_per_shard: int,
    build: Mapping[str, str] | None = None,
) -> tuple[list[WorkItem], dict[str, Any]]:
    """Resolve the shard count, select this shard's items and describe the partition.

    Every caller that needs to know what one shard will actually run goes through
    here, so the work items, the record total and the recorded provenance can
    never disagree with each other.
    """

    full_total = sum(item.expected_results for item in items)
    count = (
        shard_count
        if shard_count is not None
        else min(len(items), resolve_shard_count(full_total, episodes_per_shard))
    )
    if count > len(items):
        raise ValueError(f"shard_count {count} exceeds the {len(items)}-item workload")
    selected = select_shard_items(items, shard_index, count)
    metadata = shard_info(
        shard_index,
        count,
        full_total,
        episodes_per_shard,
        explicit_count=shard_count is not None,
        build=build,
    )
    return selected, metadata


__all__ = [
    "DEFAULT_EPISODES_PER_SHARD",
    "plan_shard",
    "resolve_shard_count",
    "select_shard_items",
    "shard_info",
]

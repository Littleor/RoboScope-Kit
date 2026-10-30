# Copyright 2026 RoboScope Team
# SPDX-License-Identifier: Apache-2.0

"""Cross-instance sharding: partition properties and the legacy contract.

The legacy contract is covered here because it is the one consumer that reads a
record total rather than the work items themselves. A sharded run that reports
the full workload there looks stalled for its whole life and then fails its
completeness check, which is exactly the defect the LIBERO pilot hit.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from roboscope_kit.runtime.benchmark.cli import _legacy_progress, _shard_kwargs
from roboscope_kit.runtime.benchmark.legacy import LegacyContract
from roboscope_kit.runtime.benchmark.schema import WorkItem
from roboscope_kit.runtime.benchmark.sharding import (
    plan_shard,
    resolve_shard_count,
    select_shard_items,
)


def _items(count: int, expected_results: int = 1) -> list[WorkItem]:
    return [
        WorkItem(
            ordinal=index,
            item_id=f"item-{index:06d}",
            group_id=f"group-{index // 5:04d}",
            payload={"index": index},
            expected_results=expected_results,
        )
        for index in range(count)
    ]


class _StubBenchmark:
    def __init__(self, items: list[WorkItem]) -> None:
        self._items = items

    def work_items(self) -> list[WorkItem]:
        return list(self._items)


class _StubFactory:
    def __init__(self, items: list[WorkItem]) -> None:
        self._items = items
        self.create_calls = 0

    def create(self, request: Any, manifest: Any) -> _StubBenchmark:
        del request, manifest
        self.create_calls += 1
        return _StubBenchmark(self._items)


class _StubRequest:
    """Only the field ``_legacy_progress`` reads off the run request."""

    suite = "libero_spatial"


@pytest.mark.parametrize("shard_count", [1, 2, 3, 4, 7])
def test_shards_partition_the_workload_exactly(shard_count: int) -> None:
    items = _items(50)
    seen: dict[str, int] = {}
    for shard_index in range(shard_count):
        for item in select_shard_items(items, shard_index, shard_count):
            assert item.item_id not in seen, "shards must be disjoint"
            seen[item.item_id] = shard_index
    assert set(seen) == {item.item_id for item in items}, "shards must cover the workload"


def test_shard_count_derives_one_shard_per_episode_budget() -> None:
    assert resolve_shard_count(500, 500) == 1
    assert resolve_shard_count(501, 500) == 2
    assert resolve_shard_count(2500, 500) == 5


def test_select_shard_items_rejects_an_out_of_range_index() -> None:
    with pytest.raises(ValueError, match="out of range"):
        select_shard_items(_items(10), 2, 2)


def test_select_shard_items_rejects_a_shard_with_no_work() -> None:
    with pytest.raises(ValueError, match="selected no work items"):
        select_shard_items(_items(2), 2, 3)


def test_plan_shard_reports_the_full_total_not_the_shard_total() -> None:
    selected, metadata = plan_shard(_items(50), 1, 2, 500)
    assert len(selected) == 25
    assert metadata == {
        "index": 1,
        "count": 2,
        "explicit_count": True,
        "episodes_per_shard": 500,
        "full_total_records": 50,
    }


def test_plan_shard_records_the_build_it_is_given() -> None:
    build = {"sdk_version": "0.7.0b47", "evaluation_kit_commit": "kit-a"}
    _, metadata = plan_shard(_items(50), 0, 2, 500, build=build)
    assert metadata["build"] == build


def test_a_shard_run_records_the_images_build() -> None:
    """The CLI hands the image manifest's identity to the shard, not the SDK's own idea of itself."""

    manifest = SimpleNamespace(sdk_version="0.7.0b47", evaluation_kit_commit="kit-a")
    args = SimpleNamespace(shard_index=1, shard_count=2, shard_episodes=500)
    kwargs = _shard_kwargs(args, manifest)
    assert kwargs["shard_build"] == {"sdk_version": "0.7.0b47", "evaluation_kit_commit": "kit-a"}
    assert _shard_kwargs(SimpleNamespace(shard_index=None), manifest) == {}


def test_plan_shard_derives_the_count_when_it_is_omitted() -> None:
    _, metadata = plan_shard(_items(1200), 0, None, 500)
    assert metadata["count"] == 3
    assert metadata["explicit_count"] is False


def test_legacy_total_counts_only_this_shards_records(tmp_path: Path) -> None:
    """A shard's legacy total is its own partition, never the whole workload."""

    factory = _StubFactory(_items(50, expected_results=2))
    contract = LegacyContract(tmp_path, "libero", "libero_spatial")
    progress = _legacy_progress(
        contract,
        factory,
        request=_StubRequest(),
        manifest=None,
        shard_kwargs={"shard_index": 0, "shard_count": 2, "shard_episodes": 500},
    )
    assert progress is not None
    # 25 of the 50 items, each carrying 2 expected records.
    assert progress.total == 50
    assert contract.total == 50


def test_legacy_total_covers_the_workload_across_every_shard(tmp_path: Path) -> None:
    items = _items(50, expected_results=2)
    totals = []
    for shard_index in range(2):
        progress = _legacy_progress(
            LegacyContract(tmp_path, "libero", "libero_spatial"),
            _StubFactory(items),
            request=_StubRequest(),
            manifest=None,
            shard_kwargs={"shard_index": shard_index, "shard_count": 2, "shard_episodes": 500},
        )
        assert progress is not None
        totals.append(progress.total)
    unsharded = _legacy_progress(
        LegacyContract(tmp_path, "libero", "libero_spatial"),
        _StubFactory(items),
        request=_StubRequest(),
        manifest=None,
        shard_kwargs={},
    )
    assert unsharded is not None
    assert sum(totals) == unsharded.total == 100


def test_legacy_progress_is_absent_without_a_contract() -> None:
    assert (
        _legacy_progress(
            None,
            _StubFactory(_items(4)),
            request=_StubRequest(),
            manifest=None,
            shard_kwargs={},
        )
        is None
    )

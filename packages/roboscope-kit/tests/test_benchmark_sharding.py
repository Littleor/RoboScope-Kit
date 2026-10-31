# Copyright 2026 RoboScope Team
# SPDX-License-Identifier: Apache-2.0

"""Cross-instance sharding: partition properties and the build each shard records."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from roboscope_kit.runtime.benchmark.cli import _shard_kwargs
from roboscope_kit.runtime.benchmark.schema import WorkItem
from roboscope_kit.runtime.benchmark.sharding import (
    plan_shard,
    resolve_shard_count,
    select_shard_items,
    shard_build,
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
    build = {"kit_version": "0.8.0b7", "manifest_sha256": "a" * 64}
    _, metadata = plan_shard(_items(50), 0, 2, 500, build=build)
    assert metadata["build"] == build


def test_a_shard_run_records_the_kit_version_and_manifest(tmp_path: Path) -> None:
    """The shard build changes with the benchmark manifest's content."""

    path = tmp_path / "benchmark-manifest.json"
    path.write_text("{}", encoding="utf-8")
    manifest = SimpleNamespace(path=path)
    args = SimpleNamespace(shard_index=1, shard_count=2, shard_episodes=500)
    build = _shard_kwargs(args, manifest)["shard_build"]
    assert build == shard_build(manifest)
    assert set(build) == {"kit_version", "manifest_sha256"}
    path.write_text('{"changed": true}', encoding="utf-8")
    assert shard_build(manifest)["manifest_sha256"] != build["manifest_sha256"]
    assert _shard_kwargs(SimpleNamespace(shard_index=None), manifest) == {}


def test_plan_shard_derives_the_count_when_it_is_omitted() -> None:
    _, metadata = plan_shard(_items(1200), 0, None, 500)
    assert metadata["count"] == 3
    assert metadata["explicit_count"] is False

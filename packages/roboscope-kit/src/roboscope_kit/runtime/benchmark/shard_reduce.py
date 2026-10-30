# Copyright 2026 RoboScope Team
# SPDX-License-Identifier: Apache-2.0

"""Reduce the ``result.json`` documents of sharded runs into one merged result.

``shard_merge`` re-aggregates from every shard's completed-record tree, which needs the
benchmark runtime on the machine doing the merge. This module works from the shard
``result.json`` documents alone: each shard already reports what it ran (episode counts, and
for benchmarks whose metric needs the whole run, raw partials from ``aggregate_partial``), so
merging is arithmetic plus a coverage check, with no simulator, factory or manifest involved.

Coverage is established by accounting instead of by re-listing the workload: the shard indexes
must be exactly ``0..N-1``, all shards must describe the same benchmark and suite, and their
record counts must add up to the ``full_total_records`` every shard recorded. Because the
partition is ``position % N``, a consistent index set plus a matching total leaves no room for
a missing or duplicated shard index. Result-only accounting cannot prove record-level
ownership or uniqueness; use record-tree merging for that stronger check.

Rate metrics (steps/s, elapsed) are never aggregated across instances.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from roboscope_kit.runtime.benchmarks.calvin.adapter import merge_sequence_lengths
from roboscope_kit.runtime.benchmarks.metaworld.adapter import merge_task_results

from .shard_validation import (
    require_count,
    shard_metadata,
    validate_shard_accounting,
    validate_shard_identity,
    validate_shard_set,
)

# Benchmarks whose official_result is a per-task table that can be summed row by row:
# name -> (row key, whether the result also carries runtime_metrics to sum).
_ADDITIVE_TABLES: dict[str, tuple[str, bool]] = {
    "libero": ("task_id", True),
    "libero_plus": ("task_id", True),
    "robotwin": ("task", False),
    "robocasa": ("task", False),
}
_SHARD_RATE_KEYS = ("elapsed_s", "steps_per_s", "episodes_per_s", "total_steps")


def merge_shard_results(shard_results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Merge the parsed ``result.json`` of every shard; raises ``ValueError`` on any inconsistency."""

    if not shard_results:
        raise ValueError("no shard results to merge")
    shard_meta = validate_shard_set(shard_results)
    ordered = sorted(shard_results, key=lambda shard: shard_metadata(shard)["index"])
    identity = validate_shard_identity(ordered)
    for shard in ordered:
        validate_shard_accounting(shard)
    standard = _merge_standard(ordered, shard_meta["full_total_records"])
    official = _merge_official(identity["benchmark"], [shard["official_result"] for shard in ordered])
    for shard in ordered:
        _validate_partial_counts(identity["benchmark"], shard)
    _validate_official_counts(identity["benchmark"], official, standard)
    return {
        "sdk_version": 1,
        "merged": True,
        "merge_method": "shard-result-reduce",
        "shard": {
            "count": shard_meta["count"],
            "indexes": [shard_metadata(shard)["index"] for shard in ordered],
            "episodes_per_shard": shard_meta["episodes_per_shard"],
            "build": shard_meta["build"],
        },
        "benchmark": ordered[0]["benchmark"],
        "result": standard,
        "official_result": official,
        "shards": [_shard_summary(shard) for shard in ordered],
        "validation": {
            "total_records": shard_meta["full_total_records"],
            "coverage": "accounted",
        },
    }


def _merge_standard(shards: Sequence[Mapping[str, Any]], full_total: int) -> dict[str, Any]:
    results = [shard["result"] for shard in shards]
    episodes = sum(result["episodes"] for result in results)
    if episodes != full_total or sum(result["total_episodes"] for result in results) != full_total:
        raise ValueError(
            f"shards report {episodes} records against a {full_total}-record workload; "
            "the partition is not complete"
        )
    successes = sum(result["successes"] for result in results)
    return {
        "state": "complete",
        "episodes": episodes,
        "successes": successes,
        "failures": episodes - successes,
        "total_steps": sum(result["total_steps"] for result in results),
        "total_episodes": full_total,
        "success_rate": successes / episodes,
        "completed_items": sum(result["completed_items"] for result in results),
    }


def _merge_official(benchmark: str, partials: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if benchmark in _ADDITIVE_TABLES:
        key, has_runtime_metrics = _ADDITIVE_TABLES[benchmark]
        return _sum_task_tables(partials, key, has_runtime_metrics)
    if benchmark == "calvin":
        return merge_sequence_lengths(list(partials))
    if benchmark == "metaworld":
        return merge_task_results(list(partials))
    raise ValueError(f"no shard reducer for benchmark {benchmark!r}")


def _sum_task_tables(
    partials: Sequence[Mapping[str, Any]], key: str, has_runtime_metrics: bool
) -> dict[str, Any]:
    """Sum per-task episode/success counts; every shard lists every task but runs only some."""

    layout = [row[key] for row in partials[0]["tasks"]]
    if len(layout) != len(set(layout)):
        raise ValueError("official_result contains duplicate tasks")
    merged: dict[Any, dict[str, Any]] = {}
    for partial in partials:
        if [row[key] for row in partial["tasks"]] != layout:
            raise ValueError("shards list different task sets in their official_result")
        for row in partial["tasks"]:
            episodes = require_count(row["episodes"], "task episodes")
            successes = require_count(row["successes"], "task successes")
            if successes > episodes:
                raise ValueError("task successes exceed its episodes")
            target = merged.setdefault(row[key], {**row, "episodes": 0, "successes": 0})
            target["episodes"] += row["episodes"]
            target["successes"] += row["successes"]
            if not target.get("task_description") and row.get("task_description"):
                target["task_description"] = row["task_description"]
    rows = []
    for task_key in layout:
        row = merged[task_key]
        row["success_rate"] = row["successes"] / row["episodes"] if row["episodes"] else 0.0
        rows.append(row)
    official: dict[str, Any] = {"tasks": rows}
    if has_runtime_metrics:
        wait = sum(p["runtime_metrics"]["inference_wait_s"] for p in partials)
        elapsed = sum(p["runtime_metrics"]["episode_elapsed_s"] for p in partials)
        official["runtime_metrics"] = {
            "inference_wait_s": wait,
            "episode_elapsed_s": elapsed,
            "inference_wait_ratio": wait / max(elapsed, 1e-9),
        }
    return official


def _validate_partial_counts(benchmark: str, shard: Mapping[str, Any]) -> None:
    partial = shard["official_result"]
    if benchmark == "calvin":
        lengths = partial["sequence_success_lengths_by_index"].values()
        episodes, successes = len(lengths), sum(length == 5 for length in lengths)
    else:
        rows = partial["tasks"].values() if benchmark == "metaworld" else partial["tasks"]
        episodes = sum(row["episodes"] for row in rows)
        successes = sum(row["successes"] for row in rows)
    standard = shard["result"]
    if (episodes, successes) != (standard["episodes"], standard["successes"]):
        raise ValueError(f"shard {shard_metadata(shard)['index']} official counts disagree with its records")


def _validate_official_counts(
    benchmark: str, official: Mapping[str, Any], standard: Mapping[str, Any]
) -> None:
    if benchmark == "calvin":
        episodes = official["num_sequences"]
        successes = sum(length == 5 for length in official["sequence_success_lengths"])
    else:
        success_key = "success_count" if benchmark == "metaworld" else "successes"
        episodes = sum(row["episodes"] for row in official["tasks"])
        successes = sum(row[success_key] for row in official["tasks"])
    if episodes != standard["episodes"] or successes != standard["successes"]:
        raise ValueError("official result counts disagree with terminal record counts")


def _shard_summary(shard: Mapping[str, Any]) -> dict[str, Any]:
    result = shard["result"]
    summary: dict[str, Any] = {
        "index": shard_metadata(shard)["index"],
        "records": result["episodes"],
        "successes": result["successes"],
        "autotune": shard.get("autotune"),
    }
    for key in _SHARD_RATE_KEYS:
        if key in result:
            summary[key] = result[key]
    if "resources" in shard:
        summary["resources"] = shard["resources"]
    return summary


__all__ = ["merge_shard_results"]

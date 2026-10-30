# Copyright 2026 RoboScope Team
# SPDX-License-Identifier: Apache-2.0

"""Merge sharded evaluation results from independent GPU instances.

Each shard instance runs the same benchmark command with ``--shard-index I``
(``--shard-count N`` explicit or derived from the workload size) and produces
its own ``result.json`` plus ``queue/completed/*.json``. This tool validates
that the shards form the exact deterministic partition of the full workload —
disjoint items, full coverage, consistent shard count, complete terminal
records — then re-runs the benchmark's own aggregation over the concatenated
records, yielding merged results comparable to a single-instance run.

Rate metrics (steps/s, elapsed) are never aggregated across instances; each
shard keeps its own values as provenance.

Usage:
    python -m roboscope_kit.runtime.benchmark.shard_merge \
        --benchmark robotwin --suite demo_clean \
        --shard-dir /runs/shard-0 --shard-dir /runs/shard-1 \
        --output-dir /runs/merged
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

from roboscope_kit.benchmark_specs import BenchmarkName

from .factories import BenchmarkRunRequest
from .image_manifest import (
    BENCHMARK_IMAGE_MANIFEST_PATH,
    BenchmarkImageManifest,
    load_benchmark_image_manifest,
)
from .orchestrator import aggregate_standard_results
from .queue import WorkQueue, atomic_json
from .registry import benchmark_factory
from .schema import WorkItem, WorkResult, validate_terminal_accounting
from .shard_validation import validate_shard_accounting, validate_shard_identity, validate_shard_set
from .sharding import plan_shard


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", choices=tuple(item.value for item in BenchmarkName), required=True)
    parser.add_argument("--suite", required=True)
    parser.add_argument("--episodes", type=int)
    parser.add_argument("--evaluation-config", type=Path, help="resolved MetaWorld run configuration")
    parser.add_argument("--image-manifest", type=Path, default=BENCHMARK_IMAGE_MANIFEST_PATH)
    parser.add_argument(
        "--shard-dir",
        action="append",
        type=Path,
        required=True,
        help="output directory of one shard run; repeat for every shard",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.episodes is not None and args.episodes < 1:
            raise ValueError("--episodes must be positive")
        merged = merge_shards(
            benchmark=BenchmarkName(args.benchmark),
            suite=args.suite,
            episodes=args.episodes,
            image_manifest_path=args.image_manifest,
            shard_dirs=args.shard_dir,
            evaluation_config_path=args.evaluation_config,
        )
        args.output_dir.mkdir(parents=True, exist_ok=True)
        atomic_json(args.output_dir / "result.json", merged)
    except (ValueError, RuntimeError, OSError, KeyError, TypeError) as exc:
        print(f"shard merge failed: {exc}", file=sys.stderr)
        return 1
    standard = merged["result"]
    print(
        json.dumps(
            {
                "state": standard["state"],
                "episodes": standard["episodes"],
                "successes": standard["successes"],
                "total_episodes": standard["total_episodes"],
                "success_rate": standard["success_rate"],
                "shard_count": merged["shard"]["count"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def merge_shards(
    *,
    benchmark: BenchmarkName,
    suite: str,
    episodes: int | None,
    image_manifest_path: Path,
    shard_dirs: list[Path],
    evaluation_config_path: Path | None = None,
) -> dict[str, Any]:
    manifest = load_benchmark_image_manifest(image_manifest_path)
    factory = benchmark_factory(benchmark)
    factory.validate_suite(suite)
    manifest.validate_request(benchmark, suite)
    request = BenchmarkRunRequest(
        benchmark=benchmark,
        suite=suite,
        # Aggregation never touches the checkpoint; only the path value is stored.
        checkpoint_path=Path("/unused/shard-merge-placeholder"),
        output_dir=Path("/unused/shard-merge-output"),
        episodes=episodes,
        quick_test=False,
        evaluation_config_path=evaluation_config_path,
    )
    adapter = factory.create(request, manifest)
    full_items = list(adapter.work_items())
    full_by_id = _index_full_items(full_items)

    shards: list[tuple[Path, dict[str, Any], dict[str, Any]]] = [
        (shard_dir, *_load_shard(shard_dir)) for shard_dir in shard_dirs
    ]
    _validate_shard_metadata(shards, adapter.metadata(), manifest)
    results_by_shard = _collect_results(shards, full_by_id)
    all_results = [result for shard_results in results_by_shard for result in shard_results]
    total_records = sum(item.expected_results for item in full_items)
    _validate_coverage(shards, full_by_id, total_records)
    validate_terminal_accounting(all_results, total_records)

    ordered = sorted(
        all_results,
        key=lambda result: full_by_id[result.item_id].ordinal,
    )
    official = adapter.aggregate(ordered)
    standard = aggregate_standard_results(ordered, total_records)
    shard_summaries = [
        _shard_summary(shard_dir, shard_manifest, shard_result, shard_results)
        for (shard_dir, shard_manifest, shard_result), shard_results in zip(
            shards, results_by_shard, strict=True
        )
    ]
    return {
        "sdk_version": 1,
        "merged": True,
        "shard": {
            "count": shards[0][1]["shard"]["count"],
            "indexes": [shard_manifest["shard"]["index"] for _, shard_manifest, _ in shards],
            "episodes_per_shard": shards[0][1]["shard"]["episodes_per_shard"],
            "build": shards[0][1]["shard"]["build"],
        },
        "benchmark": adapter.metadata(),
        "result": standard,
        "official_result": official,
        "shards": shard_summaries,
        "validation": {
            "total_records": total_records,
            "unique_record_ids": len({r.record_id for item in ordered for r in item.records})
            == total_records,
            "coverage": "complete",
        },
    }


def _index_full_items(full_items: list[WorkItem]) -> dict[str, WorkItem]:
    by_id: dict[str, WorkItem] = {}
    for item in full_items:
        if item.item_id in by_id:
            raise ValueError(f"duplicate full-workload item id {item.item_id}")
        by_id[item.item_id] = item
    return by_id


def _load_shard(shard_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest_path = shard_dir / "manifest.json"
    result_path = shard_dir / "result.json"
    if not manifest_path.is_file():
        raise ValueError(f"{shard_dir}: manifest.json missing")
    if not result_path.is_file():
        raise ValueError(f"{shard_dir}: result.json missing (shard run incomplete)")
    shard_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    shard_result = json.loads(result_path.read_text(encoding="utf-8"))
    if "shard" not in shard_manifest:
        raise ValueError(f"{shard_dir}: manifest.json carries no shard metadata")
    if shard_result.get("result", {}).get("state") != "complete":
        raise ValueError(f"{shard_dir}: shard result state is not complete")
    if shard_result.get("shard") != shard_manifest["shard"]:
        raise ValueError(f"{shard_dir}: result.json shard metadata disagrees with manifest.json")
    return shard_manifest, shard_result


def _validate_shard_metadata(
    shards: list[tuple[Path, dict[str, Any], dict[str, Any]]],
    expected_identity: dict[str, Any],
    image: BenchmarkImageManifest,
) -> None:
    documents = [result for _, _, result in shards]
    meta = validate_shard_set(documents)
    identity = validate_shard_identity(documents)
    if json.dumps(identity, sort_keys=True) != json.dumps(expected_identity, sort_keys=True):
        raise ValueError("shard benchmark identity disagrees with the merge workload")
    build = {"sdk_version": image.sdk_version, "evaluation_kit_commit": image.evaluation_kit_commit}
    if meta["build"] != build:
        raise ValueError("shard SDK build disagrees with the merge image manifest")
    for shard_dir, manifest, result in shards:
        if manifest.get("benchmark") != result["benchmark"]:
            raise ValueError(f"{shard_dir}: manifest benchmark identity disagrees with result.json")
        validate_shard_accounting(result)


def _validate_shard_items(
    shard_dir: Path, manifest: dict[str, Any], full_items: list[WorkItem]
) -> list[WorkItem]:
    meta = manifest["shard"]
    expected, expected_meta = plan_shard(
        full_items,
        meta["index"],
        meta["count"] if meta["explicit_count"] else None,
        meta["episodes_per_shard"],
        build=meta["build"],
    )
    if meta != expected_meta:
        raise ValueError(f"{shard_dir}: shard metadata disagrees with the full workload")
    items = [WorkItem.from_dict(payload) for payload in manifest["items"]]
    if [item.to_dict() for item in items] != [item.to_dict() for item in expected]:
        raise ValueError(f"{shard_dir}: items disagree with the deterministic shard partition")
    if sum(item.expected_results for item in items) != manifest["total_records"]:
        raise ValueError(f"{shard_dir}: manifest item records do not sum to total_records")
    return items


def _validate_item_results(
    shard_dir: Path, items: list[WorkItem], results: list[WorkResult]
) -> None:
    by_id = {item.item_id: item for item in items}
    if len(results) != len(by_id) or {result.item_id for result in results} != set(by_id):
        raise ValueError(f"{shard_dir}: completed items do not match this shard's items exactly once")
    for result in results:
        item = by_id[result.item_id]
        if result.group_id != item.group_id or len(result.records) != item.expected_results:
            raise ValueError(f"{shard_dir}: item {item.item_id} has incorrect group or record count")


def _collect_results(
    shards: list[tuple[Path, dict[str, Any], dict[str, Any]]],
    full_by_id: dict[str, WorkItem],
) -> list[list[WorkResult]]:
    results_by_shard: list[list[WorkResult]] = []
    for shard_dir, shard_manifest, shard_result in shards:
        items = _validate_shard_items(shard_dir, shard_manifest, list(full_by_id.values()))
        results = WorkQueue(shard_dir / "queue").completed_results()
        _validate_item_results(shard_dir, items, results)
        standard = aggregate_standard_results(results, shard_manifest["total_records"])
        if any(shard_result["result"].get(key) != value for key, value in standard.items()):
            raise ValueError(f"{shard_dir}: result.json disagrees with the completed records")
        results_by_shard.append(results)
    return results_by_shard


def _validate_coverage(
    shards: list[tuple[Path, dict[str, Any], dict[str, Any]]],
    full_by_id: dict[str, WorkItem],
    total_records: int,
) -> None:
    assigned = {
        item_id
        for _, shard_manifest, _ in shards
        for item_id in (WorkItem.from_dict(payload).item_id for payload in shard_manifest["items"])
    }
    missing = set(full_by_id) - assigned
    if missing:
        preview = sorted(missing)[:5]
        raise ValueError(f"{len(missing)} full-workload items not covered by any shard, e.g. {preview}")
    if total_records < 1:
        raise ValueError("full workload has no records")


def _shard_summary(
    shard_dir: Path,
    shard_manifest: dict[str, Any],
    shard_result: dict[str, Any],
    results: list[WorkResult],
) -> dict[str, Any]:
    records = [record for result in results for record in result.records]
    result_bytes = (shard_dir / "result.json").read_bytes()
    summary: dict[str, Any] = {
        "dir": str(shard_dir),
        "index": shard_manifest["shard"]["index"],
        "result_json_sha256": hashlib.sha256(result_bytes).hexdigest(),
        "records": len(records),
        "successes": sum(record.success for record in records),
        "autotune": shard_result.get("autotune"),
    }
    shard_standard = shard_result.get("result", {})
    for key in ("elapsed_s", "steps_per_s", "episodes_per_s", "total_steps"):
        if key in shard_standard:
            summary[key] = shard_standard[key]
    if "resources" in shard_result:
        summary["resources"] = shard_result["resources"]
    return summary


if __name__ == "__main__":
    raise SystemExit(main())

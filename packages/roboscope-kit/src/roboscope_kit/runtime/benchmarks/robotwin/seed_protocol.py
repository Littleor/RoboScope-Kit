# Copyright 2026 RoboScope Team
# SPDX-License-Identifier: Apache-2.0

"""Immutable RoboTwin benchmark protocol and seed-cache generations."""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import subprocess
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .seed_cache import validate_seed_cache

GENERATION_MANIFEST_SCHEMA_VERSION = 1


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_digest(value: dict[str, Any]) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def protocol_suite(protocol: dict[str, Any], suite: str) -> dict[str, Any]:
    protocol_id = str(protocol.get("protocol_id") or "")
    runtime_commit = str(protocol.get("robotwin_runtime_commit") or "")
    suites = protocol.get("suites")
    if not protocol_id or not runtime_commit or not isinstance(suites, dict):
        raise ValueError("RoboTwin benchmark protocol identity is incomplete")
    suite_protocol = suites.get(suite)
    if not isinstance(suite_protocol, dict):
        raise ValueError(f"RoboTwin benchmark protocol does not declare suite {suite!r}")
    required = {
        "task_config_path",
        "seed_selection",
        "episodes_per_task",
        "action_horizon",
        "success_criterion",
        "aggregation",
    }
    if set(suite_protocol) != required:
        raise ValueError(f"RoboTwin suite protocol fields are incomplete for {suite}")
    if int(suite_protocol["episodes_per_task"]) < 1:
        raise ValueError("RoboTwin protocol episodes_per_task must be positive")
    return suite_protocol


def protocol_tasks(protocol: dict[str, Any]) -> tuple[str, ...]:
    value = protocol.get("tasks")
    if not isinstance(value, list):
        raise ValueError("RoboTwin benchmark protocol tasks must be an array")
    tasks = tuple(str(task) for task in value)
    if not tasks or len(tasks) != len(set(tasks)):
        raise ValueError("RoboTwin benchmark protocol tasks must be unique and ordered")
    return tasks


def robotwin_runtime_commit(robotwin_root: Path) -> str:
    commit = subprocess.run(
        ("git", "-C", str(robotwin_root), "rev-parse", "HEAD"),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    changes = subprocess.run(
        ("git", "-C", str(robotwin_root), "status", "--short", "--untracked-files=no"),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if changes:
        raise RuntimeError("RoboTwin runtime tracked sources differ from the pinned commit")
    return commit


def _task_identity(task: str, path: Path, entries: list[dict]) -> dict[str, Any]:
    seeds = [int(entry["seed"]) for entry in entries]
    return {
        "task": task,
        "count": len(entries),
        "seed_min": min(seeds),
        "seed_max": max(seeds),
        "pickle_sha256": sha256_file(path),
    }


def _validate_seed_selection(entries: list[dict], seed_selection: dict[str, Any]) -> None:
    if seed_selection.get("algorithm") != "ascending-first-expert-valid-v1":
        raise ValueError("RoboTwin seed selection algorithm is unsupported")
    seeds = [int(entry["seed"]) for entry in entries]
    if seeds != sorted(set(seeds)):
        raise RuntimeError("RoboTwin generation seeds must be unique and ascending")
    if seeds[0] < int(seed_selection["candidate_start"]):
        raise RuntimeError("RoboTwin generation starts before the protocol candidate range")


def create_seed_cache_generation(
    *,
    source_cache_dir: Path,
    generation_root: Path,
    suite: str,
    tasks: tuple[str, ...],
    protocol: dict[str, Any],
    evaluation_kit_commit: str,
    robotwin_root: Path,
    scanner_path: Path,
) -> Path:
    suite_protocol = protocol_suite(protocol, suite)
    if tasks != protocol_tasks(protocol):
        raise RuntimeError("RoboTwin generation task order differs from the benchmark protocol")
    runtime_commit = robotwin_runtime_commit(robotwin_root)
    if runtime_commit != str(protocol["robotwin_runtime_commit"]):
        raise RuntimeError("RoboTwin runtime checkout differs from the benchmark protocol")
    episodes = int(suite_protocol["episodes_per_task"])
    task_config_path = robotwin_root / str(suite_protocol["task_config_path"])
    if not task_config_path.is_file() or not scanner_path.is_file():
        raise RuntimeError("RoboTwin task config and scanner source must exist")
    generation_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".robotwin-generation-", dir=generation_root) as temp:
        staging = Path(temp)
        suite_dir = staging / suite
        suite_dir.mkdir()
        identities = []
        for task in tasks:
            entries = validate_seed_cache(source_cache_dir / suite / f"{task}.pkl", episodes)
            _validate_seed_selection(entries[:episodes], suite_protocol["seed_selection"])
            target = suite_dir / f"{task}.pkl"
            with target.open("wb") as handle:
                pickle.dump(entries[:episodes], handle, protocol=pickle.HIGHEST_PROTOCOL)
            identities.append(_task_identity(task, target, entries[:episodes]))
        identity = {
            "schema_version": GENERATION_MANIFEST_SCHEMA_VERSION,
            "protocol_id": str(protocol["protocol_id"]),
            "suite": suite,
            "evaluation_kit_commit": evaluation_kit_commit,
            "robotwin_runtime_commit": runtime_commit,
            "task_config_sha256": sha256_file(task_config_path),
            "scanner_sha256": sha256_file(scanner_path),
            "seed_selection": suite_protocol["seed_selection"],
            "tasks": identities,
        }
        generation_digest = canonical_digest(identity)
        manifest = {
            **identity,
            "generation_digest": generation_digest,
            "generated_at": datetime.now(UTC).isoformat(),
        }
        (staging / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        target = generation_root / generation_digest
        if target.exists():
            evidence = validate_seed_cache_generation(
                generation_dir=target,
                suite=suite,
                tasks=tasks,
                episodes=episodes,
                protocol=protocol,
                evaluation_kit_commit=evaluation_kit_commit,
                robotwin_root=robotwin_root,
                scanner_path=scanner_path,
            )
            if evidence["generation_digest"] != generation_digest:
                raise RuntimeError("existing RoboTwin generation has conflicting identity")
            return target
        staging.replace(target)
        return target


def validate_seed_cache_generation(
    *,
    generation_dir: Path,
    suite: str,
    tasks: tuple[str, ...],
    episodes: int,
    protocol: dict[str, Any],
    evaluation_kit_commit: str,
    robotwin_root: Path,
    scanner_path: Path,
) -> dict[str, Any]:
    suite_protocol = protocol_suite(protocol, suite)
    if tasks != protocol_tasks(protocol):
        raise RuntimeError("RoboTwin formal task order differs from the benchmark protocol")
    runtime_commit = robotwin_runtime_commit(robotwin_root)
    if runtime_commit != str(protocol["robotwin_runtime_commit"]):
        raise RuntimeError("RoboTwin runtime checkout differs from the benchmark protocol")
    if episodes != int(suite_protocol["episodes_per_task"]):
        raise RuntimeError("RoboTwin formal episodes differ from the benchmark protocol")
    manifest_path = generation_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    generated_at = manifest.pop("generated_at", None)
    generation_digest = str(manifest.pop("generation_digest", ""))
    if generated_at is None or canonical_digest(manifest) != generation_digest:
        raise RuntimeError("RoboTwin seed generation manifest digest is invalid")
    if generation_dir.name != generation_digest:
        raise RuntimeError("RoboTwin seed generation directory must equal its digest")
    task_config_path = robotwin_root / str(suite_protocol["task_config_path"])
    expected = {
        "schema_version": GENERATION_MANIFEST_SCHEMA_VERSION,
        "protocol_id": str(protocol["protocol_id"]),
        "suite": suite,
        "evaluation_kit_commit": evaluation_kit_commit,
        "robotwin_runtime_commit": runtime_commit,
        "task_config_sha256": sha256_file(task_config_path),
        "scanner_sha256": sha256_file(scanner_path),
        "seed_selection": suite_protocol["seed_selection"],
    }
    for name, value in expected.items():
        if manifest.get(name) != value:
            raise RuntimeError(f"RoboTwin seed generation {name} differs from protocol")
    task_rows = manifest.get("tasks")
    if not isinstance(task_rows, list) or [row.get("task") for row in task_rows] != list(tasks):
        raise RuntimeError("RoboTwin seed generation task order differs from protocol")
    for row, task in zip(task_rows, tasks, strict=True):
        path = generation_dir / suite / f"{task}.pkl"
        entries = validate_seed_cache(path, episodes)
        _validate_seed_selection(entries, suite_protocol["seed_selection"])
        if len(entries) != episodes or _task_identity(task, path, entries) != row:
            raise RuntimeError(f"RoboTwin seed generation content differs for {task}")
    return {
        "protocol_id": str(protocol["protocol_id"]),
        "generation_digest": generation_digest,
        "suite": suite,
        "task_count": len(tasks),
        "episodes_per_task": episodes,
        "total_episodes": len(tasks) * episodes,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-cache-dir", type=Path, required=True)
    parser.add_argument("--generation-root", type=Path, required=True)
    parser.add_argument("--suite", required=True)
    parser.add_argument("--benchmark-manifest", type=Path, required=True)
    parser.add_argument("--robotwin-root", type=Path, required=True)
    parser.add_argument("--scanner-path", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    from roboscope_kit.benchmark_specs import BenchmarkName
    from roboscope_kit.runtime.benchmark.manifest import (
        load_benchmark_manifest,
    )

    args = parse_args(argv)
    manifest = load_benchmark_manifest(args.benchmark_manifest)
    protocol = manifest.benchmark_protocols[BenchmarkName.ROBOTWIN]
    target = create_seed_cache_generation(
        source_cache_dir=args.source_cache_dir,
        generation_root=args.generation_root,
        suite=args.suite,
        tasks=protocol_tasks(protocol),
        protocol=protocol,
        evaluation_kit_commit=str(protocol["scanner_evaluation_kit_commit"]),
        robotwin_root=args.robotwin_root,
        scanner_path=args.scanner_path,
    )
    print(target)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "create_seed_cache_generation",
    "protocol_tasks",
    "robotwin_runtime_commit",
    "validate_seed_cache_generation",
]

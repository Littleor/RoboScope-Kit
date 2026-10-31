# Copyright 2026 RoboScope Team
# SPDX-License-Identifier: Apache-2.0

"""Factory for the official RoboTwin benchmark adapter."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from roboscope_kit.benchmark_specs import BenchmarkName
from roboscope_kit.runtime.benchmark.autotune import probe_items_per_worker
from roboscope_kit.runtime.benchmark.factories import (
    BenchmarkRunRequest,
    nested_task_order,
    representative_indices,
)
from roboscope_kit.runtime.benchmark.manifest import BenchmarkManifest

from .adapter import ROBOTWIN_ALL_TASKS, RoboTwinBenchmarkAdapter, RoboTwinRuntimeConfig
from .seed_cache import validate_seed_cache_dir
from .seed_protocol import protocol_suite, validate_seed_cache_generation


class RoboTwinBenchmarkFactory:
    @staticmethod
    def validate_suite(suite: str) -> None:
        if suite not in {"demo_clean", "demo_randomized"}:
            raise ValueError(f"robotwin does not support suite {suite!r}")

    def create(
        self,
        request: BenchmarkRunRequest,
        manifest: BenchmarkManifest,
    ) -> RoboTwinBenchmarkAdapter:
        tasks = self._tasks(manifest)
        if request.quick_test:
            tasks = tasks[:1]
        episodes = self._episodes(request, manifest)
        adapter = self._create(
            request,
            manifest,
            tasks,
            episodes,
            protocol_run=not request.quick_test,
            require_exact_episodes=not request.quick_test,
        )
        if not request.quick_test:
            self._validate_formal_generation(adapter, manifest, tasks, episodes)
        return adapter

    @staticmethod
    def _validate_formal_generation(
        adapter: RoboTwinBenchmarkAdapter,
        manifest: BenchmarkManifest,
        tasks: tuple[str, ...],
        episodes: int,
    ) -> None:
        protocol = manifest.benchmark_protocols.get(BenchmarkName.ROBOTWIN)
        if protocol is None:
            validate_seed_cache_dir(
                adapter.config.seed_cache_dir,
                adapter.config.suite,
                tasks,
                episodes,
            )
            return
        scanner_path = manifest.runtime_path("seed_scanner_path")
        evidence = validate_seed_cache_generation(
            generation_dir=adapter.config.seed_cache_dir,
            suite=adapter.config.suite,
            tasks=tasks,
            episodes=episodes,
            protocol=protocol,
            evaluation_kit_commit=str(protocol["scanner_evaluation_kit_commit"]),
            robotwin_root=adapter.config.robotwin_root,
            scanner_path=scanner_path,
        )
        if evidence["generation_digest"] != adapter.config.seed_generation_digest:
            raise RuntimeError("RoboTwin seed generation differs from benchmark manifest")

    def create_probe(
        self,
        request: BenchmarkRunRequest,
        manifest: BenchmarkManifest,
        concurrency: int,
    ) -> RoboTwinBenchmarkAdapter:
        tasks = self._tasks(manifest)
        task_count = int(manifest.defaults.get("probe_task_count", 10))
        episodes = int(manifest.defaults.get("probe_episodes_per_task", 1))
        # Give every worker items_per_worker items: first add tasks as a prefix of one
        # nested order, then whole items per task, so each probe repeats the episodes of
        # every smaller probe.
        items_per_worker = probe_items_per_worker(manifest.defaults)
        if items_per_worker > 0:
            needed = items_per_worker * concurrency
            order = nested_task_order(len(tasks))
            indices = tuple(sorted(order[: max(task_count, needed)]))
            items_per_task = -(-needed // len(indices))
            if items_per_task > 1:
                shard = int(manifest.defaults["episode_shard_size"])
                episodes = max(episodes, items_per_task * shard)
        else:
            indices = representative_indices(len(tasks), task_count)
        protocol = manifest.benchmark_protocols.get(BenchmarkName.ROBOTWIN)
        if protocol is not None:
            episodes = min(episodes, int(protocol_suite(protocol, request.suite)["episodes_per_task"]))
        selected = tuple(tasks[index] for index in indices)
        adapter = self._create(
            request,
            manifest,
            selected,
            episodes,
            protocol_run=True,
            require_exact_episodes=False,
        )
        if protocol is None:
            return adapter
        formal_episodes = int(protocol_suite(protocol, request.suite)["episodes_per_task"])
        self._validate_formal_generation(adapter, manifest, tasks, formal_episodes)
        return adapter

    def self_test(self, manifest: BenchmarkManifest, output_dir: Path) -> dict[str, Any]:
        python = Path(sys.executable)
        benchmark_python = manifest.runtime_path("benchmark_python")
        root = manifest.runtime_path("benchmark_root")
        deploy_policy = manifest.runtime_path("deploy_policy_path")
        launcher = manifest.runtime_path("launcher_path")
        required = (
            python,
            benchmark_python,
            root,
            root / "script" / "eval_policy.py",
            deploy_policy,
            launcher,
        )
        seed_cache_dir = manifest.runtime_path("seed_cache_dir", required=False)
        if seed_cache_dir is not None:
            required += (seed_cache_dir,)
        missing = [str(path) for path in required if not path.exists()]
        if missing:
            raise RuntimeError(f"RoboTwin runtime paths are missing: {', '.join(missing)}")
        output_dir.mkdir(parents=True, exist_ok=True)
        return {
            "benchmark": "robotwin",
            "python": str(python),
            "benchmark_root": str(root),
            "task_count": len(self._tasks(manifest)),
        }

    def _create(
        self,
        request: BenchmarkRunRequest,
        manifest: BenchmarkManifest,
        tasks: tuple[str, ...],
        episodes: int,
        *,
        protocol_run: bool,
        require_exact_episodes: bool,
    ) -> RoboTwinBenchmarkAdapter:
        self.validate_suite(request.suite)
        # The Scanner cache is mandatory for a full evaluation run (it makes the
        # optimized skip-play_once path the only path). A quick-test is exempt: it
        # may run without a cache, falling back to the official expert play_once
        # path so a fresh manifest can still be smoke-tested before seeds are scanned.
        protocol = manifest.benchmark_protocols.get(BenchmarkName.ROBOTWIN)
        if not protocol_run:
            seed_cache_dir = manifest.runtime_path("seed_cache_dir", required=False)
            protocol_id = None
            generation_digest = None
        elif protocol is None:
            seed_cache_dir = manifest.runtime_path("seed_cache_dir")
            protocol_id = None
            generation_digest = None
        else:
            suite_protocol = protocol_suite(protocol, request.suite)
            generations = manifest.runtime.get("seed_generations")
            if not isinstance(generations, dict):
                raise ValueError("RoboTwin runtime seed_generations must be an object")
            generation_digest = str(generations.get(request.suite) or "")
            if not generation_digest:
                raise ValueError(f"RoboTwin runtime has no seed generation for {request.suite}")
            generation_root = manifest.runtime_path("seed_generation_root")
            seed_cache_dir = generation_root / generation_digest
            protocol_id = str(protocol["protocol_id"])
            if require_exact_episodes and episodes != int(suite_protocol["episodes_per_task"]):
                raise ValueError("RoboTwin formal episodes must match the benchmark protocol")
        return RoboTwinBenchmarkAdapter(
            RoboTwinRuntimeConfig(
                sdk_python=Path(sys.executable),
                benchmark_python=manifest.runtime_path("benchmark_python"),
                robotwin_root=manifest.runtime_path("benchmark_root"),
                checkpoint_path=request.checkpoint_path,
                deploy_policy_path=manifest.runtime_path("deploy_policy_path"),
                launcher_path=manifest.runtime_path("launcher_path"),
                suite=request.suite,
                tasks=tasks,
                episodes_per_task=episodes,
                episode_shard_size=int(manifest.defaults["episode_shard_size"]),
                seed=int(manifest.defaults.get("seed", 0)),
                save_videos=bool(manifest.defaults.get("save_videos", False)) and not request.quick_test,
                cpu_threads=int(manifest.defaults.get("client_cpu_threads", 1)),
                seed_cache_dir=seed_cache_dir,
                protocol_id=protocol_id,
                seed_generation_digest=generation_digest,
            )
        )

    @staticmethod
    def _tasks(manifest: BenchmarkManifest) -> tuple[str, ...]:
        raw = manifest.benchmark_protocols[BenchmarkName.ROBOTWIN].get("tasks")
        if not isinstance(raw, list):
            raise ValueError("RoboTwin benchmark protocol tasks must be an array")
        tasks = tuple(str(task) for task in raw)
        if tasks != ROBOTWIN_ALL_TASKS:
            raise ValueError("RoboTwin benchmark protocol tasks must match the official task order")
        return tasks

    @staticmethod
    def _episodes(request: BenchmarkRunRequest, manifest: BenchmarkManifest) -> int:
        if request.quick_test:
            return 1
        value = request.episodes or int(manifest.defaults.get("episodes", 1))
        if value < 1:
            raise ValueError("RoboTwin episodes must be positive")
        return value




__all__ = ["RoboTwinBenchmarkFactory"]

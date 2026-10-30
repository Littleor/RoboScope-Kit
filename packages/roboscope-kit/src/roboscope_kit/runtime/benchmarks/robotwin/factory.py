# Copyright 2026 RoboScope Team
# SPDX-License-Identifier: Apache-2.0

"""Factory for the official RoboTwin benchmark adapter."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from roboscope_kit.benchmark_specs import BenchmarkName
from roboscope_kit.runtime.benchmark.factories import (
    BenchmarkRunRequest,
    representative_indices,
)
from roboscope_kit.runtime.benchmark.image_manifest import BenchmarkImageManifest

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
        image: BenchmarkImageManifest,
    ) -> RoboTwinBenchmarkAdapter:
        tasks = self._tasks(image)
        if request.quick_test:
            tasks = tasks[:1]
        episodes = self._episodes(request, image)
        adapter = self._create(
            request,
            image,
            tasks,
            episodes,
            protocol_run=not request.quick_test,
            require_exact_episodes=not request.quick_test,
        )
        if not request.quick_test:
            self._validate_formal_generation(adapter, image, tasks, episodes)
        return adapter

    @staticmethod
    def _validate_formal_generation(
        adapter: RoboTwinBenchmarkAdapter,
        image: BenchmarkImageManifest,
        tasks: tuple[str, ...],
        episodes: int,
    ) -> None:
        protocol = image.benchmark_protocols.get(BenchmarkName.ROBOTWIN)
        if protocol is None:
            validate_seed_cache_dir(
                adapter.config.seed_cache_dir,
                adapter.config.suite,
                tasks,
                episodes,
            )
            return
        runtime = image.runtime
        scanner_path = _required_path(runtime, "seed_scanner_path")
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
        image: BenchmarkImageManifest,
        concurrency: int,
    ) -> RoboTwinBenchmarkAdapter:
        del concurrency
        tasks = self._tasks(image)
        indices = representative_indices(
            len(tasks),
            int(image.defaults.get("probe_task_count", 10)),
        )
        selected = tuple(tasks[index] for index in indices)
        episodes = int(image.defaults.get("probe_episodes_per_task", 1))
        adapter = self._create(
            request,
            image,
            selected,
            episodes,
            protocol_run=True,
            require_exact_episodes=False,
        )
        protocol = image.benchmark_protocols.get(BenchmarkName.ROBOTWIN)
        if protocol is None:
            return adapter
        formal_episodes = int(protocol_suite(protocol, request.suite)["episodes_per_task"])
        self._validate_formal_generation(adapter, image, tasks, formal_episodes)
        return adapter

    def self_test(self, image: BenchmarkImageManifest, output_dir: Path) -> dict[str, Any]:
        python = _required_path(image.runtime, "python")
        benchmark_python = _required_path(image.runtime, "benchmark_python")
        root = _required_path(image.runtime, "benchmark_root")
        deploy_policy = _required_path(image.runtime, "deploy_policy_path")
        launcher = _required_path(image.runtime, "launcher_path")
        required = (
            python,
            benchmark_python,
            root,
            root / "script" / "eval_policy.py",
            deploy_policy,
            launcher,
        )
        seed_cache_dir = _optional_path(image.runtime.get("seed_cache_dir"))
        if seed_cache_dir is not None:
            required += (seed_cache_dir,)
        missing = [str(path) for path in required if not path.exists()]
        if missing:
            raise RuntimeError(f"RoboTwin image paths are missing: {', '.join(missing)}")
        output_dir.mkdir(parents=True, exist_ok=True)
        return {
            "benchmark": "robotwin",
            "python": str(python),
            "benchmark_root": str(root),
            "task_count": len(self._tasks(image)),
        }

    def _create(
        self,
        request: BenchmarkRunRequest,
        image: BenchmarkImageManifest,
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
        # path so a fresh image can still be smoke-tested before seeds are scanned.
        protocol = image.benchmark_protocols.get(BenchmarkName.ROBOTWIN)
        if not protocol_run:
            seed_cache_dir = _optional_path(image.runtime.get("seed_cache_dir"))
            protocol_id = None
            generation_digest = None
        elif protocol is None:
            seed_cache_dir = _required_path(image.runtime, "seed_cache_dir")
            protocol_id = None
            generation_digest = None
        else:
            suite_protocol = protocol_suite(protocol, request.suite)
            generations = image.runtime.get("seed_generations")
            if not isinstance(generations, dict):
                raise ValueError("RoboTwin runtime seed_generations must be an object")
            generation_digest = str(generations.get(request.suite) or "")
            if not generation_digest:
                raise ValueError(f"RoboTwin runtime has no seed generation for {request.suite}")
            generation_root = _required_path(image.runtime, "seed_generation_root")
            seed_cache_dir = generation_root / generation_digest
            protocol_id = str(protocol["protocol_id"])
            if require_exact_episodes and episodes != int(suite_protocol["episodes_per_task"]):
                raise ValueError("RoboTwin formal episodes must match the benchmark protocol")
        return RoboTwinBenchmarkAdapter(
            RoboTwinRuntimeConfig(
                sdk_python=_required_path(image.runtime, "python"),
                benchmark_python=_required_path(image.runtime, "benchmark_python"),
                robotwin_root=_required_path(image.runtime, "benchmark_root"),
                checkpoint_path=request.checkpoint_path,
                deploy_policy_path=_required_path(image.runtime, "deploy_policy_path"),
                launcher_path=_required_path(image.runtime, "launcher_path"),
                suite=request.suite,
                tasks=tasks,
                episodes_per_task=episodes,
                episode_shard_size=int(image.defaults["episode_shard_size"]),
                seed=int(image.defaults.get("seed", 0)),
                save_videos=bool(image.defaults.get("save_videos", False)) and not request.quick_test,
                cpu_threads=int(image.defaults.get("client_cpu_threads", 1)),
                seed_cache_dir=seed_cache_dir,
                protocol_id=protocol_id,
                seed_generation_digest=generation_digest,
            )
        )

    @staticmethod
    def _tasks(image: BenchmarkImageManifest) -> tuple[str, ...]:
        raw = image.benchmark_protocols[BenchmarkName.ROBOTWIN].get("tasks")
        if not isinstance(raw, list):
            raise ValueError("RoboTwin benchmark protocol tasks must be an array")
        tasks = tuple(str(task) for task in raw)
        if tasks != ROBOTWIN_ALL_TASKS:
            raise ValueError("RoboTwin benchmark protocol tasks must match the official task order")
        return tasks

    @staticmethod
    def _episodes(request: BenchmarkRunRequest, image: BenchmarkImageManifest) -> int:
        if request.quick_test:
            return 1
        value = request.episodes or int(image.defaults.get("episodes", 1))
        if value < 1:
            raise ValueError("RoboTwin episodes must be positive")
        return value


def _required_path(runtime: dict[str, Any], name: str) -> Path:
    value = Path(str(runtime.get(name) or ""))
    if not value.is_absolute():
        raise ValueError(f"RoboTwin runtime {name} must be an absolute path")
    return value


def _optional_path(value: Any) -> Path | None:
    if value is None:
        return None
    path = Path(str(value))
    if not path.is_absolute():
        raise ValueError(f"RoboTwin runtime seed_cache_dir must be an absolute path: {value}")
    return path


__all__ = ["RoboTwinBenchmarkFactory"]

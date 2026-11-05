# Copyright 2026 RoboScope Team
# SPDX-License-Identifier: Apache-2.0

"""Factory for the official LIBERO+ benchmark adapter."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from roboscope_kit.runtime.benchmark.autotune import probe_items_per_worker
from roboscope_kit.runtime.benchmark.factories import (
    BenchmarkRunRequest,
    nested_task_prefix,
    representative_indices,
)
from roboscope_kit.runtime.benchmark.manifest import BenchmarkManifest

from .adapter import LiberoPlusBenchmarkAdapter, LiberoPlusRuntimeConfig

LIBERO_PLUS_TASK_COUNTS = {
    "libero_spatial": 2402,
    "libero_object": 2518,
    "libero_goal": 2591,
    "libero_10": 2519,
}

# The LIBERO+ manifest lists these suites; libero_90 is not included.
LIBERO_PLUS_SUITES = frozenset(LIBERO_PLUS_TASK_COUNTS)


class LiberoPlusBenchmarkFactory:
    @staticmethod
    def validate_suite(suite: str) -> None:
        if suite not in LIBERO_PLUS_SUITES:
            supported = ", ".join(sorted(LIBERO_PLUS_SUITES))
            raise ValueError(
                f"libero_plus does not support suite {suite!r}; supported suites: {supported}"
            )

    def create(
        self,
        request: BenchmarkRunRequest,
        manifest: BenchmarkManifest,
    ) -> LiberoPlusBenchmarkAdapter:
        task_ids = (0,) if request.quick_test else None
        return self._create(request, manifest, task_ids=task_ids, episodes=self._episodes(request, manifest))

    def create_probe(
        self,
        request: BenchmarkRunRequest,
        manifest: BenchmarkManifest,
        concurrency: int,
    ) -> LiberoPlusBenchmarkAdapter:
        total = LIBERO_PLUS_TASK_COUNTS[request.suite]
        task_count = int(manifest.defaults.get("probe_task_count", 10))
        episodes = int(manifest.defaults.get("probe_episodes_per_task", 1))
        # Each LIBERO+ task is one perturbed variant, so the probe grows by tasks: a
        # prefix of one nested order that gives every worker items_per_worker items
        # and repeats the tasks of every smaller probe.
        items_per_worker = probe_items_per_worker(manifest.defaults)
        if items_per_worker > 0:
            chunk_size = int(manifest.defaults.get("chunk_size", 1))
            items_per_task = -(-episodes // chunk_size)
            needed_tasks = -(-items_per_worker * concurrency // items_per_task)
            task_ids = nested_task_prefix(total, max(task_count, needed_tasks))
        else:
            task_ids = representative_indices(total, task_count)
        return self._create(
            request,
            manifest,
            task_ids=task_ids,
            episodes=episodes,
        )

    def self_test(self, manifest: BenchmarkManifest, output_dir: Path) -> dict[str, Any]:
        runtime = self._runtime(manifest)
        required = (runtime["python"], runtime["benchmark_root"], runtime["config_path"])
        missing = [str(path) for path in required if not path.exists()]
        if missing:
            raise RuntimeError(f"LIBERO+ runtime paths are missing: {', '.join(missing)}")
        output_dir.mkdir(parents=True, exist_ok=True)
        return {"benchmark": "libero_plus", "paths": [str(path) for path in required]}

    def _create(
        self,
        request: BenchmarkRunRequest,
        manifest: BenchmarkManifest,
        *,
        task_ids: tuple[int, ...] | None,
        episodes: int,
    ) -> LiberoPlusBenchmarkAdapter:
        self.validate_suite(request.suite)
        runtime = self._runtime(manifest)
        return LiberoPlusBenchmarkAdapter(
            LiberoPlusRuntimeConfig(
                python=runtime["python"],
                libero_home=runtime["benchmark_root"],
                libero_config_path=runtime["config_path"],
                suite=request.suite,
                num_tasks=LIBERO_PLUS_TASK_COUNTS[request.suite],
                num_trials=episodes,
                task_ids=task_ids,
                sample_seed=int(manifest.defaults.get("sample_seed", 7)),
                chunk_size=int(manifest.defaults.get("chunk_size", 1)),
                seed=int(manifest.defaults.get("seed", 7)),
                settle_steps=int(manifest.defaults.get("settle_steps", 10)),
                render_backend=str(manifest.defaults.get("render_backend", "egl")),
                save_videos=bool(manifest.defaults.get("save_videos", False)) and not request.quick_test,
                cpu_threads=int(manifest.defaults.get("client_cpu_threads", 1)),
            )
        )

    @staticmethod
    def _runtime(manifest: BenchmarkManifest) -> dict[str, Path]:
        return {name: manifest.runtime_path(name) for name in ("python", "benchmark_root", "config_path")}

    @staticmethod
    def _episodes(request: BenchmarkRunRequest, manifest: BenchmarkManifest) -> int:
        if request.quick_test:
            return 1
        value = request.episodes or int(manifest.defaults.get("episodes", 1))
        if value < 1:
            raise ValueError("LIBERO+ episodes must be positive")
        return value



__all__ = ["LiberoPlusBenchmarkFactory"]

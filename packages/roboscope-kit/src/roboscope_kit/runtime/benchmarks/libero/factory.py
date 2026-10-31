# Copyright 2026 RoboScope Team
# SPDX-License-Identifier: Apache-2.0

"""Factory for the official LIBERO benchmark adapter."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from roboscope_kit.runtime.benchmark.autotune import probe_items_per_worker
from roboscope_kit.runtime.benchmark.factories import (
    BenchmarkRunRequest,
    representative_indices,
)
from roboscope_kit.runtime.benchmark.image_manifest import BenchmarkImageManifest

from .adapter import LiberoBenchmarkAdapter, LiberoRuntimeConfig

LIBERO_TASK_COUNTS = {
    "libero_spatial": 10,
    "libero_object": 10,
    "libero_goal": 10,
    "libero_10": 10,
    "libero_90": 90,
}

# Every official LIBERO task ships this many initial states, one per episode.
LIBERO_INITIAL_STATES = 50

# The LIBERO Benchmark definition runs every official suite.
LIBERO_SUITES = frozenset(LIBERO_TASK_COUNTS)


class LiberoBenchmarkFactory:
    @staticmethod
    def validate_suite(suite: str) -> None:
        if suite not in LIBERO_SUITES:
            supported = ", ".join(sorted(LIBERO_SUITES))
            raise ValueError(f"libero does not support suite {suite!r}; supported suites: {supported}")

    def create(
        self,
        request: BenchmarkRunRequest,
        image: BenchmarkImageManifest,
    ) -> LiberoBenchmarkAdapter:
        task_ids = (0,) if request.quick_test else None
        return self._create(request, image, task_ids=task_ids, episodes=self._episodes(request, image))

    def create_probe(
        self,
        request: BenchmarkRunRequest,
        image: BenchmarkImageManifest,
        concurrency: int,
    ) -> LiberoBenchmarkAdapter:
        task_ids = representative_indices(
            LIBERO_TASK_COUNTS[request.suite],
            int(image.defaults.get("probe_task_count", 10)),
        )
        episodes = int(image.defaults.get("probe_episodes_per_task", 1))
        # Grow only episodes per task, in chunk_size items, until every worker gets
        # items_per_worker items. The task set stays fixed, so each probe repeats the
        # episodes of every smaller probe.
        items_per_worker = probe_items_per_worker(image.defaults)
        if items_per_worker > 0:
            chunk_size = int(image.defaults.get("chunk_size", 1))
            chunks_per_task = -(-items_per_worker * concurrency // len(task_ids))
            episodes = max(episodes, chunks_per_task * chunk_size)
        episodes = min(episodes, LIBERO_INITIAL_STATES)
        return self._create(
            request,
            image,
            task_ids=task_ids,
            episodes=episodes,
        )

    def self_test(self, image: BenchmarkImageManifest, output_dir: Path) -> dict[str, Any]:
        runtime = self._runtime(image)
        required = (runtime["python"], runtime["benchmark_root"], runtime["config_path"])
        missing = [str(path) for path in required if not path.exists()]
        if missing:
            raise RuntimeError(f"LIBERO image paths are missing: {', '.join(missing)}")
        output_dir.mkdir(parents=True, exist_ok=True)
        return {"benchmark": "libero", "paths": [str(path) for path in required]}

    def _create(
        self,
        request: BenchmarkRunRequest,
        image: BenchmarkImageManifest,
        *,
        task_ids: tuple[int, ...] | None,
        episodes: int,
    ) -> LiberoBenchmarkAdapter:
        self.validate_suite(request.suite)
        runtime = self._runtime(image)
        return LiberoBenchmarkAdapter(
            LiberoRuntimeConfig(
                python=runtime["python"],
                libero_home=runtime["benchmark_root"],
                libero_config_path=runtime["config_path"],
                suite=request.suite,
                num_tasks=LIBERO_TASK_COUNTS[request.suite],
                num_trials=episodes,
                task_ids=task_ids,
                sample_seed=int(image.defaults.get("sample_seed", 7)),
                chunk_size=int(image.defaults.get("chunk_size", 1)),
                seed=int(image.defaults.get("seed", 7)),
                settle_steps=int(image.defaults.get("settle_steps", 10)),
                render_backend=str(image.defaults.get("render_backend", "egl")),
                save_videos=bool(image.defaults.get("save_videos", False)) and not request.quick_test,
                cpu_threads=int(image.defaults.get("client_cpu_threads", 1)),
            )
        )

    @staticmethod
    def _runtime(image: BenchmarkImageManifest) -> dict[str, Path]:
        runtime = image.runtime
        return {
            "python": _required_path(runtime, "python"),
            "benchmark_root": _required_path(runtime, "benchmark_root"),
            "config_path": _required_path(runtime, "config_path"),
        }

    @staticmethod
    def _episodes(request: BenchmarkRunRequest, image: BenchmarkImageManifest) -> int:
        if request.quick_test:
            return 1
        value = request.episodes or int(image.defaults.get("episodes", 50))
        if value < 1:
            raise ValueError("LIBERO episodes must be positive")
        return value


def _required_path(runtime: dict[str, Any], name: str) -> Path:
    value = Path(str(runtime.get(name) or ""))
    if not value.is_absolute():
        raise ValueError(f"LIBERO runtime {name} must be an absolute path")
    return value
__all__ = ["LiberoBenchmarkFactory"]

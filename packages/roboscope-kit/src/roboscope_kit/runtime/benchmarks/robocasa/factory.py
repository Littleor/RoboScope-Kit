# Copyright 2026 RoboScope Team
# SPDX-License-Identifier: Apache-2.0

"""Factory for the official RoboCasa GR1 Tabletop evaluator."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from roboscope_kit.runtime.benchmark.factories import (
    BenchmarkRunRequest,
    representative_indices,
)
from roboscope_kit.runtime.benchmark.image_manifest import BenchmarkImageManifest

from .adapter import (
    ROBOCASA_TABLETOP_SUITE,
    ROBOCASA_TABLETOP_TASKS,
    RoboCasaBenchmarkAdapter,
    RoboCasaRuntimeConfig,
)


class RoboCasaBenchmarkFactory:
    @staticmethod
    def validate_suite(suite: str) -> None:
        if suite != ROBOCASA_TABLETOP_SUITE:
            raise ValueError(
                f"robocasa does not support suite {suite!r}; supported suite: {ROBOCASA_TABLETOP_SUITE}"
            )

    def create(
        self,
        request: BenchmarkRunRequest,
        image: BenchmarkImageManifest,
    ) -> RoboCasaBenchmarkAdapter:
        tasks = ROBOCASA_TABLETOP_TASKS[:1] if request.quick_test else ROBOCASA_TABLETOP_TASKS
        return self._create(request, image, tasks, self._episodes(request, image))

    def create_probe(
        self,
        request: BenchmarkRunRequest,
        image: BenchmarkImageManifest,
        concurrency: int,
    ) -> RoboCasaBenchmarkAdapter:
        del concurrency
        indices = representative_indices(
            len(ROBOCASA_TABLETOP_TASKS),
            int(image.defaults.get("probe_task_count", 10)),
        )
        tasks = tuple(ROBOCASA_TABLETOP_TASKS[index] for index in indices)
        episodes = int(image.defaults.get("probe_episodes_per_task", 1))
        return self._create(request, image, tasks, episodes)

    def self_test(self, image: BenchmarkImageManifest, output_dir: Path) -> dict[str, Any]:
        python = _required_path(image.runtime, "python")
        benchmark_root = _required_path(image.runtime, "benchmark_root")
        required = (
            python,
            benchmark_root,
            benchmark_root
            / "examples"
            / "simBenchmarks"
            / "Robocasa_tabletop"
            / "eval_files"
            / "simulation_env.py",
        )
        missing = [str(path) for path in required if not path.exists()]
        if missing:
            raise RuntimeError(f"RoboCasa image paths are missing: {', '.join(missing)}")
        output_dir.mkdir(parents=True, exist_ok=True)
        return {
            "benchmark": "robocasa",
            "python": str(python),
            "benchmark_root": str(benchmark_root),
            "task_count": len(ROBOCASA_TABLETOP_TASKS),
        }

    def _create(
        self,
        request: BenchmarkRunRequest,
        image: BenchmarkImageManifest,
        tasks: tuple[tuple[str, str], ...],
        episodes: int,
    ) -> RoboCasaBenchmarkAdapter:
        self.validate_suite(request.suite)
        return RoboCasaBenchmarkAdapter(
            RoboCasaRuntimeConfig(
                python=_required_path(image.runtime, "python"),
                benchmark_root=_required_path(image.runtime, "benchmark_root"),
                suite=request.suite,
                tasks=tasks,
                episodes_per_task=episodes,
                action_steps=int(image.defaults.get("action_steps", 12)),
                max_episode_steps=int(image.defaults.get("max_episode_steps", 720)),
                save_videos=bool(image.defaults.get("save_videos", False)) and not request.quick_test,
                cpu_threads=int(image.defaults.get("client_cpu_threads", 1)),
                render_backend=str(image.defaults.get("render_backend", "egl")),
            )
        )

    @staticmethod
    def _episodes(request: BenchmarkRunRequest, image: BenchmarkImageManifest) -> int:
        if request.quick_test:
            return 1
        value = request.episodes or int(image.defaults.get("episodes", 50))
        if value < 1:
            raise ValueError("RoboCasa episodes must be positive")
        return value


def _required_path(runtime: dict[str, Any], name: str) -> Path:
    value = Path(str(runtime.get(name) or ""))
    if not value.is_absolute():
        raise ValueError(f"RoboCasa runtime {name} must be an absolute path")
    return value


__all__ = ["RoboCasaBenchmarkFactory"]

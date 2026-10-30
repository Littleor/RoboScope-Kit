# Copyright 2026 RoboScope Team
# SPDX-License-Identifier: Apache-2.0

"""Factory for the official MetaWorld MT50 evaluator."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any

from roboscope_kit.runtime.benchmark.factories import BenchmarkRunRequest
from roboscope_kit.runtime.benchmark.image_manifest import BenchmarkImageManifest

from .adapter import (
    EPISODE_HORIZON,
    MT50_GOALS_PER_TASK,
    MT50_TASKS,
    MetaworldBenchmarkAdapter,
    MetaworldRuntimeConfig,
)
from .config import load_evaluation_config

MT50_SUITES = frozenset({"mt50"})
QUICK_TEST_TASK = "reach-v3"


class MetaworldBenchmarkFactory:
    @staticmethod
    def validate_suite(suite: str) -> None:
        if suite not in MT50_SUITES:
            supported = ", ".join(sorted(MT50_SUITES))
            raise ValueError(f"metaworld does not support suite {suite!r}; supported suites: {supported}")

    def create(
        self,
        request: BenchmarkRunRequest,
        image: BenchmarkImageManifest,
    ) -> MetaworldBenchmarkAdapter:
        tasks = (QUICK_TEST_TASK,) if request.quick_test else MT50_TASKS
        return self._create(request, image, tasks, self._episodes(request, image))

    def create_probe(
        self,
        request: BenchmarkRunRequest,
        image: BenchmarkImageManifest,
        concurrency: int,
    ) -> MetaworldBenchmarkAdapter:
        del concurrency
        episodes = int(image.defaults.get("probe_episodes", 1))
        return self._create(request, image, (QUICK_TEST_TASK,), episodes)

    def self_test(self, image: BenchmarkImageManifest, output_dir: Path) -> dict[str, Any]:
        python = _required_path(image.runtime, "python")
        if not python.is_file():
            raise RuntimeError(f"MetaWorld runtime python is missing: {python}")
        subprocess.run(
            (str(python), "-c", "import metaworld.evaluation"),
            check=True,
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        return {
            "benchmark": "metaworld",
            "python": str(python),
            "suite": "mt50",
            "task_count": len(MT50_TASKS),
            "episode_horizon": EPISODE_HORIZON,
        }

    def _create(
        self,
        request: BenchmarkRunRequest,
        image: BenchmarkImageManifest,
        tasks: tuple[str, ...],
        episodes: int,
    ) -> MetaworldBenchmarkAdapter:
        self.validate_suite(request.suite)
        if request.evaluation_config_path is None:
            raise ValueError("metaworld requires a resolved evaluation-config path")
        return MetaworldBenchmarkAdapter(
            MetaworldRuntimeConfig(
                python=_required_path(image.runtime, "python"),
                suite=request.suite,
                tasks=tasks,
                episodes_per_task=episodes,
                evaluation_config=load_evaluation_config(request.evaluation_config_path),
                evaluation_config_path=request.evaluation_config_path,
                cpu_threads=int(image.defaults.get("client_cpu_threads", 1)),
                render_backend=str(image.defaults.get("render_backend", "egl")),
            )
        )

    @staticmethod
    def _episodes(request: BenchmarkRunRequest, image: BenchmarkImageManifest) -> int:
        if request.quick_test:
            return 1
        value = request.episodes or int(image.defaults.get("episodes", MT50_GOALS_PER_TASK))
        if not 1 <= value <= MT50_GOALS_PER_TASK:
            raise ValueError(
                f"MetaWorld episodes per task must be in [1, {MT50_GOALS_PER_TASK}]; received {value}"
            )
        return value


def _required_path(runtime: dict[str, Any], name: str) -> Path:
    value = Path(str(runtime.get(name) or ""))
    if not value.is_absolute():
        raise ValueError(f"MetaWorld runtime {name} must be an absolute path")
    return value


__all__ = ["MT50_SUITES", "MetaworldBenchmarkFactory"]

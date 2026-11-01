# Copyright 2026 RoboScope Team
# SPDX-License-Identifier: Apache-2.0

"""Factory for the official MetaWorld MT50 evaluator."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any

from roboscope_kit.runtime.benchmark.autotune import probe_items_per_worker
from roboscope_kit.runtime.benchmark.factories import BenchmarkRunRequest, nested_task_prefix
from roboscope_kit.runtime.benchmark.manifest import BenchmarkManifest

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
        manifest: BenchmarkManifest,
    ) -> MetaworldBenchmarkAdapter:
        tasks = (QUICK_TEST_TASK,) if request.quick_test else MT50_TASKS
        return self._create(request, manifest, tasks, self._episodes(request, manifest))

    def create_probe(
        self,
        request: BenchmarkRunRequest,
        manifest: BenchmarkManifest,
        concurrency: int,
    ) -> MetaworldBenchmarkAdapter:
        episodes = int(manifest.defaults.get("probe_episodes", 1))
        # A MetaWorld work item is one task, so every worker needs items_per_worker
        # tasks. Tasks are a prefix of one nested order, so each probe repeats the
        # tasks of every smaller probe.
        items_per_worker = probe_items_per_worker(manifest.defaults)
        if items_per_worker > 0:
            indices = nested_task_prefix(len(MT50_TASKS), items_per_worker * concurrency)
            tasks = tuple(MT50_TASKS[index] for index in indices)
        else:
            tasks = (QUICK_TEST_TASK,)
        return self._create(request, manifest, tasks, episodes)

    def self_test(self, manifest: BenchmarkManifest, output_dir: Path) -> dict[str, Any]:
        python = manifest.runtime_path("python")
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
        manifest: BenchmarkManifest,
        tasks: tuple[str, ...],
        episodes: int,
    ) -> MetaworldBenchmarkAdapter:
        self.validate_suite(request.suite)
        if request.evaluation_config_path is None:
            raise ValueError("metaworld requires a resolved evaluation-config path")
        return MetaworldBenchmarkAdapter(
            MetaworldRuntimeConfig(
                python=manifest.runtime_path("python"),
                suite=request.suite,
                tasks=tasks,
                episodes_per_task=episodes,
                evaluation_config=load_evaluation_config(request.evaluation_config_path),
                evaluation_config_path=request.evaluation_config_path,
                cpu_threads=int(manifest.defaults.get("client_cpu_threads", 1)),
                render_backend=str(manifest.defaults.get("render_backend", "egl")),
            )
        )

    @staticmethod
    def _episodes(request: BenchmarkRunRequest, manifest: BenchmarkManifest) -> int:
        if request.quick_test:
            return 1
        value = request.episodes or int(manifest.defaults.get("episodes", MT50_GOALS_PER_TASK))
        if not 1 <= value <= MT50_GOALS_PER_TASK:
            raise ValueError(
                f"MetaWorld episodes per task must be in [1, {MT50_GOALS_PER_TASK}]; received {value}"
            )
        return value



__all__ = ["MT50_SUITES", "MetaworldBenchmarkFactory"]

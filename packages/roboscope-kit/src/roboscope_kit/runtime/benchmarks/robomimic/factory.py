# Copyright 2026 RoboScope Team
# SPDX-License-Identifier: Apache-2.0

"""Factory for the robomimic v0.1 rollout adapter."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any

from roboscope_kit.runtime.benchmark.autotune import probe_items_per_worker
from roboscope_kit.runtime.benchmark.factories import BenchmarkRunRequest
from roboscope_kit.runtime.benchmark.kit_path import kit_import_root
from roboscope_kit.runtime.benchmark.manifest import BenchmarkManifest

from .adapter import RobomimicBenchmarkAdapter, RobomimicRuntimeConfig
from .tasks import EPISODES_PER_TASK, QUICK_TEST_TASK, SUITES, TASK_NAMES


class RobomimicBenchmarkFactory:
    @staticmethod
    def validate_suite(suite: str) -> None:
        if suite not in SUITES:
            supported = ", ".join(SUITES)
            raise ValueError(f"robomimic does not support suite {suite!r}; supported suites: {supported}")

    def create(self, request: BenchmarkRunRequest, manifest: BenchmarkManifest) -> RobomimicBenchmarkAdapter:
        tasks = (QUICK_TEST_TASK,) if request.quick_test else TASK_NAMES
        return self._create(request, manifest, tasks, self._episodes(request, manifest))

    def create_probe(
        self,
        request: BenchmarkRunRequest,
        manifest: BenchmarkManifest,
        concurrency: int,
    ) -> RobomimicBenchmarkAdapter:
        # Every task, a growing prefix of seeds: each probe repeats the episodes of every
        # smaller probe, and the probe mixes long (Transport, Tool Hang) and short tasks
        # in the formal run's proportions.
        episodes = int(manifest.defaults.get("probe_episodes_per_task", 1))
        items_per_worker = probe_items_per_worker(manifest.defaults)
        if items_per_worker > 0:
            episodes = max(episodes, -(-items_per_worker * concurrency // len(TASK_NAMES)))
        return self._create(request, manifest, TASK_NAMES, min(episodes, EPISODES_PER_TASK))

    def self_test(self, manifest: BenchmarkManifest, output_dir: Path) -> dict[str, Any]:
        python = manifest.runtime_path("python")
        if not python.is_file():
            raise RuntimeError(f"robomimic runtime python is missing: {python}")
        # Regenerates the Lift initial states through the worker's own reset path and
        # checks them against the frozen digest; the build checks every task.
        subprocess.run(
            (
                str(python),
                "-m",
                "roboscope_kit.runtime.benchmarks.robomimic.initial_states",
                "--task",
                QUICK_TEST_TASK,
            ),
            check=True,
            env={
                **os.environ,
                "PYTHONPATH": str(kit_import_root()),
                "CUDA_VISIBLE_DEVICES": "",
                "PYTHONDONTWRITEBYTECODE": "1",
            },
            stdout=subprocess.DEVNULL,
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        return {
            "benchmark": "robomimic",
            "python": str(python),
            "suites": list(SUITES),
            "tasks": list(TASK_NAMES),
            "initial_states_verified": [QUICK_TEST_TASK],
        }

    def _create(
        self,
        request: BenchmarkRunRequest,
        manifest: BenchmarkManifest,
        tasks: tuple[str, ...],
        episodes: int,
    ) -> RobomimicBenchmarkAdapter:
        self.validate_suite(request.suite)
        return RobomimicBenchmarkAdapter(
            RobomimicRuntimeConfig(
                python=manifest.runtime_path("python"),
                suite=request.suite,
                tasks=tasks,
                episodes_per_task=episodes,
                chunk_size=int(manifest.defaults.get("chunk_size", 1)),
                cpu_threads=int(manifest.defaults.get("client_cpu_threads", 1)),
            )
        )

    @staticmethod
    def _episodes(request: BenchmarkRunRequest, manifest: BenchmarkManifest) -> int:
        if request.quick_test:
            return 1
        value = request.episodes or int(manifest.defaults.get("episodes", EPISODES_PER_TASK))
        if not 1 <= value <= EPISODES_PER_TASK:
            raise ValueError(
                f"robomimic episodes per task must be in [1, {EPISODES_PER_TASK}]; received {value}"
            )
        return value


__all__ = ["RobomimicBenchmarkFactory"]

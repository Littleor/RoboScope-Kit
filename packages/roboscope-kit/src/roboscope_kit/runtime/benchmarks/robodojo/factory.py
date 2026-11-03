# Copyright 2026 RoboScope Team
# SPDX-License-Identifier: Apache-2.0

"""Build RoboDojo adapters from the pinned simulation manifest."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from roboscope_kit.benchmark_specs import BenchmarkName
from roboscope_kit.runtime.benchmark.autotune import probe_items_per_worker
from roboscope_kit.runtime.benchmark.factories import BenchmarkRunRequest, nested_task_prefix
from roboscope_kit.runtime.benchmark.manifest import BenchmarkManifest

from .adapter import RoboDojoBenchmarkAdapter, RoboDojoRuntimeConfig
from .tasks import SUITES, TASK_EPISODES

SOURCE_REVISION = "f57e1633ffed59761bc8e568cfe7530c4b33381c"


class RoboDojoBenchmarkFactory:
    @staticmethod
    def validate_suite(suite: str) -> None:
        if suite not in SUITES:
            raise ValueError(f"unsupported RoboDojo suite: {suite!r}")

    @staticmethod
    def _validate_manifest(manifest: BenchmarkManifest) -> None:
        protocol = manifest.benchmark_protocols.get(BenchmarkName.ROBODOJO, {})
        if protocol.get("source_revision") != SOURCE_REVISION or protocol.get("tasks") != TASK_EPISODES:
            raise ValueError("RoboDojo manifest does not match the pinned task protocol")

    def create(self, request: BenchmarkRunRequest, manifest: BenchmarkManifest) -> RoboDojoBenchmarkAdapter:
        self.validate_suite(request.suite)
        tasks = SUITES[request.suite][:1] if request.quick_test else SUITES[request.suite]
        if request.quick_test:
            counts = {task: 1 for task in tasks}
        elif request.episodes is None:
            counts = {task: TASK_EPISODES[task] for task in tasks}
        else:
            counts = {task: request.episodes for task in tasks}
        return self._create(
            manifest, request.suite, tasks, counts, int(manifest.defaults["episode_shard_size"])
        )

    def create_probe(
        self, request: BenchmarkRunRequest, manifest: BenchmarkManifest, concurrency: int
    ) -> RoboDojoBenchmarkAdapter:
        self.validate_suite(request.suite)
        if concurrency < 1:
            raise ValueError("RoboDojo probe concurrency must be positive")
        available = SUITES[request.suite]
        wanted = max(
            int(manifest.defaults.get("probe_task_count", 3)),
            max(1, probe_items_per_worker(manifest.defaults)) * concurrency,
        )
        indices = nested_task_prefix(len(available), wanted)
        tasks = tuple(available[index] for index in indices)
        per_task = max(1, -(-wanted // len(tasks)))
        counts = {task: min(per_task, TASK_EPISODES[task]) for task in tasks}
        return self._create(manifest, request.suite, tasks, counts, 1)

    def self_test(self, manifest: BenchmarkManifest, output_dir: Path) -> dict[str, Any]:
        self._validate_manifest(manifest)
        python = manifest.runtime_path("python")
        root = manifest.runtime_path("benchmark_root")
        required = (
            python,
            root / "src/eval_client/main.py",
            root / "XPolicyLab/policy/RoboScopeKit/deploy.py",
        )
        missing = [str(path) for path in required if not path.exists()]
        seed = int(manifest.defaults.get("seed", 1))
        layouts = root / "Assets/Eval_Layout/RoboDojo/arx_x5" / str(seed)
        for task, count in TASK_EPISODES.items():
            missing.extend(str(layouts / f"{task}_{i}.json") for i in range(count)
                           if not (layouts / f"{task}_{i}.json").is_file())
        if missing:
            raise RuntimeError(f"RoboDojo runtime or layouts are missing: {', '.join(missing[:10])}")
        output_dir.mkdir(parents=True, exist_ok=True)
        return {"benchmark": "robodojo", "task_count": len(TASK_EPISODES), "seed": seed}

    def _create(
        self, manifest: BenchmarkManifest, suite: str, tasks: tuple[str, ...],
        counts: dict[str, int], shard_size: int,
    ) -> RoboDojoBenchmarkAdapter:
        self._validate_manifest(manifest)
        return RoboDojoBenchmarkAdapter(RoboDojoRuntimeConfig(
            python=manifest.runtime_path("python"),
            benchmark_root=manifest.runtime_path("benchmark_root"),
            suite=suite,
            tasks=tasks,
            episodes=counts,
            episode_shard_size=shard_size,
            seed=int(manifest.defaults.get("seed", 1)),
            cpu_threads=int(manifest.defaults.get("client_cpu_threads", 2)),
            async_video=manifest.defaults.get("async_video", False),
            unpaced_render=manifest.defaults.get("unpaced_render", False),
            control_state_reuse=manifest.defaults.get("control_state_reuse", False),
            physics_threads=manifest.defaults.get("physics_threads"),
            observation_queue_size=manifest.defaults.get("observation_queue_size", 0),
        ))

# Copyright 2026 RoboScope Team
# SPDX-License-Identifier: Apache-2.0

"""Official LIBERO+ work manifest and SDK-native evaluator process adapter.

LIBERO+ runs on the same simulator build as LIBERO, so evaluator processes are
launched from the shared ``runtime.benchmarks.libero.worker`` module; this
adapter owns the LIBERO+ protocol: suite allow-list, episode counts,
and scoring metadata are fixed by the LIBERO+ Benchmark definition.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from roboscope_kit.runtime.benchmark import (
    ProcessSpec,
    WorkerContext,
    WorkItem,
    WorkResult,
)
from roboscope_kit.runtime.benchmark.kit_path import kit_import_root
from roboscope_kit.runtime.benchmarks.libero.adapter import (
    stratified_task_ids,
    write_libero_config,
)


@dataclass(frozen=True)
class LiberoPlusRuntimeConfig:
    python: Path
    libero_home: Path
    libero_config_path: Path
    suite: str
    num_tasks: int
    num_trials: int
    task_ids: tuple[int, ...] | None = None
    sample_size: int | None = None
    sample_seed: int = 7
    chunk_size: int = 1
    seed: int = 7
    settle_steps: int = 10
    render_backend: str = "egl"
    save_videos: bool = False
    cpu_threads: int = 1

    def __post_init__(self) -> None:
        if self.num_tasks < 1 or self.num_trials < 1:
            raise ValueError("LIBERO+ task and trial counts must be positive")
        if self.chunk_size < 1 or self.settle_steps < 0:
            raise ValueError("LIBERO+ chunk size must be positive and settle steps cannot be negative")
        if self.cpu_threads < 1:
            raise ValueError("LIBERO+ cpu_threads must be positive")
        if self.render_backend not in {"egl", "osmesa"}:
            raise ValueError(f"unsupported LIBERO+ render backend: {self.render_backend}")


class LiberoPlusBenchmarkAdapter:
    """Run the official LIBERO+ protocol against an SDK Policy process."""

    def __init__(self, config: LiberoPlusRuntimeConfig) -> None:
        self.config = config
        self.task_ids = stratified_task_ids(config)

    def work_items(self) -> Sequence[WorkItem]:
        items: list[WorkItem] = []
        ordinal = 0
        for start_episode in range(0, self.config.num_trials, self.config.chunk_size):
            for task_id in self.task_ids:
                count = min(self.config.chunk_size, self.config.num_trials - start_episode)
                items.append(
                    WorkItem(
                        ordinal=ordinal,
                        item_id=(
                            f"task-{task_id:04d}-episodes-{start_episode:04d}-{start_episode + count:04d}"
                        ),
                        group_id=f"task-{task_id:04d}",
                        payload={
                            "task_id": task_id,
                            "start_episode": start_episode,
                            "num_trials": count,
                        },
                        expected_results=count,
                    )
                )
                ordinal += 1
        return items

    def worker_process(self, context: WorkerContext) -> ProcessSpec:
        runtime_config_path = (
            context.state_dir / "libero-config" / self.config.libero_config_path.name
        )
        write_libero_config(runtime_config_path, self.config.libero_home)
        command = (
            str(self.config.python),
            "-m",
            "roboscope_kit.runtime.benchmarks.libero.worker",
            "--worker-id",
            str(context.worker_id),
            "--policy-socket",
            str(context.replica_socket),
            "--suite",
            self.config.suite,
            "--seed",
            str(self.config.seed),
            "--settle-steps",
            str(self.config.settle_steps),
            "--queue-dir",
            str(context.queue_dir),
            "--worker-state-dir",
            str(context.state_dir),
        )
        if self.config.save_videos:
            command += ("--video-dir", str(context.output_dir / "video"))
        threads = str(self.config.cpu_threads)
        env = {
            "PYTHONPATH": f"{self.config.libero_home}:{kit_import_root()}:{os.environ.get('PYTHONPATH', '')}",
            "LIBERO_HOME": str(self.config.libero_home),
            "LIBERO_CONFIG_PATH": str(runtime_config_path.parent),
            "MUJOCO_GL": self.config.render_backend,
            "PYOPENGL_PLATFORM": self.config.render_backend,
            "OMP_NUM_THREADS": threads,
            "MKL_NUM_THREADS": threads,
            "OPENBLAS_NUM_THREADS": threads,
            "NUMEXPR_NUM_THREADS": threads,
            # Runtimes may live on read-only or inode-limited storage; never write pyc.
            "PYTHONDONTWRITEBYTECODE": "1",
        }
        if self.config.render_backend == "egl":
            env["CUDA_VISIBLE_DEVICES"] = context.gpu_device
            env["MUJOCO_EGL_DEVICE_ID"] = context.gpu_device
        return ProcessSpec(command=command, env=env)

    def aggregate(self, results: Sequence[WorkResult]) -> dict[str, Any]:
        tasks: dict[int, dict[str, Any]] = {}
        inference_wait_s = 0.0
        episode_elapsed_s = 0.0
        for result in results:
            inference_wait_s += float(result.metadata.get("inference_wait_s", 0.0))
            episode_elapsed_s += sum(record.elapsed_s for record in result.records)
            task_id = int(result.metadata["task_id"])
            task = tasks.setdefault(
                task_id,
                {
                    "task_id": task_id,
                    "task_description": result.metadata["task_description"],
                    "episodes": 0,
                    "successes": 0,
                },
            )
            task["episodes"] += len(result.records)
            task["successes"] += sum(record.success for record in result.records)
        rows = []
        for task_id in self.task_ids:
            row = tasks.get(
                task_id,
                {"task_id": task_id, "task_description": "", "episodes": 0, "successes": 0},
            )
            row["success_rate"] = row["successes"] / row["episodes"] if row["episodes"] else 0.0
            rows.append(row)
        return {
            "tasks": rows,
            "runtime_metrics": {
                "inference_wait_s": inference_wait_s,
                "episode_elapsed_s": episode_elapsed_s,
                "inference_wait_ratio": inference_wait_s / max(episode_elapsed_s, 1e-9),
            },
        }

    def metadata(self) -> dict[str, Any]:
        return {
            "adapter": type(self).__name__,
            "benchmark": "libero_plus",
            "suite": self.config.suite,
            "protocol": "roboscope-policy-v1",
            "bound_hint": "gpu_infer",
            "task_ids": self.task_ids,
            "episodes_per_task": self.config.num_trials,
            "seed": self.config.seed,
            "render_backend": self.config.render_backend,
        }


__all__ = ["LiberoPlusBenchmarkAdapter", "LiberoPlusRuntimeConfig"]

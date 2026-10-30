# Copyright 2026 RoboScope Team
# SPDX-License-Identifier: Apache-2.0

"""Official StarVLA adapters for the LIBERO benchmark runtime."""

from __future__ import annotations

import json
import math
import os
import random
import socket
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from roboscope_kit.policy import PolicyExecutionContract
from roboscope_kit.runtime.benchmark import (
    ProcessSpec,
    ReplicaContext,
    WorkerContext,
    WorkItem,
    WorkResult,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
STARVLA_CHECKPOINT_SERVER = Path(__file__).with_name("starvla_checkpoint_server.py")


def stratified_task_ids(
    *,
    libero_home: Path,
    suite: str,
    num_tasks: int,
    sample_size: int | None,
    sample_seed: int,
) -> list[int]:
    """Select official LIBERO-Plus task variants without leaking this logic into the SDK."""

    if sample_size is None or sample_size >= num_tasks:
        return list(range(num_tasks))
    if sample_size < 1:
        raise ValueError("sample-size must be positive")
    path = libero_home / "libero" / "libero" / "benchmark" / "task_classification.json"
    rows = json.loads(path.read_text(encoding="utf-8"))[suite]
    if len(rows) != num_tasks:
        raise ValueError(f"classification contains {len(rows)} tasks, expected {num_tasks}")
    groups: dict[tuple[str, int], list[int]] = defaultdict(list)
    for row in rows:
        key = (row["category"], int(row.get("difficulty_level", 0)))
        groups[key].append(int(row["id"]) - 1)
    exact = {key: sample_size * len(ids) / len(rows) for key, ids in groups.items()}
    quotas = {key: math.floor(value) for key, value in exact.items()}
    remaining = sample_size - sum(quotas.values())
    ranked = sorted(groups, key=lambda key: (exact[key] - quotas[key], key), reverse=True)
    for key in ranked[:remaining]:
        quotas[key] += 1
    rng = random.Random(sample_seed)
    selected: list[int] = []
    for key in sorted(groups):
        selected.extend(rng.sample(groups[key], quotas[key]))
    return sorted(selected)


class StarVLAInferenceAdapter:
    """Launch the unmodified StarVLA policy server as SDK inference replicas."""

    def __init__(
        self,
        *,
        checkpoint: Path,
        starvla_dir: Path,
        python: Path,
        cpu_threads: int,
    ) -> None:
        self.checkpoint = checkpoint
        self.starvla_dir = starvla_dir
        self.python = python
        self.cpu_threads = cpu_threads

    def replica_process(self, context: ReplicaContext) -> ProcessSpec:
        thread_count = str(self.cpu_threads)
        return ProcessSpec(
            command=(
                str(self.python),
                str(STARVLA_CHECKPOINT_SERVER),
                "--ckpt_path",
                str(self.checkpoint),
                "--port",
                str(context.port),
                "--use_bf16",
                "--idle_timeout",
                "-1",
            ),
            cwd=self.starvla_dir,
            env={
                "CUDA_VISIBLE_DEVICES": context.gpu_device,
                "PYTHONPATH": (f"{self.starvla_dir}:{REPOSITORY_ROOT}:{os.environ.get('PYTHONPATH', '')}"),
                "OMP_NUM_THREADS": thread_count,
                "MKL_NUM_THREADS": thread_count,
                "OPENBLAS_NUM_THREADS": thread_count,
                "NUMEXPR_NUM_THREADS": thread_count,
            },
            ready_port=context.port,
        )

    def metadata(self) -> dict[str, Any]:
        return {
            "adapter": type(self).__name__,
            "checkpoint": str(self.checkpoint),
            "starvla_dir": str(self.starvla_dir),
            "cpu_threads": self.cpu_threads,
        }

    def initialize_replica(self, _context: ReplicaContext) -> None:
        return None

    def baseline_replicas(self, _contexts: Sequence[ReplicaContext]) -> None:
        return None

    def finalize_replicas(self, _contexts: Sequence[ReplicaContext]) -> None:
        return None

    def healthcheck_replica(self, context: ReplicaContext) -> None:
        with socket.create_connection(("127.0.0.1", context.port), timeout=1):
            pass

    def release_replica(self, _context: ReplicaContext) -> None:
        return None

    def maximum_workers_per_replica(self) -> None:
        return None

    def execution_contract(self) -> PolicyExecutionContract:
        return PolicyExecutionContract(
            benchmark="libero",
            action_representation="libero.delta-eef-v1",
        )


class LiberoBenchmarkAdapter:
    """Preserve the official LIBERO episode protocol behind the SDK interface."""

    def __init__(self, args: object) -> None:
        self.args = args
        if args.task_ids is None:
            self.task_ids = stratified_task_ids(
                libero_home=args.libero_home,
                suite=args.suite,
                num_tasks=args.num_tasks,
                sample_size=args.sample_size,
                sample_seed=args.sample_seed,
            )
        else:
            self.task_ids = list(args.task_ids)
            if len(self.task_ids) != len(set(self.task_ids)):
                raise ValueError("task_ids must be unique")
            if any(task_id < 0 or task_id >= args.num_tasks for task_id in self.task_ids):
                raise ValueError("task_ids contain an out-of-range task")

    def work_items(self) -> Sequence[WorkItem]:
        items: list[WorkItem] = []
        ordinal = 0
        for start_episode in range(0, self.args.num_trials, self.args.chunk_size):
            for task_id in self.task_ids:
                count = min(self.args.chunk_size, self.args.num_trials - start_episode)
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
        command = (
            str(self.args.libero_python),
            str(Path(__file__).with_name("libero_standard_worker.py")),
            "--worker-id",
            str(context.worker_id),
            "--port",
            str(context.replica_port),
            "--suite",
            self.args.suite,
            "--seed",
            str(self.args.seed),
            "--queue-dir",
            str(context.queue_dir),
            "--worker-state-dir",
            str(context.state_dir),
        )
        if self.args.save_videos:
            command += ("--video-dir", str(context.output_dir / "video"))
        thread_count = str(self.args.client_cpu_threads)
        repository_root = REPOSITORY_ROOT
        env = {
            "PYTHONPATH": (
                f"{self.args.libero_home}:{self.args.starvla_dir}:{repository_root}:"
                f"{os.environ.get('PYTHONPATH', '')}"
            ),
            "LIBERO_HOME": str(self.args.libero_home),
            "LIBERO_CONFIG_PATH": str(self.args.libero_config_path),
            "MUJOCO_GL": self.args.render_backend,
            "PYOPENGL_PLATFORM": self.args.render_backend,
            "OMP_NUM_THREADS": thread_count,
            "MKL_NUM_THREADS": thread_count,
            "OPENBLAS_NUM_THREADS": thread_count,
            "NUMEXPR_NUM_THREADS": thread_count,
        }
        if self.args.render_backend == "egl":
            env["MUJOCO_EGL_DEVICE_ID"] = context.gpu_device
        return ProcessSpec(command=command, env=env)

    def aggregate(self, results: Sequence[WorkResult]) -> dict[str, Any]:
        tasks: dict[int, dict[str, Any]] = {}
        for result in results:
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
        return {"tasks": [tasks[task_id] for task_id in sorted(tasks)]}

    def metadata(self) -> dict[str, Any]:
        return {
            "adapter": type(self).__name__,
            "benchmark": self.args.benchmark_name,
            "suite": self.args.suite,
            "protocol": "legacy-startbenchmark-v1",
            "task_ids": self.task_ids,
            "episodes_per_task": self.args.num_trials,
            "seed": self.args.seed,
            "render_backend": self.args.render_backend,
            "episode_sharding": self.args.allow_episode_sharding,
        }

# Copyright 2026 RoboScope Team
# SPDX-License-Identifier: Apache-2.0

"""robomimic v0.1 rollout work manifest and SDK-native worker process adapter."""

from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from roboscope_kit.runtime.benchmark import ProcessSpec, WorkerContext, WorkItem, WorkResult
from roboscope_kit.runtime.benchmark.kit_path import kit_import_root

from .tasks import (
    ACTION_REPRESENTATION,
    CONTROL_FREQ_HZ,
    EPISODES_PER_TASK,
    PROTOCOL_VERSION,
    SUCCESS_CRITERION,
    SUITES,
    TASK_NAMES,
    TASKS_BY_NAME,
    initial_states_sha256,
)


@dataclass(frozen=True)
class RobomimicRuntimeConfig:
    python: Path
    suite: str
    tasks: tuple[str, ...]
    episodes_per_task: int
    chunk_size: int = 1
    cpu_threads: int = 1

    def __post_init__(self) -> None:
        if self.suite not in SUITES:
            raise ValueError(f"unsupported robomimic suite: {self.suite}")
        if not self.tasks or len(set(self.tasks)) != len(self.tasks):
            raise ValueError("robomimic tasks must be unique and non-empty")
        unknown = sorted(set(self.tasks) - set(TASK_NAMES))
        if unknown:
            raise ValueError(f"unknown robomimic tasks: {', '.join(unknown)}")
        if not 1 <= self.episodes_per_task <= EPISODES_PER_TASK:
            raise ValueError(
                f"robomimic episodes per task must be in [1, {EPISODES_PER_TASK}] "
                "(episode i uses the frozen seed i)"
            )
        if self.chunk_size < 1 or self.cpu_threads < 1:
            raise ValueError("robomimic chunk_size and cpu_threads must be positive")

    @property
    def formal(self) -> bool:
        """Only every task with all 50 seeds is the full ``ph_low_dim`` evaluation."""

        return self.tasks == TASK_NAMES and self.episodes_per_task == EPISODES_PER_TASK


class RobomimicBenchmarkAdapter:
    """Run the robomimic v0.1 PH low-dim rollout protocol against an SDK Policy."""

    def __init__(self, config: RobomimicRuntimeConfig) -> None:
        self.config = config

    def work_items(self) -> Sequence[WorkItem]:
        items: list[WorkItem] = []
        episodes = self.config.episodes_per_task
        for start in range(0, episodes, self.config.chunk_size):
            seeds = list(range(start, min(start + self.config.chunk_size, episodes)))
            for task in self.config.tasks:
                items.append(
                    WorkItem(
                        ordinal=len(items),
                        item_id=f"{task}-seeds-{seeds[0]:04d}-{seeds[-1] + 1:04d}",
                        group_id=task,
                        payload={"task": task, "seeds": seeds},
                        expected_results=len(seeds),
                    )
                )
        return items

    def worker_process(self, context: WorkerContext) -> ProcessSpec:
        threads = str(self.config.cpu_threads)
        command = (
            str(self.config.python),
            "-m",
            "roboscope_kit.runtime.benchmarks.robomimic.worker",
            "--worker-id",
            str(context.worker_id),
            "--policy-socket",
            str(context.replica_socket),
            "--suite",
            self.config.suite,
            "--queue-dir",
            str(context.queue_dir),
            "--worker-state-dir",
            str(context.state_dir),
        )
        return ProcessSpec(
            command=command,
            env={
                "PYTHONPATH": ":".join((str(kit_import_root()), os.environ.get("PYTHONPATH", ""))),
                # Low-dim rollouts never render and never touch CUDA in the worker.
                "CUDA_VISIBLE_DEVICES": "",
                "MUJOCO_GL": "egl",
                "OMP_NUM_THREADS": threads,
                "MKL_NUM_THREADS": threads,
                "OPENBLAS_NUM_THREADS": threads,
                "NUMEXPR_NUM_THREADS": threads,
                "NUMBA_NUM_THREADS": threads,
                "PYTHONDONTWRITEBYTECODE": "1",
            },
        )

    def aggregate(self, results: Sequence[WorkResult]) -> dict[str, Any]:
        per_task = self._collect(results)
        rows = [self._row(task, per_task.get(task, [])) for task in self.config.tasks]
        official: dict[str, Any] = {
            "metric": "per_task_success_rate",
            "formal": self.config.formal,
            "tasks": rows,
        }
        missing = [row["task"] for row in rows if row["episodes"] != self.config.episodes_per_task]
        if self.config.formal and not missing:
            for task in self.config.tasks:
                actual = initial_states_sha256(
                    (int(record.metadata["seed"]), str(record.metadata["initial_state_sha256"]))
                    for record in per_task[task]
                )
                expected = TASKS_BY_NAME[task].initial_states_sha256
                if actual != expected:
                    raise RuntimeError(
                        f"robomimic task {task} rolled out initial states {actual}, "
                        f"expected frozen {expected}"
                    )
            official["initial_states_verified"] = True
        return official

    def _row(self, task: str, records: list) -> dict[str, Any]:
        episodes = len(records)
        successes = sum(record.success for record in records)
        return {
            "task": task,
            "env_name": TASKS_BY_NAME[task].env_name,
            "episodes": episodes,
            "successes": successes,
            "success_rate": successes / episodes if episodes else 0.0,
        }

    def _collect(self, results: Sequence[WorkResult]) -> dict[str, list]:
        per_task: dict[str, list] = {}
        seen: set[str] = set()
        expected_seeds = set(range(self.config.episodes_per_task))
        for result in results:
            task = result.group_id
            if task not in self.config.tasks:
                raise RuntimeError(f"unexpected robomimic task result {task!r}")
            horizon = TASKS_BY_NAME[task].horizon
            for record in result.records:
                seed = int(record.metadata["seed"])
                if record.metadata.get("task") != task or seed not in expected_seeds:
                    raise RuntimeError(f"robomimic record {record.record_id} is outside the work manifest")
                if record.record_id != f"{task}-seed-{seed:04d}" or record.record_id in seen:
                    raise RuntimeError(f"duplicate or malformed robomimic record {record.record_id}")
                if not 1 <= record.steps <= horizon:
                    raise RuntimeError(
                        f"robomimic record {record.record_id} has invalid steps {record.steps}"
                    )
                if not record.success and record.steps != horizon:
                    raise RuntimeError(
                        f"robomimic record {record.record_id} failed before its {horizon}-step horizon"
                    )
                seen.add(record.record_id)
                per_task.setdefault(task, []).append(record)
        return per_task

    def metadata(self) -> dict[str, Any]:
        return {
            "adapter": type(self).__name__,
            "benchmark": "robomimic",
            "suite": self.config.suite,
            "protocol": PROTOCOL_VERSION,
            "metric": "per_task_success_rate",
            "formal": self.config.formal,
            "task_order": list(self.config.tasks),
            "tasks": [
                {
                    "task": name,
                    "env_name": TASKS_BY_NAME[name].env_name,
                    "robots": list(TASKS_BY_NAME[name].robots),
                    "horizon": TASKS_BY_NAME[name].horizon,
                    "initial_states_sha256": TASKS_BY_NAME[name].initial_states_sha256,
                }
                for name in self.config.tasks
            ],
            "episodes_per_task": self.config.episodes_per_task,
            "seeds": list(range(self.config.episodes_per_task)),
            "control_freq_hz": CONTROL_FREQ_HZ,
            "action_representation": ACTION_REPRESENTATION,
            "success_criterion": SUCCESS_CRITERION,
            "chunk_size": self.config.chunk_size,
        }


__all__ = ["RobomimicBenchmarkAdapter", "RobomimicRuntimeConfig"]

# Copyright 2026 RoboScope Team
# SPDX-License-Identifier: Apache-2.0

"""Work scheduling and result aggregation for RoboDojo simulation."""

from __future__ import annotations

import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from roboscope_kit.runtime.benchmark import ProcessSpec, WorkerContext, WorkItem, WorkResult

from .tasks import SUITES, TASK_EPISODES


@dataclass(frozen=True)
class RoboDojoRuntimeConfig:
    python: Path
    benchmark_root: Path
    suite: str
    tasks: tuple[str, ...]
    episodes: Mapping[str, int]
    episode_shard_size: int
    seed: int
    cpu_threads: int
    async_video: bool = False
    unpaced_render: bool = False
    control_state_reuse: bool = False
    physics_threads: int | None = None
    observation_queue_size: int = 0

    def __post_init__(self) -> None:
        if self.suite not in SUITES:
            raise ValueError(f"unsupported RoboDojo suite: {self.suite}")
        if not self.tasks or len(self.tasks) != len(set(self.tasks)):
            raise ValueError("RoboDojo tasks must be nonempty and unique")
        if set(self.tasks) - set(SUITES[self.suite]) or set(self.episodes) != set(self.tasks):
            raise ValueError("RoboDojo tasks and episode counts do not match the suite")
        if self.episode_shard_size < 1 or self.cpu_threads < 1 or self.seed < 0:
            raise ValueError("RoboDojo shard size and CPU threads must be positive; seed must be nonnegative")
        if any(type(value) is not bool for value in (
            self.async_video, self.unpaced_render, self.control_state_reuse,
        )):
            raise ValueError("RoboDojo runtime switches must be booleans")
        if self.physics_threads is not None and (
            type(self.physics_threads) is not int or self.physics_threads not in (1, 4, 8)
        ):
            raise ValueError("RoboDojo physics_threads must be null, 1, 4 or 8")
        if type(self.observation_queue_size) is not int or not 0 <= self.observation_queue_size <= 10:
            raise ValueError("RoboDojo observation_queue_size must be an integer from 0 to 10")
        for task, count in self.episodes.items():
            if type(count) is not int or not 1 <= count <= TASK_EPISODES[task]:
                raise ValueError(f"invalid episode count for RoboDojo task {task!r}: {count!r}")


class RoboDojoBenchmarkAdapter:
    def __init__(self, config: RoboDojoRuntimeConfig) -> None:
        self.config = config

    def work_items(self) -> Sequence[WorkItem]:
        items: list[WorkItem] = []
        shard_size = self.config.episode_shard_size
        for start in range(0, max(self.config.episodes.values()), shard_size):
            for task in self.config.tasks:
                count = min(shard_size, self.config.episodes[task] - start)
                if count <= 0:
                    continue
                item_id = f"{self.config.suite}-{task}-layouts-{start:04d}-{start + count:04d}"
                items.append(WorkItem(
                    ordinal=len(items),
                    item_id=item_id,
                    group_id=item_id,
                    payload={"task": task, "seed": self.config.seed, "start": start, "count": count},
                    expected_results=count,
                ))
        return tuple(items)

    def worker_process(self, context: WorkerContext) -> ProcessSpec:
        threads = str(self.config.cpu_threads)
        if self.config.observation_queue_size and not context.execution_contract.step_observation_updates:
            raise ValueError("RoboDojo observation overlap requires a Policy that declares step updates")
        return ProcessSpec(
            command=(
                sys.executable, "-I", "-m", "roboscope_kit.runtime.benchmarks.robodojo.worker",
                "--worker-id", str(context.worker_id),
                "--gpu-id", context.gpu_device,
                "--policy-socket", str(context.replica_socket),
                "--action-representation", context.execution_contract.action_representation,
                "--observation-queue-size", str(self.config.observation_queue_size),
                *(("--step-observation-updates",)
                  if context.execution_contract.step_observation_updates else ()),
                "--benchmark-python", str(self.config.python),
                "--benchmark-root", str(self.config.benchmark_root),
                "--queue-dir", str(context.queue_dir),
                "--worker-state-dir", str(context.state_dir),
                "--output-dir", str(context.output_dir),
                *(("--async-video",) if self.config.async_video else ()),
                *(("--unpaced-render",) if self.config.unpaced_render else ()),
                *(("--control-state-reuse",) if self.config.control_state_reuse else ()),
                *(("--physics-threads", str(self.config.physics_threads))
                  if self.config.physics_threads is not None else ()),
            ),
            cwd=self.config.benchmark_root,
            env={
                "OMP_NUM_THREADS": threads,
                "MKL_NUM_THREADS": threads,
                "OPENBLAS_NUM_THREADS": threads,
                "TOKENIZERS_PARALLELISM": "false",
            },
        )

    def aggregate(self, results: Sequence[WorkResult]) -> dict[str, Any]:
        rows: dict[str, dict[str, Any]] = {
            task: {"task": task, "episodes": 0, "successes": 0, "score_sum": 0.0}
            for task in self.config.tasks
        }
        for result in results:
            task = str(result.metadata["task"])
            row = rows[task]
            for record in result.records:
                row["episodes"] += 1
                row["successes"] += int(record.success)
                row["score_sum"] += float(record.metadata["score"])
        for row in rows.values():
            count = row["episodes"]
            row["success_rate"] = row["successes"] / count if count else 0.0
            row["mean_score"] = row["score_sum"] / count if count else 0.0
        return {"tasks": list(rows.values())}

    def metadata(self) -> dict[str, Any]:
        return {
            "adapter": type(self).__name__,
            "benchmark": "robodojo",
            "suite": self.config.suite,
            "protocol": "robodojo-v1",
            "tasks": list(self.config.tasks),
            "episodes_per_task": dict(self.config.episodes),
            "episode_shard_size": self.config.episode_shard_size,
            "seed": self.config.seed,
            "runtime_options": {
                "async_video": self.config.async_video,
                "unpaced_render": self.config.unpaced_render,
                "control_state_reuse": self.config.control_state_reuse,
                "physics_threads": self.config.physics_threads,
                "observation_queue_size": self.config.observation_queue_size,
            },
            "bound_hint": "gpu_infer",
        }

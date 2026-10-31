# Copyright 2026 RoboScope Team
# SPDX-License-Identifier: Apache-2.0

"""Official MetaWorld MT50 work manifest and SDK-native evaluator process adapter."""

from __future__ import annotations

import math
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

from .config import resolve_evaluation_config
from .instructions import (
    DIFFICULTY_BY_TASK,
    DIFFICULTY_MAPPING_SHA256,
    DIFFICULTY_MAPPING_VERSION,
    DIFFICULTY_TO_TASKS,
    FROZEN_MT50_TASKS_SHA256,
    MT50_TASKS,
)

MT50_TASK_COUNT = 50
MT50_GOALS_PER_TASK = 50
MT50_TOTAL_EPISODES = MT50_TASK_COUNT * MT50_GOALS_PER_TASK
EPISODE_HORIZON = 500
SEED = 42
UPSTREAM_RATE_TOLERANCE = 1e-9


def summarize_smoke(task_order: Sequence[str], per_task: dict[str, dict[str, Any]]) -> dict[str, Any]:
    rows = [
        {
            "task": slug,
            "difficulty": DIFFICULTY_BY_TASK[slug],
            "episodes": per_task[slug]["episodes"],
            "success_count": per_task[slug]["successes"],
            "success_rate": per_task[slug]["successes"] / per_task[slug]["episodes"],
        }
        for slug in task_order
    ]
    success_count = sum(row["success_count"] for row in rows)
    total_count = sum(row["episodes"] for row in rows)
    return {
        "metric": "task_success_rate",
        "formal": False,
        "note": "quick-test and single-task smoke runs do not produce the official "
        "difficulty macro success rate",
        "success_count": success_count,
        "total_count": total_count,
        "tasks": rows,
    }


def summarize_formal(per_task: dict[str, dict[str, Any]]) -> dict[str, Any]:
    rows = [
        {
            "task": slug,
            "difficulty": DIFFICULTY_BY_TASK[slug],
            "episodes": MT50_GOALS_PER_TASK,
            "success_count": per_task[slug]["successes"],
            "success_rate": per_task[slug]["successes"] / MT50_GOALS_PER_TASK,
        }
        for slug in MT50_TASKS
    ]
    difficulty_rates = {
        difficulty: sum(row["success_rate"] for row in rows if row["difficulty"] == difficulty)
        / len(tasks)
        for difficulty, tasks in DIFFICULTY_TO_TASKS.items()
    }
    success_count = sum(row["success_count"] for row in rows)
    return {
        "metric": "difficulty_macro_success_rate",
        "score": sum(difficulty_rates.values()) / len(difficulty_rates),
        "difficulty_rates": difficulty_rates,
        "task_mean_success_rate": success_count / MT50_TOTAL_EPISODES,
        "success_count": success_count,
        "total_count": MT50_TOTAL_EPISODES,
        "tasks": rows,
        "difficulty_mapping": {
            "version": DIFFICULTY_MAPPING_VERSION,
            "sha256": DIFFICULTY_MAPPING_SHA256,
        },
    }


def merge_task_results(partials: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Combine per-shard ``aggregate_partial`` outputs into the full-run official result."""

    if not partials:
        raise ValueError("no MetaWorld shard results to merge")
    for partial in partials:
        if partial.get("partial") is not True:
            raise ValueError("MetaWorld shard official_result is not a partial aggregate")
    formals = {bool(partial["formal"]) for partial in partials}
    orders = {tuple(partial["task_order"]) for partial in partials}
    if len(formals) != 1 or len(orders) != 1:
        raise ValueError("MetaWorld shards disagree on formal mode or task order")
    formal = formals.pop()
    task_order = orders.pop()
    per_task: dict[str, dict[str, Any]] = {}
    for partial in partials:
        for slug, task in partial["tasks"].items():
            if slug in per_task:
                raise ValueError(f"MetaWorld task {slug!r} reported by more than one shard")
            _validate_partial_task(slug, task, formal)
            per_task[slug] = task
    missing = sorted(set(task_order) - set(per_task))
    if missing:
        raise ValueError(f"MetaWorld shards do not cover tasks: {', '.join(missing)}")
    unexpected = sorted(set(per_task) - set(task_order))
    if unexpected:
        raise ValueError(f"MetaWorld shards reported unexpected tasks: {', '.join(unexpected)}")
    if formal:
        if set(task_order) != set(MT50_TASKS):
            raise ValueError("formal MetaWorld shards must cover exactly the MT50 tasks")
        return summarize_formal(per_task)
    return summarize_smoke(task_order, per_task)


def _validate_partial_task(slug: str, task: dict[str, Any], formal: bool) -> None:
    episodes, successes = task["episodes"], task["successes"]
    if type(episodes) is not int or episodes < 1:
        raise ValueError(f"MetaWorld task {slug} has invalid episode count")
    if type(successes) is not int or not 0 <= successes <= episodes:
        raise ValueError(f"MetaWorld task {slug} has invalid success count")
    if formal and episodes != MT50_GOALS_PER_TASK:
        raise ValueError(f"formal MetaWorld task {slug} must report {MT50_GOALS_PER_TASK} episodes")


@dataclass(frozen=True)
class MetaworldRuntimeConfig:
    python: Path
    suite: str
    tasks: tuple[str, ...]
    episodes_per_task: int
    evaluation_config: dict[str, Any]
    evaluation_config_path: Path
    episode_horizon: int = EPISODE_HORIZON
    seed: int = SEED
    cpu_threads: int = 1
    render_backend: str = "egl"

    def __post_init__(self) -> None:
        if self.suite != "mt50":
            raise ValueError(f"unsupported MetaWorld suite: {self.suite}")
        if not self.tasks or len(set(self.tasks)) != len(self.tasks):
            raise ValueError("MetaWorld tasks must be unique and non-empty")
        unknown = sorted(set(self.tasks) - set(MT50_TASKS))
        if unknown:
            raise ValueError(f"unknown MetaWorld task slugs: {', '.join(unknown)}")
        if not 1 <= self.episodes_per_task <= MT50_GOALS_PER_TASK:
            raise ValueError(
                f"MetaWorld episodes per task must be in [1, {MT50_GOALS_PER_TASK}] "
                "(the frozen artifact holds 50 train goals per task)"
            )
        if self.episode_horizon != EPISODE_HORIZON:
            raise ValueError(f"MetaWorld episode horizon is frozen at {EPISODE_HORIZON}")
        if self.seed != SEED:
            raise ValueError(f"MetaWorld seed is frozen at {SEED}")
        if self.cpu_threads < 1:
            raise ValueError("MetaWorld cpu_threads must be positive")
        if self.render_backend != "egl":
            raise ValueError("MetaWorld SDK evaluation currently requires EGL rendering")
        if resolve_evaluation_config(self.evaluation_config) != self.evaluation_config:
            raise ValueError("MetaWorld evaluation_config must be fully resolved")

    @property
    def formal(self) -> bool:
        """Only the complete 50-task x 50-goal pass produces the official macro score."""

        return self.tasks == MT50_TASKS and self.episodes_per_task == MT50_GOALS_PER_TASK


class MetaworldBenchmarkAdapter:
    """Run the RoboScope MetaWorld MT50 Protocol v1 against an SDK Policy."""

    def __init__(self, config: MetaworldRuntimeConfig) -> None:
        self.config = config

    def work_items(self) -> Sequence[WorkItem]:
        return tuple(
            WorkItem(
                ordinal=index,
                item_id=f"mt50-{slug}",
                group_id=slug,
                payload={"task": slug, "episodes": self.config.episodes_per_task},
                expected_results=self.config.episodes_per_task,
            )
            for index, slug in enumerate(self.config.tasks)
        )

    def worker_process(self, context: WorkerContext) -> ProcessSpec:
        threads = str(self.config.cpu_threads)
        command = (
            str(self.config.python),
            "-m",
            "roboscope_kit.runtime.benchmarks.metaworld.worker",
            "--worker-id",
            str(context.worker_id),
            "--policy-socket",
            str(context.replica_socket),
            "--evaluation-config",
            str(self.config.evaluation_config_path),
            "--queue-dir",
            str(context.queue_dir),
            "--worker-state-dir",
            str(context.state_dir),
        )
        return ProcessSpec(
            command=command,
            env={
                "PYTHONPATH": ":".join(
                    (
                        str(kit_import_root()),
                        os.environ.get("PYTHONPATH", ""),
                    )
                ),
                "CUDA_VISIBLE_DEVICES": context.gpu_device,
                "MUJOCO_GL": self.config.render_backend,
                "PYOPENGL_PLATFORM": self.config.render_backend,
                "MUJOCO_EGL_DEVICE_ID": context.gpu_device,
                "OMP_NUM_THREADS": threads,
                "MKL_NUM_THREADS": threads,
                "OPENBLAS_NUM_THREADS": threads,
                "NUMEXPR_NUM_THREADS": threads,
                "TOKENIZERS_PARALLELISM": "false",
                "PYTHONDONTWRITEBYTECODE": "1",
            },
        )

    def aggregate(self, results: Sequence[WorkResult]) -> dict[str, Any]:
        per_task = self._collect_per_task(results)
        missing = sorted(set(self.config.tasks) - set(per_task))
        if missing:
            raise RuntimeError(f"MetaWorld results are missing tasks: {', '.join(missing)}")
        if not self.config.formal:
            return summarize_smoke(self.config.tasks, per_task)
        return summarize_formal(per_task)

    def aggregate_partial(self, results: Sequence[WorkResult]) -> dict[str, Any]:
        """Report the tasks one shard ran, without the every-task check or derived metrics.

        The difficulty macro score is only meaningful over all tasks, so it is left to
        ``merge_task_results`` once every shard has reported.
        """

        per_task = self._collect_per_task(results)
        return {
            "partial": True,
            "formal": self.config.formal,
            "task_order": list(self.config.tasks),
            "tasks": {slug: per_task[slug] for slug in self.config.tasks if slug in per_task},
        }

    def _collect_per_task(self, results: Sequence[WorkResult]) -> dict[str, dict[str, Any]]:
        per_task: dict[str, dict[str, Any]] = {}
        for result in results:
            slug = result.group_id
            if slug not in self.config.tasks:
                raise RuntimeError(f"unexpected MetaWorld task result {slug!r}")
            if slug in per_task:
                raise RuntimeError(f"duplicate MetaWorld task result {slug!r}")
            per_task[slug] = self._validate_task_records(result)
        return per_task

    def _validate_task_records(self, result: WorkResult) -> dict[str, Any]:
        slug = result.group_id
        expected = self.config.episodes_per_task
        records = result.records
        if len(records) != expected:
            raise RuntimeError(f"MetaWorld task {slug} produced {len(records)} records; expected {expected}")
        record_ids = [record.record_id for record in records]
        if len(set(record_ids)) != len(record_ids):
            raise RuntimeError(f"MetaWorld task {slug} produced duplicate record ids")
        goal_ids = [int(record.metadata["goal_id"]) for record in records]
        if len(set(goal_ids)) != expected or min(goal_ids) < 0 or max(goal_ids) >= MT50_GOALS_PER_TASK:
            raise RuntimeError(f"MetaWorld task {slug} records do not cover {expected} unique train goals")
        for record, goal_id in zip(records, goal_ids, strict=True):
            if record.record_id != f"{slug}-goal-{goal_id:02d}":
                raise RuntimeError(f"unexpected MetaWorld record id {record.record_id}")
            if record.steps < 1 or record.steps > self.config.episode_horizon:
                raise RuntimeError(
                    f"MetaWorld record {record.record_id} has invalid step count {record.steps}"
                )
        successes = sum(record.success for record in records)
        upstream_rate = float(result.metadata["upstream_task_success_rate"])
        if not math.isfinite(upstream_rate):
            raise RuntimeError(f"MetaWorld task {slug} upstream success rate is not finite")
        if abs(upstream_rate - successes / expected) > UPSTREAM_RATE_TOLERANCE:
            raise RuntimeError(
                f"MetaWorld task {slug} record success rate {successes / expected} disagrees with "
                f"upstream evaluation() rate {upstream_rate}"
            )
        return {"successes": successes, "episodes": expected, "upstream_rate": upstream_rate}

    def metadata(self) -> dict[str, Any]:
        return {
            "adapter": type(self).__name__,
            "benchmark": "metaworld",
            "suite": self.config.suite,
            "protocol": "metaworld-v1",
            "metric": "difficulty_macro_success_rate",
            "formal": self.config.formal,
            "task_count": len(self.config.tasks),
            "task_order": list(self.config.tasks),
            "goals_source": "MT50.train_tasks",
            "mt50_train_tasks_sha256": FROZEN_MT50_TASKS_SHA256,
            "goals": [
                {"task": slug, "goal_index": goal_index}
                for slug in self.config.tasks
                for goal_index in range(self.config.episodes_per_task)
            ],
            "seeds": {
                "goal_generation": self.config.seed,
                "environment": self.config.seed,
                "policy": None,
            },
            "difficulty_mapping": {
                "version": DIFFICULTY_MAPPING_VERSION,
                "sha256": DIFFICULTY_MAPPING_SHA256,
            },
            "evaluation_config": self.config.evaluation_config,
            "episode_horizon": self.config.episode_horizon,
            "episodes_per_task": self.config.episodes_per_task,
            "render_backend": self.config.render_backend,
        }


__all__ = [
    "EPISODE_HORIZON",
    "MT50_GOALS_PER_TASK",
    "MT50_TASK_COUNT",
    "MT50_TOTAL_EPISODES",
    "SEED",
    "MetaworldBenchmarkAdapter",
    "MetaworldRuntimeConfig",
    "merge_task_results",
    "summarize_formal",
    "summarize_smoke",
]

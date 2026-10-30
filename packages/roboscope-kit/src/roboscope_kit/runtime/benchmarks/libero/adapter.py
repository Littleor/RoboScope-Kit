# Copyright 2026 RoboScope Team
# SPDX-License-Identifier: Apache-2.0

"""Official LIBERO work manifest and SDK-native evaluator process adapter."""

from __future__ import annotations

import importlib
import json
import math
import os
import random
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from roboscope_kit.runtime.benchmark import (
    ProcessSpec,
    WorkerContext,
    WorkItem,
    WorkResult,
)

SDK_IMPORT_ROOT = Path(__file__).resolve().parents[4]


def _libero_paths(libero_home: Path) -> dict[str, Path]:
    benchmark_root = libero_home / "libero" / "libero"
    return {
        "benchmark_root": benchmark_root,
        "bddl_files": benchmark_root / "bddl_files",
        "init_states": benchmark_root / "init_files",
        "datasets": benchmark_root.parent / "datasets",
        "assets": benchmark_root / "assets",
    }


def write_libero_config(config_path: Path, libero_home: Path) -> None:
    paths = _libero_paths(libero_home)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        "".join(f"{name}: {path}\n" for name, path in paths.items()),
        encoding="utf-8",
    )


def validate_libero_runtime(config_path: Path, libero_home: Path, sdk_import_root: Path) -> None:
    expected_sdk_root = sdk_import_root.resolve()
    if expected_sdk_root != SDK_IMPORT_ROOT:
        raise RuntimeError(f"SDK imported from {SDK_IMPORT_ROOT}, expected {expected_sdk_root}")

    libero_package = importlib.import_module("libero.libero")
    sdk_worker = importlib.import_module("roboscope_kit.runtime.benchmarks.libero.worker")
    expected_paths = _libero_paths(libero_home.resolve())
    package_path = Path(str(libero_package.__file__)).resolve()
    expected_package_path = (expected_paths["benchmark_root"] / "__init__.py").resolve()
    if package_path != expected_package_path:
        raise RuntimeError(f"LIBERO imported from {package_path}, expected {expected_package_path}")
    worker_path = Path(str(sdk_worker.__file__)).resolve()
    if expected_sdk_root not in worker_path.parents:
        raise RuntimeError(f"SDK worker imported from {worker_path}, expected {expected_sdk_root}")
    expected_config_path = config_path.resolve()
    active_config_path = Path(str(libero_package.config_file)).resolve()
    if active_config_path != expected_config_path:
        raise RuntimeError(f"LIBERO uses config {active_config_path}, expected {expected_config_path}")
    if not expected_config_path.is_file():
        raise RuntimeError(f"LIBERO config is missing: {expected_config_path}")

    get_libero_path = libero_package.get_libero_path
    for name, expected in expected_paths.items():
        actual = Path(get_libero_path(name)).resolve()
        if actual != expected.resolve():
            raise RuntimeError(f"LIBERO config {name} is {actual}, expected {expected}")
    for name in ("bddl_files", "init_states", "assets"):
        if not expected_paths[name].is_dir():
            raise RuntimeError(f"LIBERO runtime directory is missing: {expected_paths[name]}")


@dataclass(frozen=True)
class LiberoRuntimeConfig:
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
            raise ValueError("LIBERO task and trial counts must be positive")
        if self.chunk_size < 1 or self.settle_steps < 0:
            raise ValueError("LIBERO chunk size must be positive and settle steps cannot be negative")
        if self.cpu_threads < 1:
            raise ValueError("LIBERO cpu_threads must be positive")
        if self.render_backend not in {"egl", "osmesa"}:
            raise ValueError(f"unsupported LIBERO render backend: {self.render_backend}")


class _TaskSamplingConfig(Protocol):
    task_ids: tuple[int, ...] | None
    sample_size: int | None
    sample_seed: int
    num_tasks: int
    libero_home: Path
    suite: str


def stratified_task_ids(config: _TaskSamplingConfig) -> list[int]:
    if config.task_ids is not None:
        selected = list(config.task_ids)
        if len(selected) != len(set(selected)):
            raise ValueError("LIBERO task_ids must be unique")
        if any(task_id < 0 or task_id >= config.num_tasks for task_id in selected):
            raise ValueError("LIBERO task_ids contain an out-of-range task")
        return selected
    if config.sample_size is None or config.sample_size >= config.num_tasks:
        return list(range(config.num_tasks))
    if config.sample_size < 1:
        raise ValueError("LIBERO sample_size must be positive")
    path = config.libero_home / "libero" / "libero" / "benchmark" / "task_classification.json"
    rows = json.loads(path.read_text(encoding="utf-8"))[config.suite]
    if len(rows) != config.num_tasks:
        raise ValueError(f"classification contains {len(rows)} tasks, expected {config.num_tasks}")
    groups: dict[tuple[str, int], list[int]] = defaultdict(list)
    for row in rows:
        key = (row["category"], int(row.get("difficulty_level", 0)))
        groups[key].append(int(row["id"]) - 1)
    exact = {key: config.sample_size * len(ids) / len(rows) for key, ids in groups.items()}
    quotas = {key: math.floor(value) for key, value in exact.items()}
    remaining = config.sample_size - sum(quotas.values())
    ranked = sorted(groups, key=lambda key: (exact[key] - quotas[key], key), reverse=True)
    for key in ranked[:remaining]:
        quotas[key] += 1
    rng = random.Random(config.sample_seed)
    selected: list[int] = []
    for key in sorted(groups):
        selected.extend(rng.sample(groups[key], quotas[key]))
    return sorted(selected)


class LiberoBenchmarkAdapter:
    """Run the official LIBERO protocol against an SDK Policy process."""

    def __init__(self, config: LiberoRuntimeConfig) -> None:
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
            "PYTHONPATH": (f"{self.config.libero_home}:{SDK_IMPORT_ROOT}:{os.environ.get('PYTHONPATH', '')}"),
            "LIBERO_HOME": str(self.config.libero_home),
            "LIBERO_CONFIG_PATH": str(runtime_config_path.parent),
            "MUJOCO_GL": self.config.render_backend,
            "PYOPENGL_PLATFORM": self.config.render_backend,
            "OMP_NUM_THREADS": threads,
            "MKL_NUM_THREADS": threads,
            "OPENBLAS_NUM_THREADS": threads,
            "NUMEXPR_NUM_THREADS": threads,
            # Shared AutoDL FS inode quota can be exhausted; never write pyc.
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
            "benchmark": "libero",
            "suite": self.config.suite,
            "protocol": "roboscope-policy-v1",
            "bound_hint": "gpu_infer",
            "task_ids": self.task_ids,
            "episodes_per_task": self.config.num_trials,
            "seed": self.config.seed,
            "render_backend": self.config.render_backend,
        }


__all__ = [
    "LiberoBenchmarkAdapter",
    "LiberoRuntimeConfig",
    "SDK_IMPORT_ROOT",
    "stratified_task_ids",
    "validate_libero_runtime",
    "write_libero_config",
]

#!/usr/bin/env python3
"""Populate every RoboTwin task cache to the formal evaluation episode count."""

from __future__ import annotations

import argparse
import ast
import importlib.util
import os
import pickle
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

KIT_ROOT = Path(__file__).resolve().parents[2]
SDK_SOURCE = KIT_ROOT / "packages/roboscope-kit/src"
ROBOTWIN_SOURCE = SDK_SOURCE / "roboscope_kit/runtime/benchmarks/robotwin"


def load_robotwin_tasks() -> tuple[str, ...]:
    module = ast.parse((ROBOTWIN_SOURCE / "adapter.py").read_text(encoding="utf-8"))
    for node in module.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "ROBOTWIN_ALL_TASKS"
            for target in node.targets
        ):
            return tuple(ast.literal_eval(node.value))
    raise RuntimeError("ROBOTWIN_ALL_TASKS is missing from the RoboTwin adapter")


def load_seed_cache_validator():
    module_path = ROBOTWIN_SOURCE / "seed_cache.py"
    spec = importlib.util.spec_from_file_location("roboscope_robotwin_seed_cache", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load RoboTwin seed cache validator: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.validate_seed_cache


ROBOTWIN_ALL_TASKS = load_robotwin_tasks()
validate_seed_cache = load_seed_cache_validator()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robotwin-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--task-config", required=True)
    parser.add_argument("--target-valid", type=int, default=100)
    parser.add_argument("--gpu-count", type=int, required=True)
    parser.add_argument("--max-workers", type=int, default=32)
    parser.add_argument("--seed-batch-size", type=int, default=256)
    parser.add_argument("--tasks", nargs="*")
    return parser.parse_args()


def load_existing(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    with path.open("rb") as handle:
        entries = pickle.load(handle)
    if not isinstance(entries, list):
        raise RuntimeError(f"RoboTwin seed cache must contain a list: {path}")
    return entries


def next_seed(entries: list[dict]) -> int:
    seeds = [entry.get("seed") for entry in entries if isinstance(entry, dict)]
    valid_seeds = [seed for seed in seeds if type(seed) is int]
    return max(valid_seeds, default=-1) + 1


def scanner_command(
    args: argparse.Namespace,
    task: str,
    seed_start: int,
    worker_index: int,
) -> list[str]:
    return [
        sys.executable,
        str(Path(__file__).with_name("roboscope_robotwin_seed_scan.py")),
        "--robotwin-dir",
        str(args.robotwin_dir),
        "--task",
        task,
        "--task-config",
        args.task_config,
        "--out-dir",
        str(args.cache_dir),
        "--seed-start",
        str(seed_start),
        "--seed-count",
        str(args.seed_batch_size),
        "--max-valid",
        str(args.target_valid),
        "--instruction-seed-limit",
        str(args.target_valid),
        "--cpu-threads",
        "1",
        "--init-stagger-s",
        str((worker_index // args.gpu_count) * 5),
    ]


def prepare_task(args: argparse.Namespace, task: str, worker_index: int) -> str:
    cache_path = args.cache_dir / args.task_config / f"{task}.pkl"
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(worker_index % args.gpu_count)}
    entries = load_existing(cache_path)
    while len(entries) < args.target_valid:
        subprocess.run(
            scanner_command(args, task, next_seed(entries), worker_index),
            env=env,
            check=True,
        )
        updated = load_existing(cache_path)
        if len(updated) <= len(entries):
            raise RuntimeError(f"RoboTwin Scanner made no progress for {args.task_config}/{task}")
        entries = updated
    validate_seed_cache(cache_path, args.target_valid)
    return f"{args.task_config}/{task}: {len(entries)}"


def validate_args(args: argparse.Namespace) -> tuple[str, ...]:
    if args.target_valid < 1 or args.gpu_count < 1 or args.max_workers < 1 or args.seed_batch_size < 1:
        raise ValueError("target-valid, gpu-count, max-workers, and seed-batch-size must be positive")
    tasks = tuple(args.tasks or ROBOTWIN_ALL_TASKS)
    unknown = sorted(set(tasks) - set(ROBOTWIN_ALL_TASKS))
    if not tasks or unknown or len(tasks) != len(set(tasks)):
        raise ValueError(f"invalid RoboTwin tasks: {', '.join(unknown)}")
    return tasks


def main() -> int:
    args = parse_args()
    tasks = validate_args(args)
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=min(args.max_workers, len(tasks))) as executor:
        futures = {
            executor.submit(prepare_task, args, task, index): task
            for index, task in enumerate(tasks)
        }
        for future in as_completed(futures):
            print(future.result(), flush=True)
    print(
        f"RoboTwin seed cache ready: suite={args.task_config} tasks={len(tasks)} "
        f"episodes_per_task={args.target_valid}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

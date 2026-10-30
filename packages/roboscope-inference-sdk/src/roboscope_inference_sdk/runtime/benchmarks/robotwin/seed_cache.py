# Copyright 2026 RoboScope Team
# SPDX-License-Identifier: Apache-2.0

"""Validate RoboTwin Scanner cache files before benchmark execution."""

from __future__ import annotations

import argparse
import json
import pickle
from collections.abc import Sequence
from pathlib import Path


def validate_seed_cache(path: Path, episodes: int) -> list[dict]:
    if not path.is_file():
        raise RuntimeError(f"RoboTwin seed cache is missing: {path}")
    with path.open("rb") as handle:
        entries = pickle.load(handle)
    if not isinstance(entries, list):
        raise RuntimeError(f"RoboTwin seed cache must contain a list: {path}")
    if len(entries) < episodes:
        raise RuntimeError(
            f"RoboTwin seed cache has {len(entries)} entries but {episodes} are required: {path}"
        )
    for index, entry in enumerate(entries[:episodes]):
        if not isinstance(entry, dict) or type(entry.get("seed")) is not int:
            raise RuntimeError(f"RoboTwin seed cache entry {index} has no integer seed: {path}")
        episode_info = entry.get("episode_info")
        if not isinstance(episode_info, dict) or not isinstance(episode_info.get("info"), dict):
            raise RuntimeError(f"RoboTwin seed cache entry {index} has invalid episode_info: {path}")
    return entries


def validate_seed_cache_dir(
    cache_dir: Path,
    suite: str,
    tasks: Sequence[str],
    episodes: int,
) -> dict[str, int | str]:
    if episodes < 1:
        raise ValueError("RoboTwin seed cache episode requirement must be positive")
    for task in tasks:
        validate_seed_cache(cache_dir / suite / f"{task}.pkl", episodes)
    return {
        "suite": suite,
        "task_count": len(tasks),
        "episodes_per_task": episodes,
        "total_episodes": len(tasks) * episodes,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--suite", required=True)
    parser.add_argument("--episodes", type=int, required=True)
    parser.add_argument("--tasks", nargs="*")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.tasks:
        tasks = tuple(args.tasks)
    else:
        from .adapter import ROBOTWIN_ALL_TASKS

        tasks = ROBOTWIN_ALL_TASKS
    summary = validate_seed_cache_dir(args.cache_dir, args.suite, tasks, args.episodes)
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["validate_seed_cache", "validate_seed_cache_dir"]

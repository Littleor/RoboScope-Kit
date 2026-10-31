#!/usr/bin/env python3
"""Compare cached RoboTwin episode_info with the native expert path for one seed."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import pickle
import sys
from pathlib import Path

from roboscope_robotwin_seed_scan import build_task_args, configure_cpu_threads


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robotwin-dir", type=Path, required=True)
    parser.add_argument("--cache-path", type=Path, required=True)
    parser.add_argument("--suite", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--index", type=int, default=0)
    return parser.parse_args()


def fingerprint(value: object) -> str:
    return hashlib.sha256(pickle.dumps(value, protocol=4)).hexdigest()


def verify_episode_info_pair(cached: object, native: object) -> str:
    cached_digest = fingerprint(cached)
    native_digest = fingerprint(native)
    if cached_digest != native_digest:
        raise RuntimeError(
            f"cached/native episode_info differ: cached={cached_digest} native={native_digest}"
        )
    return cached_digest


def main() -> int:
    args = parse_args()
    configure_cpu_threads(1)
    with args.cache_path.open("rb") as handle:
        entries = pickle.load(handle)
    entry = entries[args.index]
    seed = int(entry["seed"])
    cached = entry["episode_info"]

    robotwin_dir = args.robotwin_dir.resolve()
    task_args = build_task_args(robotwin_dir, args.task, args.suite)
    module = importlib.import_module(f"envs.{args.task}")
    env = getattr(module, args.task)()
    try:
        env.setup_demo(now_ep_num=0, seed=seed, is_test=True, **task_args)
        native = env.play_once()
        if not bool(env.plan_success and env.check_success()):
            raise RuntimeError("selected cached seed is not valid on the native expert path")
    finally:
        env.close_env()

    cached_digest = verify_episode_info_pair(cached, native)
    print(
        f"paired seed verified: suite={args.suite} task={args.task} "
        f"seed={seed} episode_info_sha256={cached_digest}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

# Copyright 2026 RoboScope Team
# SPDX-License-Identifier: Apache-2.0

"""Regenerate the seeded robomimic initial states and check them against the frozen digests.

Runs in the robomimic runtime environment (``build-runtime.sh`` and ``--self-test`` call it)::

    python -m roboscope_kit.runtime.benchmarks.robomimic.initial_states [--task lift ...] [--print]

For every task and seed it samples the initial state the worker's seeded reset rolls out
from and digests the flattened MuJoCo state; tasks run in parallel processes. A per-task
digest that differs from ``tasks.TASKS`` exits with status 1, so an environment that would
roll out different initial states is rejected before any evaluation.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

from .tasks import SEEDS, TASK_NAMES, TASKS_BY_NAME, initial_state_sha256, initial_states_sha256


def task_digest(task_name: str) -> str:
    from .worker import create_env, initialize_observation_modalities, sample_initial_state

    initialize_observation_modalities()
    task = TASKS_BY_NAME[task_name]
    env = create_env(task)
    try:
        return initial_states_sha256(
            (seed, initial_state_sha256(sample_initial_state(env, seed)["states"])) for seed in SEEDS
        )
    finally:
        close = getattr(env.env, "close", None)
        if close is not None:
            close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--task", action="append", choices=TASK_NAMES)
    parser.add_argument("--print", action="store_true", help="print the digests without comparing")
    args = parser.parse_args()
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    report = {}
    mismatched = []
    names = args.task or list(TASK_NAMES)
    with ProcessPoolExecutor(max_workers=len(names)) as pool:
        digests = dict(zip(names, pool.map(task_digest, names), strict=True))
    for name in names:
        actual = digests[name]
        expected = TASKS_BY_NAME[name].initial_states_sha256
        report[name] = {"seeds": len(SEEDS), "actual": actual, "expected": expected}
        if actual != expected:
            mismatched.append(name)
    print(json.dumps(report, indent=2, sort_keys=True))
    if mismatched and not args.print:
        print(f"robomimic initial-state digests differ for: {', '.join(mismatched)}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

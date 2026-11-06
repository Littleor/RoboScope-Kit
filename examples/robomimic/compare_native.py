# Copyright 2026 RoboScope Team
# SPDX-License-Identifier: Apache-2.0

"""Compare a RoboScope Kit robomimic run with the pinned upstream ``run_trained_agent.py``.

    python compare_native.py --runtime /sim/robomimic --checkpoint-dir /models/robomimic-zoo \\
        --kit-run /runs/robomimic-full --output /runs/robomimic-native [--jobs 8] [--task lift ...]

For every task and seed in the Kit run it executes, unmodified,

    <runtime>/env/bin/python -m robomimic.scripts.run_trained_agent \\
        --agent <checkpoint> --n_rollouts 1 --seed <seed>

which rolls out exactly one episode from the initial state that ``np.random.seed(seed)``
samples, with the checkpoint's own rollout horizon, after ``torch.manual_seed(seed)``.
The Kit episode with the same seed starts from the same state (the worker records its
digest) and the example Policy seeds the episode's torch RNG the same way, so each pair
is compared episode by episode: success and executed steps (upstream ``Horizon``).

``--batch-rollouts N`` additionally runs the upstream default protocol once per task
(``--n_rollouts N --seed 0``: one RNG stream across N consecutive episodes). Those
episodes are not pairable with the Kit's per-seed episodes; only their success rate is
reported, as a second, statistical reference.

Writes ``<output>/native-episodes.jsonl``, ``<output>/comparison.json`` and one upstream
log per episode under ``<output>/logs/``.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from model_zoo import CHECKPOINTS  # noqa: E402

CHECKPOINTS_BY_TASK = {checkpoint.task: checkpoint for checkpoint in CHECKPOINTS}
STATS_PATTERN = re.compile(r"Average Rollout Stats\s*(\{.*?\})", re.S)


def kit_records(run_dir: Path) -> dict[tuple[str, int], dict]:
    records: dict[tuple[str, int], dict] = {}
    for path in sorted((run_dir / "queue" / "completed").glob("item-*.json")):
        for record in json.loads(path.read_text(encoding="utf-8"))["records"]:
            key = (record["metadata"]["task"], int(record["metadata"]["seed"]))
            if key in records:
                raise SystemExit(f"Kit run {run_dir} has a duplicate record for {key}")
            records[key] = record
    if not records:
        raise SystemExit(f"no completed Kit records under {run_dir}/queue/completed")
    return records


def run_native(
    runtime: Path, checkpoint_dir: Path, log_dir: Path, task: str, seed: int, rollouts: int
) -> dict:
    checkpoint = checkpoint_dir / CHECKPOINTS_BY_TASK[task].filename
    command = [
        str(runtime / "env" / "bin" / "python"), "-m", "robomimic.scripts.run_trained_agent",
        "--agent", str(checkpoint), "--n_rollouts", str(rollouts), "--seed", str(seed),
    ]
    log_path = log_dir / f"{task}-seed-{seed:04d}-n{rollouts}.log"
    started = time.perf_counter()
    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        check=False,
    )
    elapsed_s = time.perf_counter() - started
    log_path.write_text(completed.stdout + "\n--- stderr ---\n" + completed.stderr, encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError(f"upstream run failed for {task} seed {seed}; see {log_path}")
    match = STATS_PATTERN.search(completed.stdout)
    if match is None:
        raise RuntimeError(f"upstream run for {task} seed {seed} printed no rollout stats; see {log_path}")
    stats = json.loads(match.group(1))
    return {
        "task": task,
        "seed": seed,
        "n_rollouts": rollouts,
        "success_rate": float(stats["Success_Rate"]),
        "num_success": int(stats["Num_Success"]),
        "horizon": float(stats["Horizon"]),
        "return": float(stats["Return"]),
        "process_elapsed_s": elapsed_s,
        "command": command,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--runtime", type=Path, required=True, help="build-runtime.sh install dir")
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--kit-run", type=Path, required=True, help="Kit --output-dir to compare")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--task", action="append", choices=sorted(CHECKPOINTS_BY_TASK))
    parser.add_argument("--batch-rollouts", type=int, default=0)
    args = parser.parse_args()

    kit = kit_records(args.kit_run)
    tasks = args.task or sorted({task for task, _ in kit}, key=list(CHECKPOINTS_BY_TASK).index)
    pairs = sorted(key for key in kit if key[0] in tasks)
    log_dir = args.output / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        # Submit the long sequential batch runs first so they overlap the per-seed runs.
        batch_futures = [
            pool.submit(run_native, args.runtime, args.checkpoint_dir, log_dir, task, 0, args.batch_rollouts)
            for task in (tasks if args.batch_rollouts > 0 else [])
        ]
        native_futures = [
            pool.submit(run_native, args.runtime, args.checkpoint_dir, log_dir, task, seed, 1)
            for task, seed in pairs
        ]
        native = [future.result() for future in native_futures]
        batch = [future.result() for future in batch_futures]
    native_elapsed_s = time.perf_counter() - started

    with (args.output / "native-episodes.jsonl").open("w", encoding="utf-8") as handle:
        for row in native + batch:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    per_task = {}
    mismatches = []
    for task in tasks:
        rows = [(kit[(t, s)], n) for (t, s), n in zip(pairs, native, strict=True) if t == task]
        kit_success = sum(bool(k["success"]) for k, _ in rows)
        native_success = sum(n["num_success"] for _, n in rows)
        same_success = sum(bool(k["success"]) == bool(n["num_success"]) for k, n in rows)
        same_steps = sum(int(k["steps"]) == int(n["horizon"]) for k, n in rows)
        for k, n in rows:
            if bool(k["success"]) != bool(n["num_success"]) or int(k["steps"]) != int(n["horizon"]):
                mismatches.append(
                    {
                        "task": task,
                        "seed": n["seed"],
                        "kit": {"success": bool(k["success"]), "steps": int(k["steps"])},
                        "native": {"success": bool(n["num_success"]), "steps": int(n["horizon"])},
                    }
                )
        per_task[task] = {
            "episodes": len(rows),
            "kit_successes": kit_success,
            "native_successes": native_success,
            "kit_success_rate": kit_success / len(rows),
            "native_success_rate": native_success / len(rows),
            "kit_total_steps": sum(int(k["steps"]) for k, _ in rows),
            "native_total_steps": int(sum(n["horizon"] for _, n in rows)),
            "episodes_with_same_success": same_success,
            "episodes_with_same_steps": same_steps,
        }
    for row in batch:
        per_task[row["task"]]["native_batch"] = {
            "n_rollouts": row["n_rollouts"],
            "seed": row["seed"],
            "success_rate": row["success_rate"],
            "mean_horizon": row["horizon"],
        }
    comparison = {
        "kit_run": str(args.kit_run),
        "runtime": str(args.runtime),
        "checkpoint_dir": str(args.checkpoint_dir),
        "pairing": "per seed: run_trained_agent.py --n_rollouts 1 --seed <seed> "
        "against the Kit episode <task>-seed-<seed>",
        "episodes": len(pairs),
        "identical_episodes": len(pairs) - len(mismatches),
        "mismatches": mismatches,
        "tasks": per_task,
        "native_wall_s": native_elapsed_s,
        "native_jobs": args.jobs,
    }
    (args.output / "comparison.json").write_text(json.dumps(comparison, indent=2) + "\n", encoding="utf-8")
    summary = {key: comparison[key] for key in ("episodes", "identical_episodes")}
    print(json.dumps(summary | {"tasks": per_task}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

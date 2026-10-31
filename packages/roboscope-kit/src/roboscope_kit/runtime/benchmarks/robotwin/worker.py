# Copyright 2026 RoboScope Team
# SPDX-License-Identifier: Apache-2.0

"""Persistent RoboTwin evaluator worker using an SDK Policy replica."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from collections import deque
from pathlib import Path

from roboscope_kit.runtime.benchmark import (
    EvaluationRecord,
    WorkerLoop,
    WorkItem,
    WorkQueue,
    WorkResult,
)

from .seed_cache import validate_seed_cache

RECORD_PREFIX = "ROBOSCOPE_ROBOTWIN_RECORD "
DONE_PREFIX = "ROBOSCOPE_ROBOTWIN_TASK_DONE "
FAILURE_LOG_TAIL_BYTES = 32 * 1024
MEMORY_FAILURE_TOKENS = (
    "out of memory",
    "cannot create buffer",
    "failed to allocate device memory",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker-id", type=int, required=True)
    parser.add_argument("--gpu-id", required=True)
    parser.add_argument("--policy-socket", type=Path, required=True)
    parser.add_argument("--robotwin-dir", type=Path, required=True)
    parser.add_argument("--robotwin-python", type=Path, required=True)
    parser.add_argument("--checkpoint-path", type=Path, required=True)
    parser.add_argument("--deploy-policy", type=Path, required=True)
    parser.add_argument("--launcher", type=Path, required=True)
    parser.add_argument("--queue-dir", type=Path, required=True)
    parser.add_argument("--worker-state-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--disable-video", action="store_true")
    parser.add_argument("--seed-cache-dir", type=Path)
    parser.add_argument(
        "--eval-session",
        action="store_true",
        help="keep one launcher process alive across tasks so the shared "
        "CuroboPlanner patch pays planner construction once per worker",
    )
    parser.add_argument("--poll-interval", type=float, default=0.05)
    return parser.parse_args()


def _item_window(item: WorkItem) -> tuple[int, int, int]:
    episode_start = int(item.payload["episode_start"])
    episodes = int(item.payload["episodes"])
    total_episodes = int(item.payload["total_episodes"])
    if episode_start < 0 or episodes < 1 or episode_start + episodes > total_episodes:
        raise ValueError(
            "RoboTwin work item episode window must be positive and within total_episodes"
        )
    return episode_start, episodes, total_episodes


def _task_spec(args: argparse.Namespace, item: WorkItem, deploy_policy: Path) -> tuple[dict, list[str]]:
    """Build the per-task environment overrides and launcher argv tail."""

    task = str(item.payload["task"])
    suite = str(item.payload["suite"])
    seed = int(item.payload["seed"])
    episode_start, episodes, total_episodes = _item_window(item)
    videos_dir = args.output_dir / "video" / f"worker-{args.worker_id:03d}"
    env = {
        "ROBOSCOPE_ROBOTWIN_TASK": task,
        "ROBOTWIN_EPISODE_START": str(episode_start),
        "ROBOTWIN_TEST_NUM": str(episodes),
        "ROBOTWIN_INSTRUCTION_SEED_LIMIT": str(total_episodes),
        "ROBOTWIN_VIDEO_ROOT": str(videos_dir),
        "ROBOTWIN_DISABLE_VIDEO": "1" if args.disable_video else "0",
    }
    # Scanner cache is used only when provided (always for a full run; a
    # quick-test may omit it and fall back to the official expert play_once path).
    if args.seed_cache_dir is not None:
        seed_cache_path = args.seed_cache_dir / suite / f"{task}.pkl"
        validate_seed_cache(seed_cache_path, total_episodes)
        env["ROBOTWIN_SEED_CACHE_PATH"] = str(seed_cache_path)
    argv_tail = [
        "--config",
        str(deploy_policy),
        "--policy_ckpt_path",
        str(args.checkpoint_path),
        "--overrides",
        "--task_name",
        task,
        "--task_config",
        suite,
        "--ckpt_setting",
        "roboscope_sdk",
        "--seed",
        str(seed),
        "--policy_name",
        "model2robotwin_interface",
    ]
    return env, argv_tail


class _EvalSession:
    """Persistent launcher process evaluating one task per stdin spec line.

    The launcher applies each spec's env to ``os.environ`` and execs the
    instrumented evaluator, so per-task variation (task name, episodes, seed
    cache path) flows through the spec while static configuration stays on
    the process environment. Keeping the process alive across tasks lets the
    shared CuroboPlanner patch pay planner construction once per worker
    instead of once per task. Any crashed task ends the session; the next
    task transparently starts a fresh process.
    """

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.process: subprocess.Popen | None = None

    def _spawn(self) -> subprocess.Popen:
        env = {
            **os.environ,
            # The RoboTwin runtime may live on read-only or inode-limited storage,
            # where writing __pycache__ fails and makes imports (e.g.
            # planner.CuroboPlanner) intermittently crash. Evaluate without
            # bytecode caching so imports never need to write pyc files.
            "PYTHONDONTWRITEBYTECODE": "1",
            "CUDA_VISIBLE_DEVICES": str(self.args.gpu_id),
            "ROBOSCOPE_INFERENCE_SOCKET": str(self.args.policy_socket),
            "ROBOTWIN_EVAL_SESSION": "1",
            "ROBOTWIN_DISABLE_VIDEO": "1" if self.args.disable_video else "0",
            "ROBOTWIN_SEED_CACHE_ENABLED": (
                "1" if self.args.seed_cache_dir is not None else "0"
            ),
        }
        command = (
            str(self.args.robotwin_python),
            str(self.args.launcher),
            "--script",
            str(self.args.robotwin_dir / "script" / "eval_policy.py"),
        )
        self.process = subprocess.Popen(
            command,
            cwd=self.args.robotwin_dir,
            env=env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        return self.process

    def stop(self) -> None:
        if self.process is None:
            return
        process, self.process = self.process, None
        try:
            if process.stdin is not None:
                process.stdin.close()
        except OSError:
            pass
        stop_child_process(process)

    def stop_and_returncode(self) -> int:
        process = self.process
        self.process = None
        if process is None:
            return 1
        try:
            if process.stdin is not None:
                process.stdin.close()
        except OSError:
            pass
        stop_child_process(process)
        return process.returncode if process.returncode is not None else 1


_SESSION: _EvalSession | None = None


def _get_session(args: argparse.Namespace) -> _EvalSession:
    global _SESSION
    if _SESSION is None:
        _SESSION = _EvalSession(args)
    if _SESSION.process is None or _SESSION.process.poll() is not None:
        _SESSION._spawn()
    return _SESSION


def run_task(args: argparse.Namespace, item: WorkItem) -> WorkResult:
    if args.eval_session:
        return _run_task_in_session(args, item)
    return _run_task_single_process(args, item)


def _run_task_in_session(args: argparse.Namespace, item: WorkItem) -> WorkResult:
    task = str(item.payload["task"])
    suite = str(item.payload["suite"])
    episode_start, episodes, _total_episodes = _item_window(item)
    episode_stop = episode_start + episodes
    logs_dir = args.output_dir / "native-logs"
    videos_dir = args.output_dir / "video" / f"worker-{args.worker_id:03d}"
    logs_dir.mkdir(parents=True, exist_ok=True)
    videos_dir.mkdir(parents=True, exist_ok=True)
    item_name = f"{suite}-{task}-episodes-{episode_start:04d}-{episode_stop:04d}"
    log_path = logs_dir / f"worker-{args.worker_id:03d}-{item_name}.log"
    deploy_policy = logs_dir / f"worker-{args.worker_id:03d}-{item_name}.yml"
    write_deploy_policy(args.deploy_policy, deploy_policy)
    task_env, argv_tail = _task_spec(args, item, deploy_policy)

    session = _get_session(args)
    if session.process is None or session.process.stdin is None:
        raise RuntimeError("RoboTwin evaluator session failed to start")
    done_ok: bool | None = None
    records: list[EvaluationRecord] = []
    memory_failures: deque[str] = deque(maxlen=16)
    queue = WorkQueue(args.queue_dir)
    started = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as log:
        try:
            session.process.stdin.write(json.dumps({"env": task_env, "args": argv_tail}) + "\n")
            session.process.stdin.flush()
        except (BrokenPipeError, OSError):
            # The evaluator died between tasks; classify this task through the
            # same failure path as a mid-task crash below.
            done_ok = None
        else:
            if session.process.stdout is None:
                raise RuntimeError("RoboTwin evaluator session stdout is unavailable")
            for line in session.process.stdout:
                log.write(line)
                log.flush()
                if any(token in line.lower() for token in MEMORY_FAILURE_TOKENS):
                    memory_failures.append(line.strip())
                if line.startswith(DONE_PREFIX):
                    done_ok = bool(json.loads(line[len(DONE_PREFIX) :]).get("ok"))
                    break
                if not line.startswith(RECORD_PREFIX):
                    continue
                payload = json.loads(line[len(RECORD_PREFIX) :])
                index = int(payload["episode_index"])
                records.append(
                    EvaluationRecord(
                        record_id=f"{suite}-{task}-episode-{index:04d}",
                        success=bool(payload["success"]),
                        steps=int(payload["steps"]),
                        elapsed_s=float(payload["elapsed_s"]),
                        metadata={
                            "task": task,
                            "suite": suite,
                            "seed": int(payload["seed"]),
                            "episode_index": index,
                        },
                    )
                )
                queue.publish_partial(
                    item,
                    WorkResult(
                        item_id=item.item_id,
                        group_id=item.group_id,
                        worker_id=args.worker_id,
                        records=tuple(records),
                        metadata={
                            "task": task,
                            "suite": suite,
                            "episode_start": episode_start,
                            "episode_stop": episode_stop,
                        },
                    ),
                )
    if done_ok is None or not done_ok:
        # EOF before DONE (evaluator died mid-task) or a reported task crash:
        # end the session so the next task starts in a fresh process.
        returncode: int
        if done_ok is None:
            returncode = session.stop_and_returncode()
        else:
            session.stop()
            returncode = 1
        detail = read_log_tail(log_path)
        if memory_failures or returncode < 0:
            memory_detail = ""
            if memory_failures:
                memory_detail = "\nNative evaluator memory failures:\n" + "\n".join(memory_failures)
            raise RuntimeError(
                f"RoboTwin task {task} exited with {returncode}; see {log_path}"
                f"{memory_detail}"
                f"\nNative evaluator log tail:\n{detail}"
            )
        return _failed_task_result(
            item,
            args.worker_id,
            task=task,
            suite=suite,
            episode_start=episode_start,
            episodes=episodes,
            records=records,
            returncode=returncode,
            log_path=log_path,
            detail=detail,
        )
    if len(records) != episodes:
        raise RuntimeError(f"RoboTwin task {task} emitted {len(records)}/{episodes} records")
    _validate_episode_indices(records, episode_start, episodes, task)
    return WorkResult(
        item_id=item.item_id,
        group_id=item.group_id,
        worker_id=args.worker_id,
        records=tuple(records),
        metadata={
            "task": task,
            "suite": suite,
            "episode_start": episode_start,
            "episode_stop": episode_stop,
            "elapsed_s": time.perf_counter() - started,
        },
    )


def _run_task_single_process(args: argparse.Namespace, item: WorkItem) -> WorkResult:
    task = str(item.payload["task"])
    suite = str(item.payload["suite"])
    episode_start, episodes, _total_episodes = _item_window(item)
    episode_stop = episode_start + episodes
    eval_script = args.robotwin_dir / "script" / "eval_policy.py"
    logs_dir = args.output_dir / "native-logs"
    videos_dir = args.output_dir / "video" / f"worker-{args.worker_id:03d}"
    logs_dir.mkdir(parents=True, exist_ok=True)
    videos_dir.mkdir(parents=True, exist_ok=True)
    item_name = f"{suite}-{task}-episodes-{episode_start:04d}-{episode_stop:04d}"
    log_path = logs_dir / f"worker-{args.worker_id:03d}-{item_name}.log"
    deploy_policy = logs_dir / f"worker-{args.worker_id:03d}-{item_name}.yml"
    write_deploy_policy(args.deploy_policy, deploy_policy)
    task_env, argv_tail = _task_spec(args, item, deploy_policy)
    env = {
        **os.environ,
        # The RoboTwin runtime may live on read-only or inode-limited storage,
        # where writing __pycache__ fails and makes imports (e.g.
        # planner.CuroboPlanner) intermittently crash. Evaluate without
        # bytecode caching so imports never need to write pyc files.
        "PYTHONDONTWRITEBYTECODE": "1",
        "CUDA_VISIBLE_DEVICES": str(args.gpu_id),
        "ROBOSCOPE_INFERENCE_SOCKET": str(args.policy_socket),
        "ROBOTWIN_SEED_CACHE_ENABLED": "1" if args.seed_cache_dir is not None else "0",
        **task_env,
    }
    command = (
        str(args.robotwin_python),
        str(args.launcher),
        "--script",
        str(eval_script),
        *argv_tail,
    )
    records: list[EvaluationRecord] = []
    memory_failures: deque[str] = deque(maxlen=16)
    queue = WorkQueue(args.queue_dir)
    started = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            cwd=args.robotwin_dir,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        try:
            if process.stdout is None:
                raise RuntimeError("RoboTwin worker stdout is unavailable")
            for line in process.stdout:
                log.write(line)
                log.flush()
                if any(token in line.lower() for token in MEMORY_FAILURE_TOKENS):
                    memory_failures.append(line.strip())
                if not line.startswith(RECORD_PREFIX):
                    continue
                payload = json.loads(line[len(RECORD_PREFIX) :])
                index = int(payload["episode_index"])
                records.append(
                    EvaluationRecord(
                        record_id=f"{suite}-{task}-episode-{index:04d}",
                        success=bool(payload["success"]),
                        steps=int(payload["steps"]),
                        elapsed_s=float(payload["elapsed_s"]),
                        metadata={
                            "task": task,
                            "suite": suite,
                            "seed": int(payload["seed"]),
                            "episode_index": index,
                        },
                    )
                )
                queue.publish_partial(
                    item,
                    WorkResult(
                        item_id=item.item_id,
                        group_id=item.group_id,
                        worker_id=args.worker_id,
                        records=tuple(records),
                        metadata={
                            "task": task,
                            "suite": suite,
                            "episode_start": episode_start,
                            "episode_stop": episode_stop,
                        },
                    ),
                )
            returncode = process.wait()
        finally:
            stop_child_process(process)
    if returncode != 0:
        detail = read_log_tail(log_path)
        if memory_failures or returncode < 0:
            memory_detail = ""
            if memory_failures:
                memory_detail = "\nNative evaluator memory failures:\n" + "\n".join(memory_failures)
            raise RuntimeError(
                f"RoboTwin task {task} exited with {returncode}; see {log_path}"
                f"{memory_detail}"
                f"\nNative evaluator log tail:\n{detail}"
            )
        return _failed_task_result(
            item,
            args.worker_id,
            task=task,
            suite=suite,
            episode_start=episode_start,
            episodes=episodes,
            records=records,
            returncode=returncode,
            log_path=log_path,
            detail=detail,
        )
    if len(records) != episodes:
        raise RuntimeError(f"RoboTwin task {task} emitted {len(records)}/{episodes} records")
    _validate_episode_indices(records, episode_start, episodes, task)
    return WorkResult(
        item_id=item.item_id,
        group_id=item.group_id,
        worker_id=args.worker_id,
        records=tuple(records),
        metadata={
            "task": task,
            "suite": suite,
            "episode_start": episode_start,
            "episode_stop": episode_stop,
            "elapsed_s": time.perf_counter() - started,
        },
    )


def _validate_episode_indices(
    records: list[EvaluationRecord],
    episode_start: int,
    episodes: int,
    task: str,
) -> set[int]:
    recorded = {record.metadata["episode_index"] for record in records}
    expected = range(episode_start, episode_start + episodes)
    if len(recorded) != len(records) or not recorded.issubset(expected):
        raise RuntimeError(
            f"RoboTwin task {task} emitted records with duplicate or out-of-range episode indices"
        )
    return recorded


def _failed_task_result(
    item: WorkItem,
    worker_id: int,
    *,
    task: str,
    suite: str,
    episode_start: int,
    episodes: int,
    records: list[EvaluationRecord],
    returncode: int,
    log_path: Path,
    detail: str,
) -> WorkResult:
    """Record a native evaluator failure as a terminal failed item so the worker
    continues instead of aborting the whole evaluation pass.

    Records already emitted before the crash are kept; the missing episodes are
    filled with failed records so the item still satisfies its expected result
    count and the missing episodes count as failures in the official aggregate.
    """
    print(
        f"RoboTwin task {task} exited with {returncode}; see {log_path}"
        f"\nNative evaluator log tail:\n{detail}",
        file=sys.stderr,
    )
    recorded = _validate_episode_indices(records, episode_start, episodes, task)
    terminal_records = list(records)
    expected = set(range(episode_start, episode_start + episodes))
    for episode_index in sorted(expected - recorded):
        terminal_records.append(
            EvaluationRecord(
                record_id=f"{suite}-{task}-episode-{episode_index:04d}",
                success=False,
                steps=0,
                elapsed_s=0.0,
                metadata={
                    "task": task,
                    "suite": suite,
                    "episode_index": episode_index,
                },
            )
        )
    terminal_records.sort(key=lambda record: record.metadata["episode_index"])
    return WorkResult(
        item_id=item.item_id,
        group_id=item.group_id,
        worker_id=worker_id,
        records=tuple(terminal_records),
        metadata={
            "task": task,
            "suite": suite,
            "episode_start": episode_start,
            "episode_stop": episode_start + episodes,
            "failed": True,
            "exit_code": returncode,
        },
    )


def write_deploy_policy(source: Path, output: Path) -> None:
    if not source.is_file():
        raise RuntimeError(f"RoboTwin deploy policy is missing: {source}")
    output.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")


def read_log_tail(path: Path) -> str:
    with path.open("rb") as handle:
        handle.seek(0, os.SEEK_END)
        size = handle.tell()
        handle.seek(max(0, size - FAILURE_LOG_TAIL_BYTES))
        return handle.read().decode("utf-8", errors="replace").strip()


def stop_child_process(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=10)


def main() -> None:
    args = parse_args()
    if args.poll_interval <= 0:
        raise ValueError("poll-interval must be positive")
    try:
        WorkerLoop(
            worker_id=args.worker_id,
            queue_dir=args.queue_dir,
            state_dir=args.worker_state_dir,
            poll_interval_s=args.poll_interval,
        ).run(lambda item: run_task(args, item))
    finally:
        if _SESSION is not None:
            _SESSION.stop()


if __name__ == "__main__":
    main()

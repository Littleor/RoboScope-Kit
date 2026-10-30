# Copyright 2026 RoboScope Team
# SPDX-License-Identifier: Apache-2.0

"""Run an official persistent LIBERO worker over a shared queue."""

# ruff: noqa: E402, I001

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import imageio
import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPOSITORY_ROOT))

from roboscope_kit.runtime.benchmark import (
    EvaluationRecord,
    WorkerLoop,
    WorkItem,
    WorkResult,
)

from benchmarks.libero_official.starvla_runtime.libero_client import (
    LIBERO_DUMMY_ACTION,
    LIBERO_ENV_RESOLUTION,
    build_example,
    response_action,
)
from benchmarks.libero_official.starvla_runtime.libero_policy_stepper import (
    PolicyStepper,
)

MAX_STEPS = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
    "libero_90": 400,
}


@dataclass
class TaskRuntime:
    task_id: int
    task_description: str
    initial_states: np.ndarray
    env: object


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker-id", type=int, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--suite", default="libero_10", choices=sorted(MAX_STEPS))
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--settle-steps", type=int, default=10)
    parser.add_argument("--queue-dir", type=Path, required=True)
    parser.add_argument("--worker-state-dir", type=Path, required=True)
    parser.add_argument("--video-dir", type=Path)
    parser.add_argument("--poll-interval", type=float, default=0.05)
    return parser.parse_args()


def save_video(
    video_dir: Path | None,
    task_id: int,
    episode_index: int,
    success: bool,
    frames: list[np.ndarray],
) -> None:
    if video_dir is None:
        return
    video_dir.mkdir(parents=True, exist_ok=True)
    suffix = "success" if success else "failure"
    path = video_dir / f"task-{task_id:02d}-episode-{episode_index:02d}-{suffix}.mp4"
    imageio.mimwrite(path, frames, fps=10)


def create_task_runtime(task_suite: object, task_id: int, seed: int) -> TaskRuntime:
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    task = task_suite.get_task(task_id)
    task_bddl_file = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = OffScreenRenderEnv(
        bddl_file_name=str(task_bddl_file),
        camera_heights=LIBERO_ENV_RESOLUTION,
        camera_widths=LIBERO_ENV_RESOLUTION,
    )
    env.seed(seed)
    return TaskRuntime(
        task_id=task_id,
        task_description=task.language,
        initial_states=task_suite.get_task_init_states(task_id),
        env=env,
    )


def run_episode(
    runtime: TaskRuntime,
    stepper: PolicyStepper,
    episode_index: int,
    *,
    max_steps: int,
    settle_steps: int,
    save_frames: bool,
) -> tuple[bool, int, float, list[np.ndarray]]:
    stepper.reset(runtime.task_description)
    runtime.env.reset()
    observation = runtime.env.set_init_state(runtime.initial_states[episode_index])
    frames: list[np.ndarray] = []
    action_step = 0
    started = time.perf_counter()

    for _ in range(settle_steps):
        observation, _reward, task_done, _info = runtime.env.step(LIBERO_DUMMY_ACTION)
        if task_done or runtime.env.env.done:
            return bool(task_done), action_step, time.perf_counter() - started, frames

    for _ in range(max_steps):
        if save_frames:
            frames.append(build_example(observation)["image"][0])
        policy_step = stepper.step(observation, runtime.task_description, action_step)
        observation, _reward, task_done, _info = runtime.env.step(response_action(policy_step.response))
        action_step += 1
        if task_done or runtime.env.env.done:
            return bool(task_done), action_step, time.perf_counter() - started, frames

    return False, action_step, time.perf_counter() - started, frames


def run_chunk(
    args: argparse.Namespace,
    item: WorkItem,
    runtime: TaskRuntime,
    stepper: PolicyStepper,
) -> WorkResult:
    records: list[EvaluationRecord] = []
    started = time.perf_counter()
    start_episode = int(item.payload["start_episode"])
    num_trials = int(item.payload["num_trials"])
    for episode_index in range(start_episode, start_episode + num_trials):
        success, steps, elapsed_s, frames = run_episode(
            runtime,
            stepper,
            episode_index,
            max_steps=MAX_STEPS[args.suite],
            settle_steps=args.settle_steps,
            save_frames=args.video_dir is not None,
        )
        save_video(args.video_dir, runtime.task_id, episode_index, success, frames)
        records.append(
            EvaluationRecord(
                record_id=f"task-{runtime.task_id:04d}-episode-{episode_index:04d}",
                success=success,
                steps=steps,
                elapsed_s=elapsed_s,
                metadata={"task_id": runtime.task_id, "episode_index": episode_index},
            )
        )
    return WorkResult(
        item_id=item.item_id,
        group_id=item.group_id,
        worker_id=args.worker_id,
        records=tuple(records),
        metadata={
            "suite": args.suite,
            "task_id": runtime.task_id,
            "task_description": runtime.task_description,
            "elapsed_s": time.perf_counter() - started,
        },
    )


def main() -> None:
    args = parse_args()
    if args.poll_interval <= 0:
        raise ValueError("poll-interval must be positive")
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    from examples.LIBERO.eval_files.model2libero_interface import ModelClient
    from libero.libero import benchmark

    task_suite = benchmark.get_benchmark_dict()[args.suite]()
    client = ModelClient(host=args.host, port=args.port, unnorm_key="franka")
    stepper = PolicyStepper(client, build_example)
    runtime: TaskRuntime | None = None

    def evaluate(item: WorkItem) -> WorkResult:
        nonlocal runtime
        task_id = int(item.payload["task_id"])
        if runtime is None or runtime.task_id != task_id:
            if runtime is not None:
                runtime.env.close()
            runtime = create_task_runtime(task_suite, task_id, args.seed)
        return run_chunk(args, item, runtime, stepper)

    try:
        WorkerLoop(
            worker_id=args.worker_id,
            queue_dir=args.queue_dir,
            state_dir=args.worker_state_dir,
            poll_interval_s=args.poll_interval,
        ).run(evaluate)
    finally:
        client.client.close()
        if runtime is not None:
            runtime.env.close()


if __name__ == "__main__":
    main()

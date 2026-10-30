# Copyright 2026 RoboScope Team
# SPDX-License-Identifier: Apache-2.0

"""Run persistent LIBERO episodes against an SDK-native user policy."""

from __future__ import annotations

import argparse
import os
import time
from dataclasses import dataclass
from pathlib import Path

import imageio
import numpy as np

from roboscope_kit import EpisodeContext, Observation, PolicyClient
from roboscope_kit.runtime.benchmark import EvaluationRecord, WorkerLoop, WorkItem, WorkResult

LIBERO_DUMMY_ACTION = np.asarray([0.0] * 6 + [-1.0], dtype=np.float32)
LIBERO_ENV_RESOLUTION = 256
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


class PolicyActionQueue:
    def __init__(self, socket_path: Path) -> None:
        self.socket_path = socket_path
        self.episode_id = ""
        self.actions = np.empty((0, 7), dtype=np.float32)
        self.inference_wait_s = 0.0

    def reset(self, context: EpisodeContext) -> None:
        self.episode_id = context.episode_id
        self.actions = np.empty((0, 7), dtype=np.float32)
        with PolicyClient(self.socket_path) as client:
            client.reset(context)

    def step(self, observation: dict, task_description: str, step: int) -> np.ndarray:
        if not self.actions.size:
            request = build_observation(self.episode_id, step, observation, task_description)
            started = time.perf_counter()
            with PolicyClient(self.socket_path) as client:
                chunk = client.act(request).actions
            self.inference_wait_s += time.perf_counter() - started
            self.actions = chunk.reshape(1, -1) if chunk.ndim == 1 else chunk
        action = self.actions[0].copy()
        self.actions = self.actions[1:]
        return action

    def end_episode(self) -> None:
        if not self.episode_id:
            return
        episode_id = self.episode_id
        self.episode_id = ""
        self.actions = np.empty((0, 7), dtype=np.float32)
        with PolicyClient(self.socket_path) as client:
            client.end_episode(episode_id)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker-id", type=int, required=True)
    parser.add_argument("--policy-socket", type=Path, required=True)
    parser.add_argument("--suite", choices=sorted(MAX_STEPS), required=True)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--settle-steps", type=int, default=10)
    parser.add_argument("--queue-dir", type=Path, required=True)
    parser.add_argument("--worker-state-dir", type=Path, required=True)
    parser.add_argument("--video-dir", type=Path)
    parser.add_argument("--poll-interval", type=float, default=0.05)
    return parser.parse_args()


def build_observation(
    episode_id: str,
    step: int,
    observation: dict,
    task_description: str,
) -> Observation:
    primary = np.ascontiguousarray(observation["agentview_image"][::-1, ::-1], dtype=np.uint8)
    wrist = np.ascontiguousarray(
        observation["robot0_eye_in_hand_image"][::-1, ::-1],
        dtype=np.uint8,
    )
    state = np.concatenate(
        (
            _state_vector(observation, "robot0_eef_pos", 3),
            _state_vector(observation, "robot0_eef_quat", 4),
            _state_vector(observation, "robot0_gripper_qpos", 2),
            _state_vector(observation, "robot0_joint_pos", 7),
        )
    ).astype(np.float32, copy=False)
    return Observation(
        episode_id=episode_id,
        step=step,
        timestamp_s=time.time(),
        features={
            "images.primary": primary,
            "images.wrist": wrist,
            "state.proprioception": state,
            "language.instruction": task_description,
        },
    )


def _state_vector(observation: dict, name: str, expected: int) -> np.ndarray:
    if name not in observation:
        raise KeyError(f"LIBERO observation is missing {name!r}")
    value = np.asarray(observation[name], dtype=np.float32).reshape(-1)
    if value.size != expected:
        raise ValueError(f"LIBERO observation {name!r} has {value.size} values; expected {expected}")
    return value


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
    policy: PolicyActionQueue,
    episode_index: int,
    *,
    max_steps: int,
    settle_steps: int,
    save_frames: bool,
) -> tuple[bool, int, float, list[np.ndarray]]:
    episode_id = f"task-{runtime.task_id:04d}-episode-{episode_index:04d}"
    policy.reset(
        EpisodeContext(
            episode_id=episode_id,
            task_id=str(runtime.task_id),
            metadata={"episode_index": episode_index, "instruction": runtime.task_description},
        )
    )
    try:
        runtime.env.reset()
        observation = runtime.env.set_init_state(runtime.initial_states[episode_index])
        frames: list[np.ndarray] = []
        started = time.perf_counter()

        for _ in range(settle_steps):
            observation, _reward, task_done, _info = runtime.env.step(LIBERO_DUMMY_ACTION)
            if task_done or runtime.env.env.done:
                return bool(task_done), 0, time.perf_counter() - started, frames

        for action_step in range(max_steps):
            if save_frames:
                frames.append(
                    np.ascontiguousarray(observation["agentview_image"][::-1, ::-1], dtype=np.uint8)
                )
            action = policy.step(observation, runtime.task_description, action_step)
            observation, _reward, task_done, _info = runtime.env.step(action)
            if task_done or runtime.env.env.done:
                return bool(task_done), action_step + 1, time.perf_counter() - started, frames

        return False, max_steps, time.perf_counter() - started, frames
    finally:
        policy.end_episode()


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


def run_chunk(
    args: argparse.Namespace,
    item: WorkItem,
    runtime: TaskRuntime,
    policy: PolicyActionQueue,
) -> WorkResult:
    records: list[EvaluationRecord] = []
    started = time.perf_counter()
    inference_wait_before = policy.inference_wait_s
    start_episode = int(item.payload["start_episode"])
    num_trials = int(item.payload["num_trials"])
    for episode_index in range(start_episode, start_episode + num_trials):
        success, steps, elapsed_s, frames = run_episode(
            runtime,
            policy,
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
            "inference_wait_s": policy.inference_wait_s - inference_wait_before,
        },
    )


def main() -> None:
    args = parse_args()
    if args.poll_interval <= 0:
        raise ValueError("poll-interval must be positive")
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    from libero.libero import benchmark

    task_suite = benchmark.get_benchmark_dict()[args.suite]()
    policy = PolicyActionQueue(args.policy_socket)
    runtime: TaskRuntime | None = None

    def evaluate(item: WorkItem) -> WorkResult:
        nonlocal runtime
        task_id = int(item.payload["task_id"])
        if runtime is None or runtime.task_id != task_id:
            if runtime is not None:
                runtime.env.close()
            runtime = create_task_runtime(task_suite, task_id, args.seed)
        return run_chunk(args, item, runtime, policy)

    try:
        WorkerLoop(
            worker_id=args.worker_id,
            queue_dir=args.queue_dir,
            state_dir=args.worker_state_dir,
            poll_interval_s=args.poll_interval,
        ).run(evaluate)
    finally:
        if runtime is not None:
            runtime.env.close()


if __name__ == "__main__":
    main()

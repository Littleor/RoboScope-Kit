# Copyright 2026 RoboScope Team
# SPDX-License-Identifier: Apache-2.0

"""Persistent RoboCasa GR1 Tabletop worker using an SDK Policy replica."""

from __future__ import annotations

import argparse
import time
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np

from roboscope_kit import EpisodeContext, Observation, PolicyClient
from roboscope_kit.runtime.benchmark import (
    EvaluationRecord,
    WorkerLoop,
    WorkItem,
    WorkQueue,
    WorkResult,
)

ACTION_DIMENSION = 29
STATE_KEYS = ("state.left_arm", "state.right_arm", "state.left_hand", "state.right_hand", "state.waist")
IMAGE_KEYS = ("video.ego_view", "video.ego_view_bg_crop_pad_res256_freq20")
LANGUAGE_KEY = "annotation.human.coarse_action"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker-id", type=int, required=True)
    parser.add_argument("--policy-socket", type=Path, required=True)
    parser.add_argument("--benchmark-root", type=Path, required=True)
    parser.add_argument("--queue-dir", type=Path, required=True)
    parser.add_argument("--worker-state-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--action-steps", type=int, default=12)
    parser.add_argument("--max-episode-steps", type=int, default=720)
    parser.add_argument("--disable-video", action="store_true")
    parser.add_argument("--poll-interval", type=float, default=0.05)
    return parser.parse_args()


def _latest(value: Any) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim < 2 or array.shape[0] != 1:
        raise RuntimeError(f"RoboCasa observation must have one environment, received {array.shape}")
    return np.asarray(array[0, -1] if array.ndim >= 3 else array[0])


def _instruction(value: Any) -> str:
    current = value
    while isinstance(current, (list, tuple, np.ndarray)):
        if len(current) == 0:
            raise RuntimeError("RoboCasa instruction cannot be empty")
        current = current[0]
    instruction = str(current).strip()
    if not instruction:
        raise RuntimeError("RoboCasa instruction cannot be blank")
    return instruction


def build_observation(episode_id: str, step: int, native: dict[str, Any]) -> Observation:
    image_key = next((name for name in IMAGE_KEYS if name in native), None)
    if image_key is None:
        raise RuntimeError("RoboCasa observation is missing the ego-view image")
    missing_state = [name for name in STATE_KEYS if name not in native]
    if missing_state:
        raise RuntimeError(f"RoboCasa observation is missing state features: {', '.join(missing_state)}")
    state = np.concatenate([_latest(native[name]).reshape(-1) for name in STATE_KEYS])
    if state.shape != (ACTION_DIMENSION,):
        raise RuntimeError(f"RoboCasa state shape is {state.shape}, expected {(ACTION_DIMENSION,)}")
    return Observation(
        episode_id=episode_id,
        step=step,
        timestamp_s=time.time(),
        features={
            "images.primary": np.ascontiguousarray(_latest(native[image_key]), dtype=np.uint8),
            "state.proprioception": np.ascontiguousarray(state, dtype=np.float32),
            "language.instruction": _instruction(native[LANGUAGE_KEY]),
        },
    )


def native_action(chunk: np.ndarray, action_steps: int) -> dict[str, np.ndarray]:
    actions = np.asarray(chunk, dtype=np.float32)
    if actions.ndim == 1:
        actions = actions.reshape(1, -1)
    if actions.ndim != 2 or actions.shape[1] != ACTION_DIMENSION:
        raise RuntimeError(
            f"RoboCasa Policy action shape is {actions.shape}, expected (N, {ACTION_DIMENSION})"
        )
    if actions.shape[0] < action_steps:
        raise RuntimeError(
            f"RoboCasa Policy returned {actions.shape[0]} actions, but {action_steps} are required"
        )
    selected = np.ascontiguousarray(actions[:action_steps])
    return {
        "action.left_arm": selected[None, :, 0:7],
        "action.right_arm": selected[None, :, 7:14],
        "action.left_hand": selected[None, :, 14:20],
        "action.right_hand": selected[None, :, 20:26],
        "action.waist": selected[None, :, 26:29],
    }


def run_task(args: argparse.Namespace, item: WorkItem) -> WorkResult:
    import gymnasium as gym
    from examples.simBenchmarks.Robocasa_tabletop.eval_files.simulation_env import (
        MultiStepConfig,
        SimulationConfig,
        VideoConfig,
        _create_single_env,
    )

    task_name = str(item.payload["task_name"])
    env_name = str(item.payload["env_name"])
    episodes = int(item.payload["episodes"])
    video_dir = None
    if not args.disable_video:
        video_dir = str(args.output_dir / "video" / f"worker-{args.worker_id:03d}" / task_name)
    config = SimulationConfig(
        env_name=env_name,
        n_episodes=1,
        n_envs=1,
        video=VideoConfig(video_dir=video_dir),
        multistep=MultiStepConfig(
            n_action_steps=args.action_steps,
            max_episode_steps=args.max_episode_steps,
        ),
    )
    environment = gym.vector.SyncVectorEnv([partial(_create_single_env, config=config, idx=0)])
    records: list[EvaluationRecord] = []
    queue = WorkQueue(args.queue_dir)
    started = time.perf_counter()
    try:
        for episode_index in range(episodes):
            native_observation, _ = environment.reset()
            episode_id = f"{task_name}-episode-{episode_index:04d}"
            episode_started = time.perf_counter()
            native_steps = 0
            success = False
            done = False
            with PolicyClient(args.policy_socket) as client:
                client.reset(
                    EpisodeContext(
                        episode_id=episode_id,
                        task_id=task_name,
                        metadata={"env_name": env_name, "episode_index": episode_index},
                    )
                )
                try:
                    while not done:
                        action_chunk = client.act(
                            build_observation(episode_id, native_steps, native_observation)
                        ).actions
                        native_observation, _, terminations, truncations, infos = environment.step(
                            native_action(action_chunk, args.action_steps)
                        )
                        native_steps += args.action_steps
                        success |= bool(np.asarray(infos["success"])[0, 0])
                        done = bool(terminations[0] or truncations[0])
                finally:
                    client.end_episode(episode_id)
            records.append(
                EvaluationRecord(
                    record_id=f"{task_name}-episode-{episode_index:04d}",
                    success=success,
                    steps=native_steps,
                    elapsed_s=time.perf_counter() - episode_started,
                    metadata={
                        "task_name": task_name,
                        "env_name": env_name,
                        "episode_index": episode_index,
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
                    metadata={"task_name": task_name, "env_name": env_name},
                ),
            )
    finally:
        environment.close()
    return WorkResult(
        item_id=item.item_id,
        group_id=item.group_id,
        worker_id=args.worker_id,
        records=tuple(records),
        metadata={
            "task_name": task_name,
            "env_name": env_name,
            "elapsed_s": time.perf_counter() - started,
        },
    )


def main() -> None:
    args = parse_args()
    if args.action_steps < 1 or args.max_episode_steps < 1 or args.poll_interval <= 0:
        raise ValueError("RoboCasa worker numeric arguments must be positive")
    WorkerLoop(
        worker_id=args.worker_id,
        queue_dir=args.queue_dir,
        state_dir=args.worker_state_dir,
        poll_interval_s=args.poll_interval,
    ).run(lambda item: run_task(args, item))


if __name__ == "__main__":
    main()

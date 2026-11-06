# Copyright 2026 RoboScope Team
# SPDX-License-Identifier: Apache-2.0

"""Persistent robomimic rollout worker using an SDK Policy replica.

The worker reproduces the episode loop of the pinned upstream
``robomimic/scripts/run_trained_agent.py`` (``rollout()``) for one seeded episode:

1. ``np.random.seed(seed)``, then ``env.reset()``, ``env.get_state()`` and
   ``env.reset_to(state)`` (upstream's "hack that is necessary for robosuite tasks
   for deterministic action playback");
2. up to ``horizon`` steps of ``action = policy(obs)``, ``env.step(action)``, then
   ``success = env.is_success()["task"]``, stopping on ``done or success``;
3. the episode succeeds when the last executed step reported task success, and its
   executed steps are ``step_i + 1``.

Environments are created through robomimic's ``EnvUtils.create_env_from_metadata``
from the frozen task table, so observation assembly (``object`` from ``object-state``
and the per-robot keys) is upstream's own ``EnvRobosuite`` code.
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import Any

import numpy as np

from roboscope_kit import EpisodeContext, Observation
from roboscope_kit.client import PolicyClient
from roboscope_kit.runtime.benchmark import EvaluationRecord, WorkerLoop, WorkItem, WorkResult

from .tasks import (
    PROTOCOL_VERSION,
    SUCCESS_CRITERION,
    TASKS_BY_NAME,
    RobomimicTask,
    initial_state_sha256,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker-id", type=int, required=True)
    parser.add_argument("--policy-socket", type=Path, required=True)
    parser.add_argument("--suite", required=True)
    parser.add_argument("--queue-dir", type=Path, required=True)
    parser.add_argument("--worker-state-dir", type=Path, required=True)
    parser.add_argument("--poll-interval", type=float, default=0.05)
    return parser.parse_args()


def initialize_observation_modalities() -> None:
    """Register every low-dim key of the frozen tasks, as ``policy_from_checkpoint`` would.

    ``EnvRobosuite.get_observation`` consults robomimic's global key-to-modality map; the
    upstream script fills it from the checkpoint config, which lists only low-dim keys.
    """

    import robomimic.utils.obs_utils as ObsUtils

    keys = sorted({key for task in TASKS_BY_NAME.values() for key in task.observation_keys})
    ObsUtils.initialize_obs_utils_with_obs_specs({"obs": {"low_dim": keys, "rgb": []}})


def create_env(task: RobomimicTask):
    import robomimic.utils.env_utils as EnvUtils

    env = EnvUtils.create_env_from_metadata(
        env_meta=task.env_meta(),
        render=False,
        render_offscreen=False,
        use_image_obs=False,
    )
    if env.action_dimension != task.action_dim:
        raise RuntimeError(
            f"robomimic task {task.name} action dimension is {env.action_dimension}, "
            f"expected {task.action_dim}"
        )
    return env


def sample_initial_state(env, seed: int) -> dict[str, Any]:
    """The state upstream ``run_trained_agent.py --seed <seed> --n_rollouts 1`` rolls out from."""

    np.random.seed(seed)
    env.reset()
    return env.get_state()


def reset_episode(env, seed: int) -> tuple[dict[str, Any], str]:
    """Seed and reset exactly as upstream ``rollout()``: reset, get_state, reset_to."""

    state = sample_initial_state(env, seed)
    observation = env.reset_to(state)
    return observation, initial_state_sha256(state["states"])


def build_observation(task: RobomimicTask, episode_id: str, step: int, observation: dict) -> Observation:
    features: dict[str, Any] = {}
    for key, dimension in task.observation_dims().items():
        if key not in observation:
            raise KeyError(f"robomimic observation for {task.name} is missing {key!r}")
        value = np.ascontiguousarray(observation[key], dtype=np.float32).reshape(-1)
        if value.size != dimension:
            raise ValueError(
                f"robomimic observation {key!r} for {task.name} has {value.size} values; expected {dimension}"
            )
        features[f"state.{key}"] = value
    return Observation(episode_id=episode_id, step=step, timestamp_s=time.time(), features=features)


def episode_id_for(task: RobomimicTask, seed: int) -> str:
    return f"{task.name}-seed-{seed:04d}"


def run_episode(
    client: PolicyClient,
    env,
    task: RobomimicTask,
    seed: int,
) -> tuple[EvaluationRecord, float]:
    episode_id = episode_id_for(task, seed)
    observation, state_digest = reset_episode(env, seed)
    client.reset(
        EpisodeContext(
            episode_id=episode_id,
            task_id=task.name,
            metadata={
                "protocol": PROTOCOL_VERSION,
                "seed": seed,
                "env_name": task.env_name,
                "horizon": task.horizon,
            },
        )
    )
    started = time.perf_counter()
    inference_wait_s = 0.0
    actions = np.empty((0, task.action_dim), dtype=np.float32)
    success = False
    steps = 0
    try:
        for step in range(task.horizon):
            if not actions.size:
                request = build_observation(task, episode_id, step, observation)
                requested = time.perf_counter()
                chunk = np.asarray(client.act(request).actions)
                inference_wait_s += time.perf_counter() - requested
                if chunk.ndim == 1:
                    chunk = chunk.reshape(1, -1)
                if chunk.ndim != 2 or chunk.shape[0] == 0 or chunk.shape[1] != task.action_dim:
                    raise RuntimeError(
                        f"robomimic Policy action shape is {chunk.shape}, expected (N, {task.action_dim})"
                    )
                actions = chunk
            action = actions[0]
            actions = actions[1:]
            observation, _reward, done, _info = env.step(action)
            steps = step + 1
            success = bool(env.is_success()["task"])
            if done or success:
                break
    finally:
        client.end_episode(episode_id)
    elapsed_s = time.perf_counter() - started
    record = EvaluationRecord(
        record_id=episode_id,
        success=success,
        steps=steps,
        elapsed_s=elapsed_s,
        metadata={
            "task": task.name,
            "seed": seed,
            "initial_state_sha256": state_digest,
            "horizon": task.horizon,
            "protocol": PROTOCOL_VERSION,
            "success_criterion": SUCCESS_CRITERION,
        },
    )
    return record, inference_wait_s


def main() -> None:
    args = parse_args()
    if args.worker_id < 0 or args.poll_interval <= 0:
        raise ValueError("robomimic worker numeric arguments must be positive")
    # Runtimes may live on read-only or inode-limited storage; never write pyc.
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    initialize_observation_modalities()
    envs: dict[str, Any] = {}

    def evaluate(item: WorkItem, client: PolicyClient) -> WorkResult:
        task = TASKS_BY_NAME[str(item.payload["task"])]
        seeds = [int(seed) for seed in item.payload["seeds"]]
        env = envs.get(task.name)
        if env is None:
            env = envs[task.name] = create_env(task)
        started = time.perf_counter()
        records = []
        inference_wait_s = 0.0
        for seed in seeds:
            record, wait = run_episode(client, env, task, seed)
            records.append(record)
            inference_wait_s += wait
        return WorkResult(
            item_id=item.item_id,
            group_id=item.group_id,
            worker_id=args.worker_id,
            records=tuple(records),
            metadata={
                "suite": args.suite,
                "task": task.name,
                "elapsed_s": time.perf_counter() - started,
                "inference_wait_s": inference_wait_s,
            },
        )

    with PolicyClient(args.policy_socket) as client:
        WorkerLoop(
            worker_id=args.worker_id,
            queue_dir=args.queue_dir,
            state_dir=args.worker_state_dir,
            poll_interval_s=args.poll_interval,
        ).run(lambda item: evaluate(item, client))


if __name__ == "__main__":
    main()

# Copyright 2026 RoboScope Team
# SPDX-License-Identifier: Apache-2.0

"""Persistent MetaWorld MT50 evaluator worker using an SDK Policy replica.

The worker consumes the upstream ``metaworld.evaluation.evaluation`` loop
unchanged. It builds one single-task vector environment per Work Item from a
single frozen ``MT50(seed=42)`` train-task artifact, wraps the vector to
translate finished episodes into SDK records, and records every executed
training goal by its artifact identity.
"""

from __future__ import annotations

import argparse
import os
import pickle
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from roboscope_kit.client import PolicyClient
from roboscope_kit.runtime.benchmark import (
    EvaluationRecord,
    WorkerLoop,
    WorkItem,
    WorkResult,
)

from .adapter import (
    EPISODE_HORIZON,
    MT50_GOALS_PER_TASK,
    MT50_TASK_COUNT,
    MT50_TOTAL_EPISODES,
    SEED,
)
from .agent import SUCCESS_CRITERION, MetaworldAgent
from .config import load_evaluation_config
from .instructions import FROZEN_MT50_TASKS_SHA256, INSTRUCTIONS, MT50_TASKS, mt50_train_tasks_sha256


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker-id", type=int, required=True)
    parser.add_argument("--policy-socket", type=Path, required=True)
    parser.add_argument("--evaluation-config", type=Path, required=True)
    parser.add_argument("--queue-dir", type=Path, required=True)
    parser.add_argument("--worker-state-dir", type=Path, required=True)
    parser.add_argument("--poll-interval", type=float, default=0.05)
    return parser.parse_args()


class RecordingVectorWrapper:
    """Thin VectorEnv wrapper that turns each finished episode into one record."""

    def __init__(self, env, *, slug: str, goals: dict[bytes, tuple[int, str]]) -> None:
        self.env = env
        self.slug = slug
        self.goals = goals
        self.records: list[EvaluationRecord] = []
        self._goal: tuple[int, str] | None = None
        self._started = 0.0

    def __getattr__(self, name: str) -> Any:
        return getattr(self.env, name)

    def reset(self, **kwargs):
        result = self.env.reset(**kwargs)
        self._goal = self._current_goal()
        self._started = time.perf_counter()
        return result

    def step(self, actions):
        goal = self._goal
        if goal is None:
            raise RuntimeError("MetaWorld vector environment stepped before its first reset")
        output = self.env.step(actions)
        terminations, truncations, infos = output[2], output[3], output[4]
        for index, done in enumerate(np.logical_or(terminations, truncations)):
            if done:
                self.records.append(
                    self._record(index, goal, bool(terminations[index]), bool(truncations[index]), infos)
                )
                self._started = time.perf_counter()
        # The SAME_STEP auto reset already selected the next goal; capture it
        # now so the following episode reports its frozen-artifact identity.
        self._goal = self._current_goal()
        return output

    def _record(
        self,
        index: int,
        goal: tuple[int, str],
        terminated: bool,
        truncated: bool,
        infos: dict[str, Any],
    ) -> EvaluationRecord:
        final_info = infos["final_info"]
        goal_id, goal_sha256 = goal
        return EvaluationRecord(
            record_id=f"{self.slug}-goal-{goal_id:02d}",
            success=bool(final_info["success"][index]),
            steps=int(final_info["episode"]["l"][index]),
            elapsed_s=time.perf_counter() - self._started,
            metadata={
                "task": self.slug,
                "goal_id": goal_id,
                "goal_sha256": goal_sha256,
                "terminated": terminated,
                "truncated": truncated,
                "protocol": "metaworld-v1",
                "episode_horizon": EPISODE_HORIZON,
                "success_criterion": SUCCESS_CRITERION,
            },
        )

    def _current_goal(self) -> tuple[int, str]:
        rand_vec = self.env.envs[0].unwrapped._last_rand_vec
        if rand_vec is None:
            raise RuntimeError("MetaWorld environment has no active task")
        key = np.asarray(rand_vec).tobytes()
        if key not in self.goals:
            raise RuntimeError(f"MetaWorld task {self.slug} active goal is not in the frozen artifact")
        return self.goals[key]


def _validate_upstream(metaworld_module, env_dict) -> None:
    from metaworld import _init_each_env

    if set(INSTRUCTIONS) != set(env_dict):
        raise RuntimeError("metaworld instructions do not match the upstream environment list")
    if metaworld_module.SawyerXYZEnv.max_path_length != EPISODE_HORIZON:
        raise RuntimeError(
            f"MetaWorld max_path_length is {metaworld_module.SawyerXYZEnv.max_path_length}, "
            f"expected {EPISODE_HORIZON}"
        )
    env = _init_each_env(env_cls=env_dict["reach-v3"], tasks=[], render_mode=None)
    try:
        if env.observation_space.shape[-1] != 39:
            raise RuntimeError(
                f"MetaWorld observation dimension is {env.observation_space.shape[-1]}, expected 39"
            )
        action_space = env.action_space
        if (
            action_space.shape != (4,)
            or action_space.dtype != np.float32
            or not np.array_equal(action_space.low, np.full(4, -1.0, dtype=np.float32))
            or not np.array_equal(action_space.high, np.full(4, 1.0, dtype=np.float32))
        ):
            raise RuntimeError(f"MetaWorld action space is {action_space}, expected Box(4, float32) [-1, 1]")
    finally:
        env.close()


def _load_mt50_tasks(metaworld_module) -> dict[str, tuple]:
    tasks = list(metaworld_module.MT50(seed=SEED).train_tasks)
    if len(tasks) != MT50_TOTAL_EPISODES:
        raise RuntimeError(f"MT50(seed={SEED}) generated {len(tasks)} tasks, expected {MT50_TOTAL_EPISODES}")
    grouped: dict[str, list] = defaultdict(list)
    for task in tasks:
        grouped[task.env_name].append(task)
    if len(grouped) != MT50_TASK_COUNT or set(grouped) != set(MT50_TASKS):
        raise RuntimeError("MT50(seed=42) does not cover exactly the 50 Farama task slugs")
    if any(len(values) != MT50_GOALS_PER_TASK for values in grouped.values()):
        raise RuntimeError("MT50(seed=42) does not hold 50 training goals for every task")
    digest = mt50_train_tasks_sha256(tasks)
    if digest != FROZEN_MT50_TASKS_SHA256:
        raise RuntimeError(
            f"MT50 train-task artifact digest {digest} does not match the frozen "
            f"{FROZEN_MT50_TASKS_SHA256}"
        )
    return {slug: tuple(grouped[slug]) for slug in MT50_TASKS}


def _validate_cameras(cameras: tuple[dict[str, Any], ...], env_dict) -> None:
    import mujoco

    for slug, env_cls in env_dict.items():
        env = env_cls()
        try:
            for camera in cameras:
                name = str(camera["camera_name"])
                if mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_CAMERA, name) < 0:
                    raise RuntimeError(f"metaworld camera {name!r} is missing from task {slug}")
        finally:
            env.close()


def _seed_environment(env, seed: int) -> None:
    from gymnasium.utils import seeding

    unwrapped = env.unwrapped
    unwrapped.np_random, _ = seeding.np_random(seed)
    unwrapped.action_space.seed(seed)
    unwrapped.observation_space.seed(seed)
    if getattr(unwrapped, "goal_space", None) is not None:
        unwrapped.goal_space.seed(seed)


def _build_vector(env):
    import gymnasium as gym

    vector = gym.vector.SyncVectorEnv(
        [lambda: env],
        autoreset_mode=gym.vector.AutoresetMode.SAME_STEP,
    )
    vector.call("toggle_sample_tasks_on_reset", True)
    return vector


def _goal_identities(tasks: tuple) -> dict[bytes, tuple[int, str]]:
    identities: dict[bytes, tuple[int, str]] = {}
    for index, task in enumerate(tasks):
        rand_vec = pickle.loads(task.data)["rand_vec"]
        key = np.asarray(rand_vec).tobytes()
        if key in identities:
            raise RuntimeError("MT50 train goals are not unique within a task")
        identities[key] = (
            index,
            mt50_train_tasks_sha256((task,)),
        )
    return identities


def _build_env(env_cls, tasks: tuple, camera: dict[str, Any]):
    from metaworld import _init_each_env as upstream_init_each_env

    env = upstream_init_each_env(
        env_cls=env_cls,
        tasks=list(tasks),
        seed=SEED,
        max_episode_steps=EPISODE_HORIZON,
        task_select="pseudorandom",
        use_one_hot=False,
        render_mode="rgb_array",
        camera_name=str(camera["camera_name"]),
        width=int(camera["render_width"]),
        height=int(camera["render_height"]),
    )
    _seed_environment(env, SEED)
    return env


def _run_task(
    item: WorkItem,
    *,
    worker_id: int,
    client: PolicyClient,
    tasks_by_env: dict[str, tuple],
    env_dict,
    cameras: tuple[dict[str, Any], ...],
) -> WorkResult:
    import metaworld.evaluation as metaworld_evaluation

    slug = str(item.payload["task"])
    episodes = int(item.payload["episodes"])
    if slug not in tasks_by_env:
        raise RuntimeError(f"MetaWorld work item references an unknown task {slug!r}")
    started = time.perf_counter()
    env = _build_env(env_dict[slug], tasks_by_env[slug], cameras[0])
    vector = None
    agent = None
    try:
        vector = _build_vector(env)
        recorder = RecordingVectorWrapper(vector, slug=slug, goals=_goal_identities(tasks_by_env[slug]))
        agent = MetaworldAgent(
            client=client,
            env=env,
            slug=slug,
            instruction=INSTRUCTIONS[slug],
            cameras=cameras,
        )
        metrics = metaworld_evaluation.evaluation(agent, recorder, num_episodes=episodes)
        records = tuple(recorder.records)
        if len(records) != episodes:
            raise RuntimeError(f"MetaWorld task {slug} produced {len(records)} records, expected {episodes}")
        if len({record.record_id for record in records}) != episodes:
            raise RuntimeError(f"MetaWorld task {slug} produced duplicate goal identities")
    finally:
        if agent is not None:
            agent.close()
        if vector is not None:
            vector.close()
        else:
            env.close()
    return WorkResult(
        item_id=item.item_id,
        group_id=slug,
        worker_id=worker_id,
        records=records,
        metadata={
            "task": slug,
            "upstream_task_success_rate": float(metrics[2][slug]),
            "elapsed_s": time.perf_counter() - started,
        },
    )


def main() -> None:
    args = parse_args()
    if args.worker_id < 0 or args.poll_interval <= 0:
        raise ValueError("MetaWorld worker numeric arguments must be positive")
    # Shared AutoDL FS inode quota can be exhausted; never write pyc.
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

    import metaworld
    from metaworld.env_dict import ALL_V3_ENVIRONMENTS

    evaluation_config = load_evaluation_config(args.evaluation_config)
    cameras = tuple(evaluation_config["cameras"])
    _validate_upstream(metaworld, ALL_V3_ENVIRONMENTS)
    tasks_by_env = _load_mt50_tasks(metaworld)
    _validate_cameras(cameras, ALL_V3_ENVIRONMENTS)

    with PolicyClient(args.policy_socket) as client:
        WorkerLoop(
            worker_id=args.worker_id,
            queue_dir=args.queue_dir,
            state_dir=args.worker_state_dir,
            poll_interval_s=args.poll_interval,
        ).run(
            lambda item: _run_task(
                item,
                worker_id=args.worker_id,
                client=client,
                tasks_by_env=tasks_by_env,
                env_dict=ALL_V3_ENVIRONMENTS,
                cameras=cameras,
            )
        )


if __name__ == "__main__":
    main()

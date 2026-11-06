# Copyright 2026 RoboScope Team
# SPDX-License-Identifier: Apache-2.0

"""Frozen robomimic v0.1 task table for the ``ph_low_dim`` profile.

Every value here is the robomimic v0.1 (CoRL 2021) study setting that the released
proficient-human (PH) BC-RNN low-dimensional model-zoo checkpoints were trained and
evaluated with: the ``env_metadata`` stored in each checkpoint (identical to the PH
low-dim dataset's ``env_args``) and the checkpoint's ``experiment.rollout.horizon``.
The worker builds each environment from this table, never from a model checkpoint,
so the benchmark side does not read model files.

This module must import without the simulator: the kit, the CLI, and the build script
all read it.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

PROTOCOL_VERSION = "robomimic-v1"
SUITE = "ph_low_dim"
SUITES = (SUITE,)
EPISODES_PER_TASK = 50
# Episode i of every task runs with seed i: the worker calls ``np.random.seed(i)`` before
# the environment reset that samples the initial state, and the Policy receives the same
# seed for its own sampling. The upstream ``run_trained_agent.py --seed i --n_rollouts 1``
# rolls out exactly the same initial state.
SEEDS = tuple(range(EPISODES_PER_TASK))
CONTROL_FREQ_HZ = 20
ACTION_REPRESENTATION = "robomimic.osc-pose-delta-v1"
SUCCESS_CRITERION = "terminate_on_first_task_success"
INITIAL_STATE_DIGEST_DECIMALS = 9

# The OSC_POSE controller of the robomimic v0.1 datasets and checkpoints.
OSC_POSE_CONTROLLER: dict[str, Any] = {
    "type": "OSC_POSE",
    "input_max": 1,
    "input_min": -1,
    "output_max": [0.05, 0.05, 0.05, 0.5, 0.5, 0.5],
    "output_min": [-0.05, -0.05, -0.05, -0.5, -0.5, -0.5],
    "kp": 150,
    "damping": 1,
    "impedance_mode": "fixed",
    "kp_limits": [0, 300],
    "damping_limits": [0, 10],
    "position_limits": None,
    "orientation_limits": None,
    "uncouple_pos_ori": True,
    "control_delta": True,
    "interpolation": None,
    "ramp_ratio": 0.2,
}

ROBOT_STATE_KEYS = ("eef_pos", "eef_quat", "gripper_qpos")
ROBOT_STATE_DIMS = {"eef_pos": 3, "eef_quat": 4, "gripper_qpos": 2}
ARM_ACTION_DIM = 7


@dataclass(frozen=True)
class RobomimicTask:
    name: str
    env_name: str
    robots: tuple[str, ...]
    horizon: int
    object_dim: int
    initial_states_sha256: str
    env_configuration: str | None = None

    @property
    def action_dim(self) -> int:
        return ARM_ACTION_DIM * len(self.robots)

    @property
    def observation_keys(self) -> tuple[str, ...]:
        """robomimic low-dim observation keys, in the checkpoints' config order."""

        keys = [f"robot{index}_{key}" for index in range(len(self.robots)) for key in ROBOT_STATE_KEYS]
        return (*keys, "object")

    def observation_dims(self) -> dict[str, int]:
        dims = {
            f"robot{index}_{key}": ROBOT_STATE_DIMS[key]
            for index in range(len(self.robots))
            for key in ROBOT_STATE_KEYS
        }
        dims["object"] = self.object_dim
        return dims

    def env_meta(self) -> dict[str, Any]:
        """The robomimic ``env_meta`` this task's environment is created from."""

        kwargs: dict[str, Any] = {
            "has_renderer": False,
            "has_offscreen_renderer": False,
            "ignore_done": True,
            "use_object_obs": True,
            "use_camera_obs": False,
            "control_freq": CONTROL_FREQ_HZ,
            "controller_configs": dict(OSC_POSE_CONTROLLER),
            "robots": list(self.robots),
            "camera_depths": False,
            "camera_heights": 84,
            "camera_widths": 84,
            "reward_shaping": False,
        }
        if self.env_configuration is not None:
            kwargs["env_configuration"] = self.env_configuration
        # robomimic EnvType.ROBOSUITE_TYPE
        return {"env_name": self.env_name, "type": 1, "env_kwargs": kwargs}


TASKS: tuple[RobomimicTask, ...] = (
    RobomimicTask(
        name="lift",
        env_name="Lift",
        robots=("Panda",),
        horizon=400,
        object_dim=10,
        initial_states_sha256="33ab18606642725c9c43b52bf753b34d2518bdec0e85f5aceaa7e176f7ebae27",
    ),
    RobomimicTask(
        name="can",
        env_name="PickPlaceCan",
        robots=("Panda",),
        horizon=400,
        object_dim=14,
        initial_states_sha256="645962b9270d0ac982f2422e85e1dcf5431f028dcbc769b7c4621903d30a2e33",
    ),
    RobomimicTask(
        name="square",
        env_name="NutAssemblySquare",
        robots=("Panda",),
        horizon=400,
        object_dim=14,
        initial_states_sha256="7bb5a5ae88197ef482c3e7188ed773fc2f5802d8d31b98223657c0f31575f99a",
    ),
    RobomimicTask(
        name="transport",
        env_name="TwoArmTransport",
        robots=("Panda", "Panda"),
        env_configuration="single-arm-opposed",
        horizon=700,
        object_dim=41,
        initial_states_sha256="0a657d8dbe3fc132cdcdaf691b87c295d3b7fdcf7dff90dee7ccd83fbe0a5963",
    ),
    RobomimicTask(
        name="tool_hang",
        env_name="ToolHang",
        robots=("Panda",),
        horizon=700,
        object_dim=44,
        initial_states_sha256="5e6d17058a2803bcc1acce745aebb84c2db71d5e13adf9557ebc981a7e628576",
    ),
)
TASKS_BY_NAME = {task.name: task for task in TASKS}
TASK_NAMES = tuple(task.name for task in TASKS)
QUICK_TEST_TASK = "lift"
MAX_ROBOTS = max(len(task.robots) for task in TASKS)


def initial_state_sha256(states: Any) -> str:
    """Digest of one flattened MuJoCo initial state, rounded to absorb last-bit libm differences."""

    import numpy as np

    rounded = np.round(np.asarray(states, dtype=np.float64), INITIAL_STATE_DIGEST_DECIMALS) + 0.0
    return hashlib.sha256(rounded.astype("<f8").tobytes()).hexdigest()


def initial_states_sha256(episode_digests: Iterable[tuple[int, str]]) -> str:
    """Per-task digest over ``(seed, initial_state_sha256)`` in seed order."""

    digest = hashlib.sha256()
    for seed, episode_digest in sorted(episode_digests):
        digest.update(f"{seed}:{episode_digest}\n".encode())
    return digest.hexdigest()


__all__ = [
    "ACTION_REPRESENTATION",
    "ARM_ACTION_DIM",
    "CONTROL_FREQ_HZ",
    "EPISODES_PER_TASK",
    "MAX_ROBOTS",
    "OSC_POSE_CONTROLLER",
    "PROTOCOL_VERSION",
    "QUICK_TEST_TASK",
    "SEEDS",
    "SUCCESS_CRITERION",
    "SUITE",
    "SUITES",
    "TASKS",
    "TASKS_BY_NAME",
    "TASK_NAMES",
    "RobomimicTask",
    "initial_state_sha256",
    "initial_states_sha256",
]

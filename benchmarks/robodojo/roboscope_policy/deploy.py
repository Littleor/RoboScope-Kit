# Copyright 2026 RoboScope Team
# SPDX-License-Identifier: Apache-2.0

"""RoboDojo's single-environment evaluation loop for a RoboScope Policy."""

from __future__ import annotations

import os
import time

import numpy as np

from roboscope_kit import Observation


def _observation(episode_id, step, raw):
    vision, state = raw["vision"], raw["state"]
    joints = []
    for side in ("left", "right"):
        arm = np.asarray(state[f"{side}_arm_joint_state"], dtype=np.float32).reshape(-1)
        grip = np.asarray(state[f"{side}_ee_joint_state"], dtype=np.float32).reshape(-1)
        if arm.shape != (6,) or grip.shape != (1,):
            raise ValueError("RoboDojo X5 observation requires six joints and one gripper per arm")
        joints.extend((arm, grip))
    instruction = raw.get("instruction", "")
    if isinstance(instruction, (list, tuple)):
        instruction = instruction[0] if instruction else ""
    return Observation(
        episode_id=episode_id,
        step=step,
        features={
            "images.head": np.array(vision["cam_head"]["color"], dtype=np.uint8, order="C", copy=True),
            "images.left_wrist": np.array(
                vision["cam_left_wrist"]["color"], dtype=np.uint8, order="C", copy=True
            ),
            "images.right_wrist": np.array(
                vision["cam_right_wrist"]["color"], dtype=np.uint8, order="C", copy=True
            ),
            "state.joints": np.concatenate(joints),
            "state.ee_pose": np.concatenate([
                np.asarray(state[f"{side}_ee_pose"], dtype=np.float32).reshape(7)
                for side in ("left", "right")
            ]),
            "language.instruction": str(instruction or ""),
        },
        timestamp_s=None,
    )


def _quaternion(qxyz):
    xyz = np.asarray(qxyz, dtype=np.float64)
    squared_norm = float(np.dot(xyz, xyz))
    if squared_norm > 1.0 + 1e-5:
        raise ValueError("RoboDojo compact quaternion xyz squared norm exceeds one")
    if squared_norm > 1.0:
        xyz = xyz / np.sqrt(squared_norm)
        squared_norm = 1.0
    return np.concatenate(([np.sqrt(1.0 - squared_norm)], xyz)).astype(np.float32)


def _action(row, representation):
    row = np.asarray(row, dtype=np.float32)
    if row.shape != (14,) or not np.isfinite(row).all():
        raise ValueError("RoboDojo Policy actions must contain 14 finite values")
    if representation == "robodojo.x5-joint-v1":
        return {
            "left_arm_joint_state": row[:6],
            "left_ee_joint_state": row[6:7],
            "right_arm_joint_state": row[7:13],
            "right_ee_joint_state": row[13:14],
        }
    if representation == "robodojo.x5-ee-compact-quat-v1":
        return {
            "left_ee_pose": np.concatenate((row[:3], _quaternion(row[3:6]))),
            "left_ee_joint_state": row[6:7],
            "right_ee_pose": np.concatenate((row[7:10], _quaternion(row[10:13]))),
            "right_ee_joint_state": row[13:14],
        }
    raise ValueError(f"unsupported RoboDojo action representation: {representation!r}")


def eval_one_episode(TASK_ENV, model_client):
    task = str(TASK_ENV.task_name)
    layout_id = int(TASK_ENV.env_seeds[0])
    representation = os.environ["ROBOSCOPE_ROBODOJO_ACTION_REPRESENTATION"]
    step_updates = os.environ.get("ROBOSCOPE_ROBODOJO_STEP_UPDATES") == "1"
    episode_id = model_client.begin_episode(task, layout_id)
    started = time.monotonic()
    try:
        while not TASK_ENV.is_episode_end():
            step = int(TASK_ENV.take_action_cnt[0])
            observation = _observation(episode_id, step, TASK_ENV.get_obs())
            actions = np.asarray(model_client.act(observation), dtype=np.float32)
            if actions.ndim == 1:
                actions = actions[None, :]
            if actions.ndim != 2 or actions.shape[0] == 0 or actions.shape[1] != 14:
                raise ValueError("RoboDojo Policy returned an invalid action chunk")
            for index, action in enumerate(actions):
                TASK_ENV.take_action(_action(action, representation))
                if TASK_ENV.is_episode_end():
                    break
                if step_updates and index + 1 < len(actions):
                    observation = _observation(
                        episode_id, int(TASK_ENV.take_action_cnt[0]), TASK_ENV.get_obs()
                    )
                    model_client.observe(observation)
    except BaseException as primary:
        try:
            model_client.end_episode()
        except BaseException as cleanup:
            raise primary from cleanup
        raise
    finally:
        TASK_ENV._roboscope_episode_elapsed_s = time.monotonic() - started
    if model_client.episode_id:
        model_client.end_episode()

#!/usr/bin/env python3
"""Validate the X-VLA robotwin FK/proprio construction against the real sim.

Runs inside the platform RoboTwin runtime: loads one episode of a task env,
reads the sim's joints/endpose, and compares against the SAPIEN-based FK and
the X-VLA client proprio built by the adapter.  Prints raw numbers so any
frame/convention mismatch is visible.
"""
import os
import sys

import numpy as np

os.chdir("/root/autodl-fs/benchmarks/robotwin/runtime/root/RoboTwin")
sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.join(os.getcwd(), "envs"))
sys.path.insert(0, os.path.join(os.getcwd(), "scripts"))

TASK = sys.argv[1] if len(sys.argv) > 1 else "adjust_bottle"
SUITE = sys.argv[2] if len(sys.argv) > 2 else "demo_clean"

import yaml  # noqa: E402


def load_env(task_name, task_config):
    CONFIGS_PATH = os.path.join(os.getcwd(), "task_config")
    envs_module = __import__(f"envs.{task_name}", fromlist=[task_name])
    TASK_ENV = getattr(envs_module, task_name)()
    with open(os.path.join(CONFIGS_PATH, f"{task_config}.yml"), encoding="utf-8") as f:
        args = yaml.safe_load(f)
    args["task_name"] = task_name
    with open(os.path.join(CONFIGS_PATH, "_embodiment_config.yml"), encoding="utf-8") as f:
        _embodiment_types = yaml.safe_load(f)
    with open(os.path.join(CONFIGS_PATH, "_camera_config.yml"), encoding="utf-8") as f:
        _camera_config = yaml.safe_load(f)

    embodiment_type = args["embodiment"]

    def get_embodiment_file(etype):
        return _embodiment_types[etype]["file_path"]

    def get_embodiment_config(robot_file):
        with open(os.path.join(os.getcwd(), robot_file, "config.yml"), encoding="utf-8") as f:
            return yaml.safe_load(f)

    head_camera_type = args["camera"]["head_camera_type"]
    args["head_camera_h"] = _camera_config[head_camera_type]["h"]
    args["head_camera_w"] = _camera_config[head_camera_type]["w"]

    if len(embodiment_type) == 1:
        args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["right_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["dual_arm_embodied"] = True
    elif len(embodiment_type) == 3:
        args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["right_robot_file"] = get_embodiment_file(embodiment_type[1])
        args["embodiment_dis"] = embodiment_type[2]
        args["dual_arm_embodied"] = False
    else:
        raise ValueError("embodiment must have 1 or 3 entries")

    args["left_embodiment_config"] = get_embodiment_config(args["left_robot_file"])
    args["right_embodiment_config"] = get_embodiment_config(args["right_robot_file"])
    args["embodiment_name"] = "+".join(embodiment_type) if len(embodiment_type) > 1 else embodiment_type[0]
    args["task_config"] = task_config
    return TASK_ENV, args


def build_my_proprio(joints, fk_left, fk_right):
    """Replicate the adapter's _act_robotwin proprio (X-VLA client convention)."""
    from scipy.spatial.transform import Rotation as R

    def quat_to_rot6d(q):
        # q is treated as-is (the X-VLA client feeds the sim endpose quat raw)
        return R.from_quat(q).as_matrix()[..., :, :2].reshape(6)

    left_qpos = joints[0:6]
    right_qpos = joints[7:13]
    left_grip = joints[6]
    right_grip = joints[13]
    lp, lq = fk_left(left_qpos)   # (pos, quat wxyz)
    rp, rq = fk_right(right_qpos)
    return np.concatenate(
        [lp, quat_to_rot6d(lq), [1 - 2 * left_grip], rp, quat_to_rot6d(rq), [1 - 2 * right_grip]]
    )


def build_xvla_proprio(obs):
    from scipy.spatial.transform import Rotation as R

    def quat_to_rot6d(q):
        return R.from_quat(q).as_matrix()[..., :, :2].reshape(6)

    left_ee = np.array(obs["endpose"]["left_endpose"])
    right_ee = np.array(obs["endpose"]["right_endpose"])
    left_grip = np.array(obs["endpose"]["left_gripper"])
    right_grip = np.array(obs["endpose"]["right_gripper"])
    return np.concatenate(
        [
            left_ee[:3],
            quat_to_rot6d(left_ee[3:]),
            [1 - left_grip * 2],
            right_ee[:3],
            quat_to_rot6d(right_ee[3:]),
            [1 - right_grip * 2],
        ]
    )


def main():
    from validate_fk_lib import build_arm_models  # standalone SAPIEN FK

    TASK_ENV, args = load_env(TASK, SUITE)
    args["eval_mode"] = True
    args["render_freq"] = 0
    args["eval_video_log"] = False
    args["clear_cache_freq"] = 5
    args["ckpt_setting"] = "60k"
    args["policy_name"] = "validation"

    left_model, right_model = build_arm_models(SUITE)
    fk_left = left_model.forward_kinematics
    fk_right = right_model.forward_kinematics

    TASK_ENV.setup_demo(now_ep_num=0, seed=2000, is_test=True, **args)
    obs = TASK_ENV.get_obs()
    joints = np.array(obs["joint_action"]["vector"], dtype=np.float32)
    le = np.array(obs["endpose"]["left_endpose"], dtype=np.float32)
    re = np.array(obs["endpose"]["right_endpose"], dtype=np.float32)

    print("joints:", np.round(joints, 4).tolist())
    print("left endpose (sim):", np.round(le, 4).tolist())
    print("right endpose (sim):", np.round(re, 4).tolist())

    lp, lq = fk_left(joints[0:6])
    rp, rq = fk_right(joints[7:13])
    print("left FK pos:", np.round(lp, 4).tolist(), "quat(wxyz):", np.round(lq, 4).tolist())
    print("right FK pos:", np.round(rp, 4).tolist(), "quat(wxyz):", np.round(rq, 4).tolist())

    # quat sign handling for comparison
    def qalign(a, b):
        return a if np.dot(a, b) >= 0 else -a

    pos_err_l = np.abs(lp - le[:3]).max()
    pos_err_r = np.abs(rp - re[:3]).max()
    quat_err_l = np.abs(qalign(lq, le[3:]) - le[3:]).max()
    quat_err_r = np.abs(qalign(rq, re[3:]) - re[3:]).max()
    print(f"pos err L/R: {pos_err_l:.5f} {pos_err_r:.5f}   quat err L/R: {quat_err_l:.5f} {quat_err_r:.5f}")

    my = build_my_proprio(joints, fk_left, fk_right)
    xv = build_xvla_proprio(obs)
    print("proprio (mine):", np.round(my, 4).tolist())
    print("proprio (xvla):", np.round(xv, 4).tolist())
    print("proprio max abs diff:", round(float(np.abs(my - xv).max()), 5))

    TASK_ENV.close_env()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Standalone FK builder mirroring the X-VLA adapter's RoboTwin proprio path.

Imports nothing from the RoboScope SDK so it can run inside the RoboTwin
conda env for validation. FK uses a SAPIEN articulation loaded from the
RoboTwin URDF, the same model the simulator uses.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

ROBOTWIN_RUNTIME = Path(os.environ["ROBOTWIN_RUNTIME_ROOT"])


class _ArmFk:
    """SAPIEN-based forward kinematics for one arm chain.

    Empirically verified (2026-08-15): the sim endpose stored in obs equals the
    raw URDF link pose (``global_trans_matrix @ delta_matrix`` in the sim's
    ``_trans_endpose`` exactly compensates the joint-vs-link frame offset), so
    no extra frame transform is applied.
    """

    def __init__(self, urdf_path, move_group, base_pose, joint_names):
        import sapien  # noqa: PLC0415

        engine = sapien.Engine()
        self._scene = engine.create_scene()
        loader = self._scene.create_urdf_loader()
        loader.fix_root_link = True
        self._robot = loader.load(str(urdf_path))
        self._robot.set_root_pose(sapien.Pose(np.asarray(base_pose[0:3]), np.asarray(base_pose[3:7])))
        self._joint_index = {
            joint.get_name(): index for index, joint in enumerate(self._robot.get_active_joints())
        }
        self._move_group = move_group
        self._arm_joints = tuple(joint_names)
        self._links = {link.get_name(): link for link in self._robot.get_links()}

    def forward_kinematics(self, qpos: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return the sim endpose (pos3, quat_wxyz) for the given 6 arm joints."""
        full = np.zeros(self._robot.dof)
        for name, value in zip(self._arm_joints, qpos, strict=True):
            full[self._joint_index[name]] = value
        self._robot.set_qpos(full)
        pose = self._links[self._move_group].get_pose()
        return np.asarray(pose.p, dtype=np.float32), np.asarray(pose.q, dtype=np.float32)


def build_arm_models(suite: str):
    import yaml  # noqa: PLC0415

    def _task_config():
        for root in (ROBOTWIN_RUNTIME / "task_config", ROBOTWIN_RUNTIME / "env_cfg" / "task_config"):
            candidate = root / f"{suite}.yml"
            if candidate.is_file():
                return candidate
        raise FileNotFoundError(suite)

    def _embodiment_config():
        for candidate in (
            ROBOTWIN_RUNTIME / "task_config" / "_embodiment_config.yml",
            ROBOTWIN_RUNTIME / "env_cfg" / "task_config" / "_embodiment_config.yml",
        ):
            if candidate.is_file():
                return candidate
        raise FileNotFoundError("_embodiment_config.yml")

    task_config = yaml.safe_load(_task_config().read_text(encoding="utf-8"))
    embodiment_name = task_config["embodiment"][0]
    embodiment_cfg = yaml.safe_load(_embodiment_config().read_text(encoding="utf-8"))
    robot_dir = (ROBOTWIN_RUNTIME / embodiment_cfg[embodiment_name]["file_path"]).resolve()
    robot_cfg = yaml.safe_load((robot_dir / "config.yml").read_text(encoding="utf-8"))
    robot_pose = robot_cfg.get("robot_pose", [[0.0, -0.65, 0.0, 1.0, 0.0, 0.0, 1.0]])
    pose = robot_pose[0] if isinstance(robot_pose[0], (list, tuple)) else robot_pose
    urdf = (robot_dir / robot_cfg["urdf_path"]).resolve()
    left_names = robot_cfg["arm_joints_name"][0]
    right_names = robot_cfg["arm_joints_name"][1]
    left = _ArmFk(urdf, robot_cfg["move_group"][0], [float(v) for v in pose], left_names)
    right = _ArmFk(urdf, robot_cfg["move_group"][1], [float(v) for v in pose], right_names)
    return left, right

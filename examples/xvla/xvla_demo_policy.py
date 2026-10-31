"""X-VLA Policy example for RoboScope Kit.

Runs the released X-VLA-0.9B checkpoints (https://github.com/2toINF/X-VLA)
inside the RoboScope Kit Policy process.  The weights and code live under
``/opt/roboscope/xvla``; the platform-side model artifact is therefore a
placeholder and its content is ignored.

Benchmark-to-checkpoint mapping and the observation/action adapters follow the
X-VLA evaluation clients verbatim:

- calvin      -> X-VLA-Calvin-ABC_D  (domain_id 2, compact absolute EE)
- libero      -> X-VLA-Libero        (domain_id 3, absolute EE)
- libero_plus -> X-VLA-Libero        (same observation/action contract as libero)
- robotwin    -> X-VLA-RoboTwin2     (domain_id 6, compact absolute EE)

robocasa is intentionally unsupported: X-VLA has no RoboCasa checkpoint and
reports no RoboCasa metric, so the suite is out of scope.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np

from roboscope_kit.policy import (
    ActionChunk,
    EpisodeContext,
    Observation,
    Policy,
    PolicyCapabilities,
    SessionContext,
)

XVLA_ROOT = Path(os.environ.get("XVLA_ROOT", "/opt/roboscope/xvla"))
XVLA_REPO = XVLA_ROOT / "X-VLA"
ROBOTWIN_RUNTIME = Path(
    os.environ.get("ROBOTWIN_RUNTIME_ROOT", "/root/autodl-fs/benchmarks/robotwin/runtime/root/RoboTwin")
)

CHECKPOINTS = {
    "calvin": XVLA_ROOT / "checkpoints" / "calvin",
    "libero": XVLA_ROOT / "checkpoints" / "libero",
    "libero_plus": XVLA_ROOT / "checkpoints" / "libero",
    "robotwin": XVLA_ROOT / "checkpoints" / "robotwin",
}

DOMAIN_IDS = {"calvin": 2, "libero": 3, "libero_plus": 3, "robotwin": 6}
# Denoising steps for flow matching (matches the X-VLA clients' payload).
ACTION_STEPS = 10
# Rows of the action chunk each client consumes per inference: libero uses the
# full model chunk (num_actions), calvin the first 20 rows, robotwin the full
# chunk.
CALVIN_CHUNK_ROWS = 20

SUPPORTED_SUITES = {
    "calvin": ("task_D_D", "task_ABC_D", "task_ABCD_D"),
    "libero": ("libero_spatial", "libero_object", "libero_goal", "libero_10", "libero_90"),
    "libero_plus": ("libero_spatial", "libero_object", "libero_goal", "libero_10"),
    "robotwin": ("demo_clean", "demo_randomized"),
}

REQUIRED_FEATURES = {
    "calvin": ("images.static", "images.gripper", "state.proprioception", "language.instruction"),
    "libero": ("images.primary", "images.wrist", "state.proprioception", "language.instruction"),
    "libero_plus": ("images.primary", "images.wrist", "state.proprioception", "language.instruction"),
    "robotwin": (
        "images.head",
        "images.left_wrist",
        "images.right_wrist",
        "state.joints",
        "language.instruction",
    ),
}


def _libero_site_rotmat(quat_xyzw: np.ndarray) -> np.ndarray:
    """The LIBERO ``robot0_eef_quat`` observation is the EE *body* quat in xyzw,
    but the X-VLA client feeds the controller's *site* orientation (site_xmat).
    The site is rotated 90deg about z relative to the body (verified constant
    across states on the platform runtime), so R_site = R_body @ R_offset."""
    from scipy.spatial.transform import Rotation

    body = Rotation.from_quat(quat_xyzw).as_matrix()
    return body @ np.asarray([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])


def _quat_xyzw_to_rot6d(quat_xyzw: np.ndarray) -> np.ndarray:
    """Convert a [x,y,z,w] unit quaternion to the interleaved 6D rotation used
    by the X-VLA calvin/robotwin clients (matrix columns, row-major flattened)."""
    from scipy.spatial.transform import Rotation

    matrix = Rotation.from_quat(quat_xyzw).as_matrix()
    return np.asarray(matrix)[..., :, :2].reshape(6)


def _euler_xyz_to_rot6d(euler: np.ndarray) -> np.ndarray:
    """Convert xyz euler angles to a 6D rotation (X-VLA calvin client)."""
    from scipy.spatial.transform import Rotation

    matrix = Rotation.from_euler("xyz", euler, degrees=False).as_matrix()
    return np.asarray(matrix)[..., :, :2].reshape(euler.shape[:-1] + (6,))


def _rot6d_to_axisangle(rot6d: np.ndarray) -> np.ndarray:
    """Convert 6D rotation to axis-angle (X-VLA libero client)."""
    from scipy.spatial.transform import Rotation

    a1 = rot6d[0:3]
    a2 = rot6d[3:6]
    b1 = a1 / (np.linalg.norm(a1) + 1e-6)
    b2 = a2 - np.sum(b1 * a2) * b1
    b2 = b2 / (np.linalg.norm(b2) + 1e-6)
    b3 = np.cross(b1, b2)
    matrix = np.stack([b1, b2, b3], axis=-1)
    quat_wxyz = Rotation.from_matrix(matrix).as_quat()  # [x,y,z,w]
    return Rotation.from_quat(quat_wxyz).as_rotvec()


def _rot6d_to_quat_xyzw(rot6d: np.ndarray) -> np.ndarray:
    """Convert the interleaved 6D rotation (X-VLA calvin/robotwin clients) to a
    [x,y,z,w] quaternion.  The two base vectors are the interleaved rows."""
    from scipy.spatial.transform import Rotation

    a1 = np.asarray([rot6d[0], rot6d[2], rot6d[4]])
    a2 = np.asarray([rot6d[1], rot6d[3], rot6d[5]])
    b1 = a1 / (np.linalg.norm(a1) + 1e-6)
    proj = np.sum(b1 * a2) * b1
    b2 = (a2 - proj) / (np.linalg.norm(a2 - proj) + 1e-6)
    b3 = np.cross(b1, b2)
    return Rotation.from_matrix(np.stack([b1, b2, b3], axis=-1)).as_quat()  # [x,y,z,w]


def _canonicalize_quat_xyzw(quat_xyzw: np.ndarray) -> np.ndarray:
    """Return a unit ``[x,y,z,w]`` quaternion with the unique ``w >= 0`` sign."""

    quat = np.asarray(quat_xyzw, dtype=np.float64)
    if quat.shape != (4,) or not np.all(np.isfinite(quat)):
        raise ValueError("quaternion must contain four finite values")
    norm = float(np.linalg.norm(quat))
    if norm == 0.0:
        raise ValueError("quaternion norm cannot be zero")
    quat = quat / norm
    if quat[3] < 0.0:
        quat = -quat
    return quat.astype(np.float32)


class _RobotArmModel:
    """SAPIEN FK for one RoboTwin arm's initial Recipe proprioception.

    Empirically verified against the real sim (2026-08-15): the sim endpose
    stored in obs equals the raw URDF link pose (``global_trans_matrix @
    delta_matrix`` in the sim's ``_trans_endpose`` exactly compensates the
    joint-vs-link frame offset), so no extra frame transform is applied.  The
    X-VLA model operates in a scrambled presentation frame (the client feeds
    the wxyz sim quat to ``scipy Rotation.from_quat`` as if it were xyzw); the
    first-chunk proprio replicates that. Later chunks use the previous target
    quaternion exactly as the Recipe's overwritten observation does.
    """

    def __init__(
        self,
        urdf_path: Path,
        move_group: str,
        base_pose: list[float],
        arm_joint_names: tuple[str, ...],
    ) -> None:
        import sapien  # noqa: PLC0415  (only needed for robotwin)

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
        self._arm_joints = tuple(arm_joint_names)
        self._links = {link.get_name(): link for link in self._robot.get_links()}

    def forward_kinematics(self, qpos: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return the sim endpose (pos3, quat_wxyz) for the given 6 arm joints."""
        full = np.zeros(self._robot.dof)
        for name, value in zip(self._arm_joints, qpos, strict=True):
            full[self._joint_index[name]] = value
        self._robot.set_qpos(full)
        pose = self._links[self._move_group].get_pose()
        return np.asarray(pose.p, dtype=np.float32), np.asarray(pose.q, dtype=np.float32)


class XvlaDemoPolicy(Policy):
    def __init__(self) -> None:
        self.model = None
        self.processor = None
        self.device = None
        self.benchmark = ""
        self.domain_id = 0
        # Several simulator workers share one policy replica and the SDK
        # server serialises requests per call, not per episode, so concurrent
        # workers interleave reset/act calls across episodes.  This policy is
        # a temporal model that feeds the previous model output back as
        # proprio; that cross-step state must be keyed by episode_id or
        # concurrent workers corrupt each other's trajectories.
        self._proprio: dict[str, np.ndarray] = {}
        self._arms: dict[str, _RobotArmModel] = {}

    # ------------------------------------------------------------------ load
    def load(self, context: SessionContext) -> PolicyCapabilities:
        benchmark = context.benchmark.name
        if benchmark not in CHECKPOINTS:
            supported = ", ".join(sorted(SUPPORTED_SUITES))
            raise ValueError(
                f"X-VLA demo Policy supports these benchmarks: {supported}; got {benchmark!r}"
            )
        checkpoint_dir = CHECKPOINTS[benchmark]
        if not checkpoint_dir.is_dir():
            raise RuntimeError(f"X-VLA checkpoint directory is missing: {checkpoint_dir}")

        import sys  # noqa: PLC0415

        import torch  # noqa: PLC0415

        if str(XVLA_REPO) not in sys.path:
            sys.path.insert(0, str(XVLA_REPO))
        from models.modeling_xvla import XVLA  # noqa: PLC0415
        from models.processing_xvla import XVLAProcessor  # noqa: PLC0415

        self.model = XVLA.from_pretrained(
            str(checkpoint_dir),
            trust_remote_code=True,
            torch_dtype=torch.float32,
        ).to("cuda").to(torch.float32).eval()
        self.processor = XVLAProcessor.from_pretrained(str(checkpoint_dir))
        self.device = next(self.model.parameters()).device
        self.benchmark = benchmark
        self.domain_id = DOMAIN_IDS[benchmark]
        if benchmark == "robotwin":
            self._init_robotwin(context.benchmark.suite)
        # The calvin recipe consumes the first 20 rows of the model chunk; libero
        # and robotwin consume the full model chunk.
        chunk_sizes = (CALVIN_CHUNK_ROWS,) if benchmark == "calvin" else (self.model.num_actions,)
        return PolicyCapabilities(
            supported_benchmarks=(benchmark,),
            supported_suites={benchmark: SUPPORTED_SUITES[benchmark]},
            required_features=REQUIRED_FEATURES[benchmark],
            max_batch_size=1,
            supported_action_chunk_sizes=chunk_sizes,
            action_representations={
                benchmark: {
                    "robotwin": "robotwin.ee-compact-quat-v1",
                    "calvin": "calvin.absolute-compact-quat-v1",
                }.get(benchmark, context.benchmark.default_action_representation)
            },
            observation_strides={benchmark: 1},
        )

    def _init_robotwin(self, suite: str) -> None:
        import yaml  # noqa: PLC0415

        task_config_path = self._robotwin_task_config(suite)
        task_config = yaml.safe_load(task_config_path.read_text(encoding="utf-8"))
        embodiment_names = task_config["embodiment"]
        if len(embodiment_names) != 1:
            raise ValueError(
                "X-VLA robotwin adapter supports single-embodiment configs only: "
                f"{embodiment_names}"
            )
        embodiment_name = embodiment_names[0]
        embodiment_cfg_path = self._robotwin_embodiment_config()
        embodiment_cfg = yaml.safe_load(embodiment_cfg_path.read_text(encoding="utf-8"))
        if embodiment_name not in embodiment_cfg:
            raise RuntimeError(f"RoboTwin embodiment {embodiment_name!r} is not declared")
        robot_dir = (ROBOTWIN_RUNTIME / embodiment_cfg[embodiment_name]["file_path"]).resolve()
        robot_cfg = yaml.safe_load((robot_dir / "config.yml").read_text(encoding="utf-8"))
        for arm in ("left", "right"):
            urdf = (robot_dir / robot_cfg["urdf_path"]).resolve()
            arm_index = 0 if arm == "left" else 1
            move_group = robot_cfg["move_group"][arm_index]
            arm_joints = robot_cfg["arm_joints_name"][arm_index]
            robot_pose = robot_cfg.get("robot_pose", [[0.0, -0.65, 0.0, 1.0, 0.0, 0.0, 1.0]])
            pose = robot_pose[0] if isinstance(robot_pose[0], (list, tuple)) else robot_pose
            self._arms[arm] = _RobotArmModel(
                urdf,
                move_group,
                [float(value) for value in pose],
                arm_joints,
            )

    def _robotwin_task_config(self, suite: str) -> Path:
        for root in (ROBOTWIN_RUNTIME / "task_config", ROBOTWIN_RUNTIME / "env_cfg" / "task_config"):
            candidate = root / f"{suite}.yml"
            if candidate.is_file():
                return candidate
        raise RuntimeError(f"RoboTwin task config is missing for suite {suite!r}")

    def _robotwin_embodiment_config(self) -> Path:
        for candidate in (
            ROBOTWIN_RUNTIME / "task_config" / "_embodiment_config.yml",
            ROBOTWIN_RUNTIME / "env_cfg" / "task_config" / "_embodiment_config.yml",
        ):
            if candidate.is_file():
                return candidate
        raise RuntimeError(f"RoboTwin embodiment config is missing: {ROBOTWIN_RUNTIME}")

    # ---------------------------------------------------------------- reset
    def reset(self, context: EpisodeContext) -> None:
        # X-VLA clients re-sync proprio from the current observation at episode
        # start; the platform reset() carries no observation, so the next act()
        # re-initialises from the live state.
        self._proprio.pop(context.episode_id, None)

    def end_episode(self, episode_id: str) -> None:
        self._proprio.pop(episode_id, None)

    # ------------------------------------------------------------------- act
    def act(self, observation: Observation) -> ActionChunk:
        if self.model is None or self.processor is None:
            raise RuntimeError("X-VLA demo Policy requires load() before act()")
        features = observation.features
        ep = observation.episode_id
        if self.benchmark == "robotwin":
            return self._act_robotwin(features, ep)
        if self.benchmark == "calvin":
            return self._act_calvin(features, ep)
        return self._act_libero(features, ep)

    # --------------------------------------------------------- libero family
    def _act_libero(self, features: dict[str, Any], ep: str) -> ActionChunk:
        # Faithful port of the X-VLA libero_client.py: absolute end-effector
        # targets [pos, axis-angle, grip], 30-action chunk, proprio tracked from
        # the live EE pose (site frame = controller ee_ori_mat).
        state = _array(features, "state.proprioception")
        if state.shape[-1] != 16:
            raise ValueError(f"LIBERO proprioception has {state.shape[-1]} values; expected 16")
        current_pos = state[0:3].astype(np.float32)
        current_rot = _libero_site_rotmat(state[3:7].astype(np.float32))
        proprio = self._proprio.get(ep)
        if proprio is None:
            rot6d = np.concatenate([current_rot[:, 0], current_rot[:, 1]])
            closed_loop = np.concatenate([current_pos, rot6d, np.zeros(1)])
            proprio = np.concatenate([closed_loop, np.zeros_like(closed_loop)])
            self._proprio[ep] = proprio
        raw = self._generate(
            images=[_array(features, "images.primary"), _flip_back(_array(features, "images.wrist"))],
            instruction=_string(features, "language.instruction"),
            proprio=proprio,
        )
        actions = np.empty((raw.shape[0], 7), dtype=np.float32)
        for i, row in enumerate(raw):
            axis_angle = _rot6d_to_axisangle(row[3:9])
            grip = 1.0 if row[9] > 0.5 else -1.0
            actions[i] = np.concatenate([row[0:3], axis_angle, [grip]])
        proprio[:9] = raw[-1, :9]

        return ActionChunk(actions)

    # ----------------------------------------------------------------- calvin
    def _act_calvin(self, features: dict[str, Any], ep: str) -> ActionChunk:
        # Faithful port of the X-VLA calvin_client.py: absolute EE targets
        # (pos, quat_xyzw, grip) as a chunk; the calvin worker passes each row
        # to the env as the absolute tuple the recipe uses.
        robot_obs = _array(features, "state.proprioception")
        current_pos = robot_obs[0:3].astype(np.float32)
        proprio = self._proprio.get(ep)
        if proprio is None:
            rot6d = _euler_xyz_to_rot6d(robot_obs[3:6])
            grip_bit = 1.0 if robot_obs[6] > 0.0 else 0.0
            proprio = np.concatenate([current_pos, rot6d, [grip_bit]])
            # The recipe pads the single-arm proprio to the model's 20-dim
            # state contract with zeros, same as the libero branch.
            proprio = np.concatenate([proprio, np.zeros_like(proprio)])
            self._proprio[ep] = proprio
        raw = self._generate(
            images=[_array(features, "images.static"), _array(features, "images.gripper")],
            instruction=_string(features, "language.instruction"),
            proprio=proprio,
        )
        raw = raw[:CALVIN_CHUNK_ROWS]
        # The platform calvin spec is 7-dim; the recipe's absolute tuple is
        # (pos, quat_xyzw, grip).  Encode quat as its xyz part (the worker
        # reconstructs w from the unit quaternion) to stay within the spec.
        actions = np.empty((raw.shape[0], 7), dtype=np.float32)
        for i, row in enumerate(raw):
            quat_xyzw = _canonicalize_quat_xyzw(_rot6d_to_quat_xyzw(row[3:9]))
            grip = 1.0 if row[9] < 0.8 else -1.0
            actions[i] = np.concatenate([row[0:3], quat_xyzw[0:3], [grip]])
        proprio[:10] = raw[-1, :10]
        return ActionChunk(actions)

    # --------------------------------------------------------------- robotwin
    def _act_robotwin(self, features: dict[str, Any], ep: str) -> ActionChunk:
        joints = _array(features, "state.joints")
        if joints.shape[-1] != 14:
            raise ValueError(f"RoboTwin joints has {joints.shape[-1]} values; expected 14")
        left_qpos = joints[0:6]
        right_qpos = joints[7:13]
        left_grip = 1.0 - 2.0 * joints[6]
        right_grip = 1.0 - 2.0 * joints[13]
        proprio = self._proprio.get(ep)
        if proprio is None:
            left_pos, left_quat = self._arms["left"].forward_kinematics(left_qpos)
            right_pos, right_quat = self._arms["right"].forward_kinematics(right_qpos)
            # RoboTwin reports the initial endpose quaternion in wxyz, while the
            # Recipe passes it unchanged to scipy's xyzw API. Preserve that
            # presentation exactly for the first chunk.
            proprio = np.concatenate(
                [
                    left_pos,
                    _quat_xyzw_to_rot6d(left_quat),
                    [left_grip],
                    right_pos,
                    _quat_xyzw_to_rot6d(right_quat),
                    [right_grip],
                ]
            )
        else:
            # The Recipe overwrites obs.endpose with each commanded target. The
            # next chunk therefore sees the previous chunk's final target pose,
            # but its gripper values still come from the new live observation.
            proprio = proprio.copy()
            proprio[9] = left_grip
            proprio[19] = right_grip
        raw = self._generate(
            images=[
                _array(features, "images.head"),
                _array(features, "images.left_wrist"),
                _array(features, "images.right_wrist"),
            ],
            instruction=_string(features, "language.instruction"),
            proprio=proprio,
        )
        actions = np.empty((raw.shape[0], 14), dtype=np.float32)
        for i, row in enumerate(raw):
            left_quat = _canonicalize_quat_xyzw(_rot6d_to_quat_xyzw(row[3:9]))
            right_quat = _canonicalize_quat_xyzw(_rot6d_to_quat_xyzw(row[13:19]))
            left_command = 1.0 - 2.0 * float(row[9] > 0.7)
            right_command = 1.0 - 2.0 * float(row[19] > 0.7)
            actions[i] = np.concatenate(
                [
                    row[0:3],
                    left_quat[0:3],
                    [left_command],
                    row[10:13],
                    right_quat[0:3],
                    [right_command],
                ]
            )

        # rotate6D_to_quat orthogonalizes the model's two predicted basis
        # vectors before execution. Re-encode those quaternions so next-chunk
        # proprio matches the Recipe's overwritten target, not the raw 6D row.
        last = raw[-1]
        last_left_quat = _canonicalize_quat_xyzw(_rot6d_to_quat_xyzw(last[3:9]))
        last_right_quat = _canonicalize_quat_xyzw(_rot6d_to_quat_xyzw(last[13:19]))
        self._proprio[ep] = np.concatenate(
            [
                last[0:3],
                _quat_xyzw_to_rot6d(last_left_quat),
                [left_grip],
                last[10:13],
                _quat_xyzw_to_rot6d(last_right_quat),
                [right_grip],
            ]
        )
        return ActionChunk(actions)

    # ------------------------------------------------------------- inference
    def _generate(self, images: list[np.ndarray], instruction: str, proprio: np.ndarray) -> np.ndarray:
        import torch  # noqa: PLC0415

        inputs = self.processor(images, instruction)
        proprio_t = torch.as_tensor(np.asarray(proprio, dtype=np.float32), device=self.device)

        def to_model(value: Any) -> Any:
            if not isinstance(value, torch.Tensor):
                value = torch.as_tensor(value)
            if value.is_floating_point():
                return value.to(device=self.device, dtype=next(self.model.parameters()).dtype)
            return value.to(device=self.device)

        batch = {key: to_model(value) for key, value in inputs.items()}
        batch["domain_id"] = torch.tensor([self.domain_id], dtype=torch.long, device=self.device)
        with torch.inference_mode():
            action = self.model.generate_actions(**batch, proprio=proprio_t.unsqueeze(0), steps=ACTION_STEPS)
        return np.asarray(action.squeeze(0).float().cpu().numpy())

    def unload(self) -> None:
        import gc  # noqa: PLC0415

        import torch  # noqa: PLC0415

        self.model = None
        self.processor = None
        self._proprio.clear()
        self._arms = {}
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _array(features: dict[str, Any], name: str) -> np.ndarray:
    value = features.get(name)
    if not isinstance(value, np.ndarray):
        raise TypeError(f"X-VLA input feature {name!r} must be a numpy array")
    return np.ascontiguousarray(value)


def _string(features: dict[str, Any], name: str) -> str:
    value = features.get(name)
    if not isinstance(value, str) or not value.strip():
        raise TypeError(f"X-VLA input feature {name!r} must be a non-empty string")
    return value


def _flip_back(image: np.ndarray) -> np.ndarray:
    # The platform LIBERO worker already flips both cameras; X-VLA only expects
    # the agent view flipped, so undo the wrist flip.
    return np.ascontiguousarray(image[::-1, ::-1])


def main() -> None:
    from roboscope_kit.server import run  # noqa: PLC0415

    run(XvlaDemoPolicy())


if __name__ == "__main__":
    main()

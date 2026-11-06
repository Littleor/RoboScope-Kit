# Copyright 2026 RoboScope Team
# SPDX-License-Identifier: Apache-2.0

"""Built-in benchmark contracts exposed to user policy implementations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum

from roboscope_kit.contracts import ActionSpec, BenchmarkSpec, FeatureKind, FeatureSpec


class BenchmarkName(str, Enum):  # noqa: UP042 -- benchmark environments include Python 3.10
    LIBERO = "libero"
    LIBERO_PLUS = "libero_plus"
    LIBERO_PRO = "libero_pro"
    ROBOCASA = "robocasa"
    ROBOTWIN = "robotwin"
    CALVIN = "calvin"
    METAWORLD = "metaworld"
    ROBODOJO = "robodojo"
    ROBOMIMIC = "robomimic"


class BenchmarkDefinition(ABC):
    """Internal provider for one benchmark specification."""

    name: BenchmarkName
    protocol_version: str
    supported_action_representations: tuple[str, ...]
    default_action_representation: str

    @abstractmethod
    def features(self, suite: str) -> tuple[FeatureSpec, ...]: ...

    @abstractmethod
    def action(self, suite: str) -> ActionSpec: ...

    def spec(self, suite: str) -> BenchmarkSpec:
        if not suite:
            raise ValueError("benchmark suite cannot be empty")
        return BenchmarkSpec(
            name=self.name.value,
            suite=suite,
            protocol_version=self.protocol_version,
            features=self.features(suite),
            action=self.action(suite),
            supported_action_representations=self.supported_action_representations,
            default_action_representation=self.default_action_representation,
        )


def _feature(
    name: str,
    kind: FeatureKind,
    description: str,
    *,
    required: bool = True,
    shape: tuple[int | None, ...] | None = None,
) -> FeatureSpec:
    dtype = {
        FeatureKind.IMAGE: "uint8",
        FeatureKind.STATE: "float32",
        FeatureKind.LANGUAGE: "string",
    }[kind]
    return FeatureSpec(
        name=name,
        kind=kind,
        description=description,
        required=required,
        dtype=dtype,
        shape=shape,
    )


class LiberoBenchmarkDefinition(BenchmarkDefinition):
    name = BenchmarkName.LIBERO
    protocol_version = "libero-v1"
    supported_action_representations = ("libero.delta-eef-v1",)
    default_action_representation = "libero.delta-eef-v1"

    def features(self, _suite: str) -> tuple[FeatureSpec, ...]:
        return (
            _feature(
                "images.primary",
                FeatureKind.IMAGE,
                "Agent-view RGB image.",
                shape=(256, 256, 3),
            ),
            _feature(
                "images.wrist",
                FeatureKind.IMAGE,
                "Robot wrist RGB image.",
                shape=(256, 256, 3),
            ),
            _feature(
                "state.proprioception",
                FeatureKind.STATE,
                "EEF position/quaternion, gripper position, and seven joint positions.",
                shape=(16,),
            ),
            _feature("language.instruction", FeatureKind.LANGUAGE, "Task instruction."),
        )

    def action(self, _suite: str) -> ActionSpec:
        return ActionSpec("action.robot", "LIBERO 6-DoF end-effector delta and gripper command.", 7)


class LiberoPlusBenchmarkDefinition(LiberoBenchmarkDefinition):
    name = BenchmarkName.LIBERO_PLUS
    protocol_version = "libero-plus-v1"


class LiberoProBenchmarkDefinition(LiberoBenchmarkDefinition):
    name = BenchmarkName.LIBERO_PRO
    protocol_version = "libero-pro-v1"


class RoboCasaBenchmarkDefinition(BenchmarkDefinition):
    name = BenchmarkName.ROBOCASA
    protocol_version = "robocasa-v1"
    supported_action_representations = ("robocasa.gr1-joint-v1",)
    default_action_representation = "robocasa.gr1-joint-v1"

    def features(self, _suite: str) -> tuple[FeatureSpec, ...]:
        return (
            _feature("images.primary", FeatureKind.IMAGE, "Primary workspace RGB image."),
            _feature("images.wrist", FeatureKind.IMAGE, "Robot wrist RGB image.", required=False),
            _feature("state.proprioception", FeatureKind.STATE, "RoboCasa embodiment state vector."),
            _feature("language.instruction", FeatureKind.LANGUAGE, "Task instruction."),
        )

    def action(self, _suite: str) -> ActionSpec:
        return ActionSpec("action.robot", "RoboCasa GR1 arms, hands, and waist action vector.", 29)


class RoboTwinBenchmarkDefinition(BenchmarkDefinition):
    name = BenchmarkName.ROBOTWIN
    protocol_version = "robotwin2-v1"
    supported_action_representations = (
        "robotwin.qpos-v1",
        "robotwin.ee-compact-quat-v1",
    )
    default_action_representation = "robotwin.qpos-v1"

    def features(self, _suite: str) -> tuple[FeatureSpec, ...]:
        return (
            _feature("images.head", FeatureKind.IMAGE, "Head-camera RGB image."),
            _feature("images.left_wrist", FeatureKind.IMAGE, "Left wrist RGB image."),
            _feature("images.right_wrist", FeatureKind.IMAGE, "Right wrist RGB image."),
            _feature("state.joints", FeatureKind.STATE, "Dual-arm joint and gripper state."),
            _feature("language.instruction", FeatureKind.LANGUAGE, "Task instruction."),
        )

    def action(self, _suite: str) -> ActionSpec:
        return ActionSpec("action.robot", "Absolute dual-arm joint and gripper target.", 14)


class RoboDojoBenchmarkDefinition(BenchmarkDefinition):
    name = BenchmarkName.ROBODOJO
    protocol_version = "robodojo-v1"
    supported_action_representations = (
        "robodojo.x5-joint-v1",
        "robodojo.x5-ee-compact-quat-v1",
    )
    default_action_representation = "robodojo.x5-joint-v1"

    def features(self, _suite: str) -> tuple[FeatureSpec, ...]:
        return (
            _feature("images.head", FeatureKind.IMAGE, "Head-camera RGB image."),
            _feature("images.left_wrist", FeatureKind.IMAGE, "Left wrist RGB image."),
            _feature("images.right_wrist", FeatureKind.IMAGE, "Right wrist RGB image."),
            _feature(
                "state.joints", FeatureKind.STATE,
                "Left arm joints and gripper, followed by right arm joints and gripper.",
                shape=(14,),
            ),
            _feature(
                "state.ee_pose", FeatureKind.STATE,
                "Left and right end-effector poses, each [x, y, z, qw, qx, qy, qz].",
                shape=(14,),
            ),
            _feature("language.instruction", FeatureKind.LANGUAGE, "Task instruction."),
        )

    def action(self, _suite: str) -> ActionSpec:
        return ActionSpec(
            "action.robot", "Dual-arm X5 joint or compact end-effector targets and gripper commands.", 14
        )


class CalvinBenchmarkDefinition(BenchmarkDefinition):
    name = BenchmarkName.CALVIN
    protocol_version = "calvin-v1"
    supported_action_representations = (
        "calvin.relative-v1",
        "calvin.absolute-compact-quat-v1",
    )
    default_action_representation = "calvin.relative-v1"

    def features(self, _suite: str) -> tuple[FeatureSpec, ...]:
        return (
            _feature("images.static", FeatureKind.IMAGE, "Static scene RGB image."),
            _feature("images.gripper", FeatureKind.IMAGE, "Gripper RGB image."),
            _feature("state.proprioception", FeatureKind.STATE, "CALVIN robot state vector."),
            _feature("language.instruction", FeatureKind.LANGUAGE, "Task instruction."),
        )

    def action(self, _suite: str) -> ActionSpec:
        return ActionSpec("action.robot", "CALVIN relative end-effector and gripper action.", 7)


METAWORLD_DEFAULT_CAMERA = {
    "feature": "images.primary",
    "camera_name": "corner2",
    "render_width": 480,
    "render_height": 480,
    "camera_position": None,
    "rotation_deg": 180,
    "center_crop_ratio": 1.0,
    "output_width": 480,
    "output_height": 480,
}


class MetaworldBenchmarkDefinition(BenchmarkDefinition):
    name = BenchmarkName.METAWORLD
    protocol_version = "metaworld-v1"
    supported_action_representations = ("metaworld.delta-eef-v1",)
    default_action_representation = "metaworld.delta-eef-v1"

    def features(self, _suite: str, cameras: tuple[dict, ...] | None = None) -> tuple[FeatureSpec, ...]:
        resolved = (METAWORLD_DEFAULT_CAMERA,) if cameras is None else cameras
        features = [str(camera["feature"]) for camera in resolved]
        if not features or len(set(features)) != len(features):
            raise ValueError("metaworld camera features must be unique and non-empty")
        image_features = tuple(
            _feature(
                feature,
                FeatureKind.IMAGE,
                f"MetaWorld RGB image from camera {camera['camera_name']!r}.",
                shape=(int(camera["output_height"]), int(camera["output_width"]), 3),
            )
            for feature, camera in zip(features, resolved, strict=True)
        )
        return (
            *image_features,
            _feature(
                "state.proprioception",
                FeatureKind.STATE,
                "End-effector xyz position and normalized gripper aperture (raw observation[:4]).",
                shape=(4,),
            ),
            _feature("language.instruction", FeatureKind.LANGUAGE, "Official MetaWorld task instruction."),
        )

    def action(self, _suite: str) -> ActionSpec:
        return ActionSpec(
            "action.robot",
            "MetaWorld delta end-effector position (dx, dy, dz) and gripper command in [-1, 1].",
            4,
        )

    def spec(self, suite: str, cameras: tuple[dict, ...] | None = None) -> BenchmarkSpec:
        if not suite:
            raise ValueError("benchmark suite cannot be empty")
        return BenchmarkSpec(
            name=self.name.value,
            suite=suite,
            protocol_version=self.protocol_version,
            features=self.features(suite, cameras),
            action=self.action(suite),
            supported_action_representations=self.supported_action_representations,
            default_action_representation=self.default_action_representation,
        )


class RobomimicBenchmarkDefinition(BenchmarkDefinition):
    """robomimic v0.1 low-dimensional observations and OSC_POSE delta actions.

    Single-arm tasks (Lift, Can, Square, Tool Hang) send the ``robot0`` features and take
    7-dimensional actions; Transport also sends the ``robot1`` features and takes 14.
    """

    name = BenchmarkName.ROBOMIMIC
    protocol_version = "robomimic-v1"
    supported_action_representations = ("robomimic.osc-pose-delta-v1",)
    default_action_representation = "robomimic.osc-pose-delta-v1"

    def features(self, _suite: str) -> tuple[FeatureSpec, ...]:
        robot_features = []
        for index in range(2):
            for key, size, description in (
                ("eef_pos", 3, "end-effector position in the world frame"),
                ("eef_quat", 4, "end-effector orientation quaternion (x, y, z, w)"),
                ("gripper_qpos", 2, "gripper finger joint positions"),
            ):
                robot_features.append(
                    _feature(
                        f"state.robot{index}_{key}",
                        FeatureKind.STATE,
                        f"robosuite robot{index}_{key}: robot {index} {description}"
                        + ("." if index == 0 else " (two-arm Transport only)."),
                        required=index == 0,
                        shape=(size,),
                    )
                )
        return (
            *robot_features,
            _feature(
                "state.object",
                FeatureKind.STATE,
                "robomimic `object` observation (robosuite object-state); its size depends on the task.",
                shape=(None,),
            ),
        )

    def action(self, _suite: str) -> ActionSpec:
        return ActionSpec(
            "action.robot",
            "Per arm, robosuite OSC_POSE delta position (3), delta axis-angle rotation (3) and gripper (1) "
            "in [-1, 1]; Transport concatenates robot0 then robot1 (14).",
        )


_DEFINITIONS: dict[BenchmarkName, BenchmarkDefinition] = {
    definition.name: definition
    for definition in (
        LiberoBenchmarkDefinition(),
        LiberoPlusBenchmarkDefinition(),
        LiberoProBenchmarkDefinition(),
        RoboCasaBenchmarkDefinition(),
        RoboTwinBenchmarkDefinition(),
        RoboDojoBenchmarkDefinition(),
        CalvinBenchmarkDefinition(),
        MetaworldBenchmarkDefinition(),
        RobomimicBenchmarkDefinition(),
    )
}


def get_benchmark_spec(name: str | BenchmarkName, suite: str) -> BenchmarkSpec:
    benchmark = BenchmarkName(name)
    return _DEFINITIONS[benchmark].spec(suite)


def benchmark_names() -> tuple[str, ...]:
    return tuple(benchmark.value for benchmark in _DEFINITIONS)


__all__ = [
    "ActionSpec",
    "BenchmarkName",
    "BenchmarkSpec",
    "FeatureKind",
    "FeatureSpec",
    "benchmark_names",
    "get_benchmark_spec",
]

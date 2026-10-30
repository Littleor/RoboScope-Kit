# Copyright 2026 RoboScope Team
# SPDX-License-Identifier: Apache-2.0

"""SDK Policy bridge and canonical observation assembly for MetaWorld MT50."""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from roboscope_kit import EpisodeContext, Observation
from roboscope_kit.client import PolicyClient

METAWORLD_PROTOCOL_VERSION = "metaworld-v1"
SUCCESS_CRITERION = "any_step_success"


def process_image(image: np.ndarray, camera: dict[str, Any]) -> np.ndarray:
    """Rotate, center-crop, and resize one rendered camera frame."""

    rotation = int(camera["rotation_deg"])
    if rotation:
        image = np.rot90(image, k=rotation // 90)
    ratio = float(camera["center_crop_ratio"])
    height, width = image.shape[:2]
    crop_height = int(height * ratio)
    crop_width = int(width * ratio)
    if crop_height == 0 or crop_width == 0:
        raise ValueError(
            f"metaworld camera {camera['camera_name']!r} center crop produced a zero-sized image"
        )
    top = (height - crop_height) // 2
    left = (width - crop_width) // 2
    image = image[top : top + crop_height, left : left + crop_width]
    output_width = int(camera["output_width"])
    output_height = int(camera["output_height"])
    if image.shape[:2] != (output_height, output_width):
        from PIL import Image

        image = np.asarray(
            Image.fromarray(image).resize((output_width, output_height), Image.Resampling.BILINEAR)
        )
    return np.ascontiguousarray(image, dtype=np.uint8)


def _create_camera_renderer(env, camera: dict[str, Any]):
    import mujoco
    from gymnasium.envs.mujoco.mujoco_rendering import MujocoRenderer

    model = env.unwrapped.model
    camera_name = str(camera["camera_name"])
    camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
    if camera_id < 0:
        raise ValueError(f"metaworld camera {camera_name!r} does not exist in this task model")
    position = camera["camera_position"]
    if position is not None:
        if model.cam_mode[camera_id] != mujoco.mjtCamLight.mjCAMLIGHT_FIXED:
            raise ValueError(
                f"metaworld camera {camera_name!r} is not a fixed camera; "
                "camera_position override is unsupported"
            )
        model.cam_pos[camera_id] = position
    return MujocoRenderer(
        model,
        env.unwrapped.data,
        None,
        int(camera["render_width"]),
        int(camera["render_height"]),
        camera_id=camera_id,
    )


def _validate_framebuffer(renderers: list[Any], cameras: tuple[dict[str, Any], ...]) -> None:
    from OpenGL import GL

    largest = max(renderers, key=lambda renderer: max(renderer.width, renderer.height))
    largest.render("rgb_array")
    limits = [int(dimension) for dimension in GL.glGetIntegerv(GL.GL_MAX_VIEWPORT_DIMS)]
    max_dimension = max(max(camera["render_width"], camera["render_height"]) for camera in cameras)
    if max_dimension > min(limits):
        raise RuntimeError(
            f"metaworld render size {max_dimension} exceeds the OpenGL framebuffer limit {min(limits)}"
        )


class MetaworldAgent:
    """Bridge the upstream MetaWorld ``Agent`` protocol to one SDK PolicyClient."""

    def __init__(
        self,
        *,
        client: PolicyClient,
        env,
        slug: str,
        instruction: str,
        cameras: tuple[dict[str, Any], ...],
    ) -> None:
        self.client = client
        self.env = env
        self.slug = slug
        self.instruction = instruction
        self.cameras = cameras
        self.renderers = [_create_camera_renderer(env, camera) for camera in cameras]
        _validate_framebuffer(self.renderers, cameras)
        self.actions = np.empty((0, 4), dtype=np.float32)
        self.episode_id = ""
        self.step_count = 0
        self.episode_index = 0

    def reset(self, env_mask: np.ndarray) -> None:
        if env_mask.shape != (1,):
            raise ValueError(f"MetaWorld agent expects one environment, received mask {env_mask}")
        if not bool(env_mask[0]):
            return
        self._end_episode()
        name = str(self.env.unwrapped.task_name)
        self.episode_id = f"{self.slug}-{name}-episode-{self.episode_index:04d}"
        self.episode_index += 1
        self.step_count = 0
        self.actions = np.empty((0, 4), dtype=np.float32)
        self.client.reset(
            EpisodeContext(
                episode_id=self.episode_id,
                task_id=self.slug,
                metadata={"protocol": METAWORLD_PROTOCOL_VERSION},
            )
        )

    def eval_action(self, observations: np.ndarray) -> np.ndarray:
        if not self.episode_id:
            raise RuntimeError("MetaWorld policy action requested before an episode reset")
        if observations.shape != (1, 39):
            raise RuntimeError(f"MetaWorld observation shape is {observations.shape}, expected (1, 39)")
        if not self.actions.size:
            chunk = self.client.act(self._build_observation(observations[0])).actions
            actions = np.asarray(chunk, dtype=np.float32)
            if actions.ndim == 1:
                actions = actions.reshape(1, -1)
            if actions.ndim != 2 or actions.shape[1] != 4 or actions.shape[0] == 0:
                raise RuntimeError(f"MetaWorld Policy action shape is {actions.shape}, expected (N, 4)")
            self.actions = actions
        action = self.actions[0].copy()
        self.actions = self.actions[1:]
        self.step_count += 1
        return action.reshape(1, 4)

    def close(self) -> None:
        self._end_episode()
        for renderer in self.renderers:
            renderer.close()

    def _build_observation(self, observation: np.ndarray) -> Observation:
        features: dict[str, Any] = {
            camera["feature"]: process_image(renderer.render("rgb_array"), camera)
            for camera, renderer in zip(self.cameras, self.renderers, strict=True)
        }
        features["state.proprioception"] = np.ascontiguousarray(observation[:4], dtype=np.float32)
        features["language.instruction"] = self.instruction
        return Observation(
            episode_id=self.episode_id,
            step=self.step_count,
            timestamp_s=time.time(),
            features=features,
        )

    def _end_episode(self) -> None:
        if not self.episode_id:
            return
        episode_id = self.episode_id
        self.episode_id = ""
        self.actions = np.empty((0, 4), dtype=np.float32)
        self.client.end_episode(episode_id)


__all__ = ["METAWORLD_PROTOCOL_VERSION", "MetaworldAgent", "process_image"]

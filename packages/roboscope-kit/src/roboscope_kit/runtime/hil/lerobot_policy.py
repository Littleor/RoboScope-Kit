# Copyright 2026 RoboScope Team
# SPDX-License-Identifier: Apache-2.0

"""Pinned LeRobot Pi0.5 components for the local policy HIL runtime."""

from __future__ import annotations

import gc
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import numpy as np

from roboscope_kit.runtime.hil.checkpoints import VerifiedCheckpoint

PINNED_LEROBOT_VERSION = "0.6.0"


def validate_lerobot_runtime(runtime_version: str) -> None:
    try:
        installed = version("lerobot")
    except PackageNotFoundError as exc:
        raise RuntimeError("the pinned LeRobot runtime is not installed") from exc
    if installed != PINNED_LEROBOT_VERSION:
        raise RuntimeError(
            f"policy HIL requires LeRobot {PINNED_LEROBOT_VERSION}, found {installed}"
        )
    if not runtime_version.startswith(f"lerobot-{installed}"):
        raise RuntimeError(
            f"configured policy runtime {runtime_version!r} does not match installed LeRobot {installed}"
        )
    try:
        from lerobot.rollout.inference.rtc import RTCInferenceEngine  # noqa: F401
    except ImportError as exc:
        raise RuntimeError("installed LeRobot runtime does not provide native RTC inference") from exc


@dataclass(frozen=True)
class Pi05RuntimeConfig:
    checkpoint: Path
    tokenizer: Path
    device: str
    camera_key_mapping: dict[str, str]
    torch_threads: int
    gpu_memory_fraction: float
    rtc_execution_horizon: int
    rtc_queue_threshold: int
    rtc_max_guidance_weight: float


@dataclass
class LoadedPi05Runtime:
    policy: Any
    preprocessor: Any
    postprocessor: Any
    rtc_config: Any
    torch: Any
    device: Any
    checkpoint_sha256: str
    policy_config_sha256: str
    preprocessor_sha256: str
    postprocessor_sha256: str

    def reset(self) -> None:
        self.policy.reset()
        self.preprocessor.reset()
        self.postprocessor.reset()

    def unload(self) -> None:
        self.policy = None
        self.preprocessor = None
        self.postprocessor = None
        gc.collect()
        if self.device.type == "cuda":
            self.torch.cuda.empty_cache()


def load_pi05_runtime(
    runtime_config: Pi05RuntimeConfig,
    checkpoint: VerifiedCheckpoint,
) -> LoadedPi05Runtime:
    import torch
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.factory import get_policy_class, make_pre_post_processors
    from lerobot.policies.rtc.configuration_rtc import RTCConfig

    checkpoint_path = runtime_config.checkpoint.expanduser().resolve(strict=True)
    verified_path = checkpoint.entry.path.expanduser().resolve(strict=True)
    if checkpoint_path != verified_path:
        raise ValueError("runtime checkpoint differs from the verified catalog entry")
    tokenizer = runtime_config.tokenizer.expanduser().resolve(strict=True)
    if not tokenizer.is_dir():
        raise ValueError(f"tokenizer is not a directory: {tokenizer}")
    if runtime_config.torch_threads <= 0:
        raise ValueError("torch_threads must be positive")
    if not 0 < runtime_config.gpu_memory_fraction <= 1:
        raise ValueError("gpu_memory_fraction must be in (0, 1]")
    if runtime_config.rtc_execution_horizon <= 0:
        raise ValueError("rtc_execution_horizon must be positive")
    if runtime_config.rtc_queue_threshold < 0:
        raise ValueError("rtc_queue_threshold must be non-negative")
    if runtime_config.rtc_max_guidance_weight <= 0:
        raise ValueError("rtc_max_guidance_weight must be positive")

    torch.set_num_threads(runtime_config.torch_threads)
    device = torch.device(runtime_config.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required but is not available")
        torch.cuda.set_per_process_memory_fraction(runtime_config.gpu_memory_fraction, device)

    policy_config = PreTrainedConfig.from_pretrained(str(checkpoint_path))
    if policy_config.type != "pi05":
        raise ValueError(f"policy HIL supports pi05, received {policy_config.type!r}")
    policy_config.device = runtime_config.device
    policy_config.pretrained_path = str(checkpoint_path)
    chunk_size = int(policy_config.chunk_size)
    if runtime_config.rtc_execution_horizon > chunk_size:
        raise ValueError("rtc_execution_horizon exceeds the policy chunk size")
    if runtime_config.rtc_queue_threshold >= chunk_size:
        raise ValueError("rtc_queue_threshold must be smaller than the policy chunk size")

    policy_class = get_policy_class(policy_config.type)
    policy = policy_class.from_pretrained(str(checkpoint_path), config=policy_config)
    policy = policy.to(device)
    policy.eval()

    processor_overrides = {
        "rename_observations_processor": {
            "rename_map": _observation_rename_map(runtime_config.camera_key_mapping)
        },
        "tokenizer_processor": {"tokenizer_name": str(tokenizer)},
        "device_processor": {"device": str(device)},
    }
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy_config,
        pretrained_path=str(checkpoint_path),
        preprocessor_overrides=processor_overrides,
    )
    rtc_config = RTCConfig(
        enabled=True,
        execution_horizon=runtime_config.rtc_execution_horizon,
        max_guidance_weight=runtime_config.rtc_max_guidance_weight,
    )
    policy.config.rtc_config = rtc_config
    if not hasattr(policy, "init_rtc_processor"):
        raise RuntimeError(f"policy {type(policy).__name__} does not support RTC")
    policy.init_rtc_processor()
    return LoadedPi05Runtime(
        policy=policy,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        rtc_config=rtc_config,
        torch=torch,
        device=device,
        checkpoint_sha256=checkpoint.checkpoint_sha256,
        policy_config_sha256=checkpoint.policy_config_sha256,
        preprocessor_sha256=checkpoint.preprocessor_sha256,
        postprocessor_sha256=checkpoint.postprocessor_sha256,
    )


def dummy_observation(schema: dict[str, Any]) -> dict[str, np.ndarray]:
    state = schema.get("state")
    images = schema.get("images")
    if not isinstance(state, dict) or not isinstance(images, dict):
        raise ValueError("observation schema requires state and images")
    observation = {"observation.state": np.zeros(_schema_shape(state), dtype=np.float32)}
    for name, feature in images.items():
        observation[name] = np.zeros(_schema_shape(feature), dtype=np.uint8)
    return observation


def validate_observation(observation: dict[str, Any], schema: dict[str, Any]) -> dict[str, np.ndarray]:
    expected = dummy_observation(schema)
    result: dict[str, np.ndarray] = {}
    for name, template in expected.items():
        value = observation.get(name)
        if not isinstance(value, np.ndarray):
            raise TypeError(f"observation field {name!r} must be an ndarray")
        if value.shape != template.shape or value.dtype != template.dtype:
            raise ValueError(
                f"observation field {name!r} has {value.dtype} {value.shape}; "
                f"expected {template.dtype} {template.shape}"
            )
        result[name] = value
    unknown = sorted(set(observation) - set(expected))
    if unknown:
        raise ValueError("observation contains undeclared fields: " + ", ".join(unknown))
    return result


def _schema_shape(value: dict[str, Any]) -> tuple[int, ...]:
    shape = value.get("shape")
    if not isinstance(shape, list) or not shape:
        raise ValueError("schema feature shape must be a non-empty list")
    dimensions = tuple(int(item) for item in shape)
    if any(item <= 0 for item in dimensions):
        raise ValueError("schema feature dimensions must be positive")
    return dimensions


def _observation_rename_map(camera_key_mapping: dict[str, str]) -> dict[str, str]:
    if not camera_key_mapping:
        raise ValueError("camera_key_mapping must not be empty")
    result: dict[str, str] = {}
    for raw_key, policy_key in camera_key_mapping.items():
        source = _observation_image_key(raw_key)
        target = _observation_image_key(policy_key)
        if source in result:
            raise ValueError(f"duplicate raw camera key {raw_key!r}")
        result[source] = target
    if len(set(result.values())) != len(result):
        raise ValueError("camera_key_mapping produces duplicate policy keys")
    return result


def _observation_image_key(value: str) -> str:
    if not value:
        raise ValueError("camera keys must be non-empty")
    prefix = "observation.images."
    return value if value.startswith(prefix) else f"{prefix}{value}"

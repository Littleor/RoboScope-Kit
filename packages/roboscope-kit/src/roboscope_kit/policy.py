# Copyright 2026 RoboScope Team
# SPDX-License-Identifier: Apache-2.0

"""Public policy lifecycle types for custom RoboScope inference code."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from roboscope_kit.contracts import BenchmarkSpec, FeatureKind


@dataclass(frozen=True)
class SessionContext:
    """Task-scoped values prepared by the RoboScope runtime before policy loading."""

    session_id: str
    checkpoint_path: str
    benchmark: BenchmarkSpec
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.session_id:
            raise ValueError("session_id cannot be empty")

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> SessionContext:
        return cls(
            session_id=str(value["session_id"]),
            checkpoint_path=str(value["checkpoint_path"]),
            benchmark=BenchmarkSpec.from_dict(value["benchmark"]),
            metadata=dict(value.get("metadata") or {}),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "checkpoint_path": self.checkpoint_path,
            "benchmark": self.benchmark.to_dict(),
            "metadata": self.metadata,
        }


@dataclass(frozen=True)
class EpisodeContext:
    episode_id: str
    task_id: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.episode_id:
            raise ValueError("episode_id cannot be empty")
        if not self.task_id:
            raise ValueError("task_id cannot be empty")

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> EpisodeContext:
        return cls(
            episode_id=str(value["episode_id"]),
            task_id=str(value["task_id"]),
            metadata=dict(value.get("metadata") or {}),
        )

    def to_dict(self) -> dict[str, Any]:
        return {"episode_id": self.episode_id, "task_id": self.task_id, "metadata": self.metadata}


@dataclass(frozen=True)
class Observation:
    episode_id: str
    step: int
    features: dict[str, Any]
    timestamp_s: float | None = None

    def __post_init__(self) -> None:
        if not self.episode_id:
            raise ValueError("episode_id cannot be empty")
        if self.step < 0:
            raise ValueError("observation step cannot be negative")

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> Observation:
        return cls(
            episode_id=str(value["episode_id"]),
            step=int(value["step"]),
            features=dict(value["features"]),
            timestamp_s=float(value["timestamp_s"]) if value.get("timestamp_s") is not None else None,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "step": self.step,
            "features": self.features,
            "timestamp_s": self.timestamp_s,
        }


@dataclass(frozen=True)
class ActionChunk:
    actions: np.ndarray

    def __post_init__(self) -> None:
        if not np.issubdtype(self.actions.dtype, np.number):
            raise TypeError("actions must use a numeric dtype")
        if self.actions.ndim not in (1, 2):
            raise ValueError("actions must be one action vector or a two-dimensional action chunk")
        if self.actions.shape[0] == 0:
            raise ValueError("actions cannot be empty")

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> ActionChunk:
        return cls(actions=np.asarray(value["actions"]))

    def to_dict(self) -> dict[str, Any]:
        return {"actions": self.actions}


@dataclass(frozen=True)
class PolicyCapabilities:
    supported_benchmarks: tuple[str, ...]
    required_features: tuple[str, ...]
    supported_suites: dict[str, tuple[str, ...]] = field(default_factory=dict)
    max_batch_size: int = 1
    supported_action_chunk_sizes: tuple[int, ...] = (1,)
    action_representations: dict[str, str] = field(default_factory=dict)
    observation_strides: dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.supported_benchmarks:
            raise ValueError("supported_benchmarks cannot be empty")
        if len(self.supported_benchmarks) != len(set(self.supported_benchmarks)):
            raise ValueError("supported_benchmarks must be unique")
        if len(self.required_features) != len(set(self.required_features)):
            raise ValueError("required_features must be unique")
        unsupported_suite_keys = set(self.supported_suites) - set(self.supported_benchmarks)
        if unsupported_suite_keys:
            names = ", ".join(sorted(unsupported_suite_keys))
            raise ValueError(f"supported_suites contains undeclared benchmarks: {names}")
        if self.max_batch_size < 1:
            raise ValueError("max_batch_size must be positive")
        if not self.supported_action_chunk_sizes:
            raise ValueError("supported_action_chunk_sizes cannot be empty")
        if any(size < 1 for size in self.supported_action_chunk_sizes):
            raise ValueError("supported_action_chunk_sizes must be positive")
        if tuple(sorted(set(self.supported_action_chunk_sizes))) != self.supported_action_chunk_sizes:
            raise ValueError("supported_action_chunk_sizes must be unique and increasing")
        unsupported_action_keys = set(self.action_representations) - set(self.supported_benchmarks)
        if unsupported_action_keys:
            names = ", ".join(sorted(unsupported_action_keys))
            raise ValueError(f"action_representations contains undeclared benchmarks: {names}")
        if any(not representation for representation in self.action_representations.values()):
            raise ValueError("action representation identifiers cannot be empty")
        if set(self.observation_strides) - set(self.supported_benchmarks):
            raise ValueError("observation_strides contains an unsupported benchmark")
        if any(stride < 1 for stride in self.observation_strides.values()):
            raise ValueError("observation strides must be positive")

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> PolicyCapabilities:
        return cls(
            supported_benchmarks=tuple(str(item) for item in value["supported_benchmarks"]),
            required_features=tuple(str(item) for item in value["required_features"]),
            supported_suites={
                str(name): tuple(str(suite) for suite in suites)
                for name, suites in (value.get("supported_suites") or {}).items()
            },
            max_batch_size=int(value.get("max_batch_size", 1)),
            supported_action_chunk_sizes=tuple(
                int(size) for size in value.get("supported_action_chunk_sizes", (1,))
            ),
            action_representations={
                str(name): str(representation)
                for name, representation in (value.get("action_representations") or {}).items()
            },
            observation_strides={
                str(name): int(stride)
                for name, stride in (value.get("observation_strides") or {}).items()
            },
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "supported_benchmarks": list(self.supported_benchmarks),
            "required_features": list(self.required_features),
            "supported_suites": {
                benchmark: list(suites) for benchmark, suites in self.supported_suites.items()
            },
            "max_batch_size": self.max_batch_size,
            "supported_action_chunk_sizes": list(self.supported_action_chunk_sizes),
            "action_representations": dict(self.action_representations),
            "observation_strides": dict(self.observation_strides),
        }


@dataclass(frozen=True)
class PolicyExecutionContract:
    benchmark: str
    action_representation: str
    action_representation_source: str = "benchmark_default"
    observation_stride: int = 1
    observation_stride_source: str = "safe_default"

    def to_dict(self) -> dict[str, Any]:
        return {
            "benchmark": self.benchmark,
            "action_representation": self.action_representation,
            "action_representation_source": self.action_representation_source,
            "observation_stride": self.observation_stride,
            "observation_stride_source": self.observation_stride_source,
        }


def negotiate_policy_action_contract(
    benchmark: BenchmarkSpec,
    capabilities: PolicyCapabilities,
) -> PolicyExecutionContract:
    action_source = "policy"
    action_representation = capabilities.action_representations.get(benchmark.name)
    if action_representation is None:
        action_source = "benchmark_default"
        action_representation = benchmark.default_action_representation
    if action_representation not in benchmark.supported_action_representations:
        supported = ", ".join(benchmark.supported_action_representations)
        raise ValueError(
            f"policy action representation {action_representation!r} is unsupported by "
            f"{benchmark.name}; expected one of {supported}"
        )
    return PolicyExecutionContract(
        benchmark=benchmark.name,
        action_representation=action_representation,
        action_representation_source=action_source,
        observation_stride=capabilities.observation_strides.get(benchmark.name, 1),
        observation_stride_source=(
            "policy" if benchmark.name in capabilities.observation_strides else "safe_default"
        ),
    )


def validate_policy_capabilities(
    benchmark: BenchmarkSpec,
    capabilities: PolicyCapabilities,
) -> None:
    if benchmark.name not in capabilities.supported_benchmarks:
        raise ValueError(f"policy does not support benchmark {benchmark.name!r}")
    supported_suites = capabilities.supported_suites.get(benchmark.name)
    if supported_suites is not None and benchmark.suite not in supported_suites:
        raise ValueError(
            f"policy does not support suite {benchmark.suite!r} for benchmark {benchmark.name!r}"
        )
    available = {feature.name for feature in benchmark.features}
    unknown = sorted(set(capabilities.required_features) - available)
    if unknown:
        raise ValueError(f"policy requires features not provided by {benchmark.name}: {', '.join(unknown)}")
    negotiate_policy_action_contract(benchmark, capabilities)


def validate_observation(
    observation: Observation,
    benchmark: BenchmarkSpec,
    capabilities: PolicyCapabilities,
) -> None:
    specs = {feature.name: feature for feature in benchmark.features}
    unknown = sorted(set(observation.features) - set(specs))
    if unknown:
        raise ValueError(f"observation contains undeclared benchmark features: {', '.join(unknown)}")
    missing_benchmark_features = sorted(
        feature.name
        for feature in benchmark.features
        if feature.required and feature.name not in observation.features
    )
    if missing_benchmark_features:
        raise ValueError(
            f"observation is missing required benchmark features: {', '.join(missing_benchmark_features)}"
        )
    missing = sorted(set(capabilities.required_features) - set(observation.features))
    if missing:
        raise ValueError(f"observation is missing required policy features: {', '.join(missing)}")
    for name, value in observation.features.items():
        spec = specs[name]
        if spec.kind is FeatureKind.LANGUAGE:
            if not isinstance(value, str):
                raise TypeError(f"observation feature {name!r} must be a string")
            continue
        if not isinstance(value, np.ndarray):
            raise TypeError(f"observation feature {name!r} must be a numpy array")
        if spec.dtype is not None and value.dtype.name != spec.dtype:
            raise TypeError(
                f"observation feature {name!r} has dtype {value.dtype.name}; expected {spec.dtype}"
            )
        if spec.shape is not None and (
            value.ndim != len(spec.shape)
            or any(
                expected is not None and actual != expected
                for actual, expected in zip(value.shape, spec.shape, strict=True)
            )
        ):
            raise ValueError(f"observation feature {name!r} has shape {value.shape}; expected {spec.shape}")


def validate_action_chunk(action: ActionChunk, benchmark: BenchmarkSpec) -> None:
    if action.actions.dtype.name != benchmark.action.dtype:
        raise TypeError(
            f"policy action dtype {action.actions.dtype.name} does not match "
            f"{benchmark.name} dtype {benchmark.action.dtype}"
        )
    if not np.all(np.isfinite(action.actions)):
        raise ValueError("policy actions must contain only finite values")
    expected = benchmark.action.dimension
    if expected is None:
        return
    actual = action.actions.shape[-1]
    if actual != expected:
        raise ValueError(
            f"policy action dimension {actual} does not match {benchmark.name} dimension {expected}"
        )


class Policy(ABC):
    """Adapter implemented by each user inference entrypoint."""

    @abstractmethod
    def load(self, context: SessionContext) -> PolicyCapabilities:
        """Load the model and declare the benchmark features it consumes."""

    def reset(self, context: EpisodeContext) -> None:
        """Prepare Policy state for a new episode."""
        return None

    @abstractmethod
    def act(self, observation: Observation) -> ActionChunk:
        """Produce one action or an action chunk for a canonical observation."""

    def act_batch(self, observations: tuple[Observation, ...]) -> tuple[ActionChunk, ...]:
        """Produce actions for an explicit batch; override to use native model batching."""
        return tuple(self.act(observation) for observation in observations)

    def end_episode(self, episode_id: str) -> None:
        """Release all mutable state owned by a completed episode."""
        return None

    def unload(self) -> None:
        """Perform graceful cleanup before the policy process exits."""
        return None

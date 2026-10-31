# Copyright 2026 RoboScope Team
# SPDX-License-Identifier: Apache-2.0

"""Benchmark-neutral data contracts shared by Policy clients and servers."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class FeatureKind(str, Enum):  # noqa: UP042 -- benchmark environments include Python 3.10
    IMAGE = "image"
    STATE = "state"
    LANGUAGE = "language"


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    kind: FeatureKind
    required: bool = True
    description: str = ""
    dtype: str | None = None
    shape: tuple[int | None, ...] | None = None

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("feature name cannot be empty")
        if self.dtype == "":
            raise ValueError("feature dtype cannot be empty")
        if self.shape is not None and any(
            dimension is not None and dimension < 1 for dimension in self.shape
        ):
            raise ValueError("feature dimensions must be positive or dynamic")

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> FeatureSpec:
        return cls(
            name=str(value["name"]),
            kind=FeatureKind(value["kind"]),
            required=bool(value.get("required", True)),
            description=str(value.get("description", "")),
            dtype=str(value["dtype"]) if value.get("dtype") is not None else None,
            shape=(
                tuple(int(dimension) if dimension is not None else None for dimension in value["shape"])
                if value.get("shape") is not None
                else None
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind.value,
            "required": self.required,
            "description": self.description,
            "dtype": self.dtype,
            "shape": list(self.shape) if self.shape is not None else None,
        }


@dataclass(frozen=True)
class ActionSpec:
    name: str
    description: str
    dimension: int | None = None
    dtype: str = "float32"

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("action name cannot be empty")
        if not self.dtype:
            raise ValueError("action dtype cannot be empty")
        if self.dimension is not None and self.dimension < 1:
            raise ValueError("action dimension must be positive")

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> ActionSpec:
        dimension = value.get("dimension")
        return cls(
            name=str(value["name"]),
            description=str(value["description"]),
            dimension=int(dimension) if dimension is not None else None,
            dtype=str(value.get("dtype", "float32")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "dimension": self.dimension,
            "dtype": self.dtype,
        }


@dataclass(frozen=True)
class BenchmarkSpec:
    name: str
    suite: str
    protocol_version: str
    features: tuple[FeatureSpec, ...]
    action: ActionSpec
    supported_action_representations: tuple[str, ...] = ()
    default_action_representation: str = ""

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("benchmark name cannot be empty")
        if not self.suite:
            raise ValueError("benchmark suite cannot be empty")
        if not self.protocol_version:
            raise ValueError("benchmark protocol version cannot be empty")
        if not self.features:
            raise ValueError("benchmark features cannot be empty")
        names = [feature.name for feature in self.features]
        if len(names) != len(set(names)):
            raise ValueError("benchmark feature names must be unique")
        if not self.supported_action_representations:
            raise ValueError("benchmark action representations cannot be empty")
        if len(self.supported_action_representations) != len(
            set(self.supported_action_representations)
        ):
            raise ValueError("benchmark action representations must be unique")
        if self.default_action_representation not in self.supported_action_representations:
            raise ValueError("benchmark default action representation must be supported")

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> BenchmarkSpec:
        representations = tuple(
            str(item) for item in value.get("supported_action_representations", ())
        )
        default_representation = str(value.get("default_action_representation", ""))
        if not representations:
            default_representation = _legacy_action_representation(str(value["name"]))
            representations = (default_representation,)
        return cls(
            name=str(value["name"]),
            suite=str(value["suite"]),
            protocol_version=str(value["protocol_version"]),
            features=tuple(FeatureSpec.from_dict(item) for item in value["features"]),
            action=ActionSpec.from_dict(value["action"]),
            supported_action_representations=representations,
            default_action_representation=default_representation,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "suite": self.suite,
            "protocol_version": self.protocol_version,
            "features": [feature.to_dict() for feature in self.features],
            "action": self.action.to_dict(),
            "supported_action_representations": list(self.supported_action_representations),
            "default_action_representation": self.default_action_representation,
        }


def _legacy_action_representation(benchmark: str) -> str:
    defaults = {
        "libero": "libero.delta-eef-v1",
        "libero_plus": "libero.delta-eef-v1",
        "robocasa": "robocasa.gr1-joint-v1",
        "robotwin": "robotwin.qpos-v1",
        "calvin": "calvin.relative-v1",
    }
    try:
        return defaults[benchmark]
    except KeyError as exc:
        raise ValueError(
            f"legacy benchmark payload {benchmark!r} has no action representation default"
        ) from exc


__all__ = ["ActionSpec", "BenchmarkSpec", "FeatureKind", "FeatureSpec"]

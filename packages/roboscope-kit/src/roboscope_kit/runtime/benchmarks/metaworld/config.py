# Copyright 2026 RoboScope Team
# SPDX-License-Identifier: Apache-2.0

"""Single parse/validate/resolve path for the MetaWorld ``evaluation_config``.

The resolved object has the same shape as ``benchmark_config.metaworld`` in the
Policy JSON: ``{"cameras": [...]}`` with every per-camera field explicit. The
platform default is one ``images.primary`` corner2 camera, rendered and emitted
at 480x480 with a 180-degree rotation and no crop. Resolution is idempotent, so
a file holding an already-resolved object parses to the same value.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from roboscope_kit.benchmark_specs import METAWORLD_DEFAULT_CAMERA

CAMERA_FEATURES = ("images.primary", "images.secondary", "images.tertiary", "images.quaternary")
ROTATION_DEGREES = (0, 90, 180, 270)
MIN_CAMERA_DIMENSION = 64
MAX_CAMERA_DIMENSION = 1024

DEFAULT_CAMERA: dict[str, Any] = METAWORLD_DEFAULT_CAMERA

_CAMERA_FIELDS = frozenset(DEFAULT_CAMERA)
_CONFIG_FIELDS = frozenset({"cameras"})


def default_evaluation_config() -> dict[str, Any]:
    return {"cameras": [dict(DEFAULT_CAMERA)]}


def load_evaluation_config(path: Path | None) -> dict[str, Any]:
    """Read an evaluation_config JSON file and resolve it; ``None`` means platform default."""

    if path is None:
        return default_evaluation_config()
    try:
        raw = json.loads(Path(path).read_bytes().decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"metaworld evaluation_config file is not readable JSON: {path}") from exc
    return resolve_evaluation_config(raw)


def resolve_evaluation_config(raw: Any) -> dict[str, Any]:
    """Fill platform defaults and validate one ``benchmark_config.metaworld`` object.

    ``None`` selects the platform default. An explicit object must contain a
    non-empty ``cameras`` list; ``{}`` and unknown fields are rejected.
    """

    if raw is None:
        return default_evaluation_config()
    if not isinstance(raw, dict):
        raise ValueError("metaworld evaluation_config must be a JSON object")
    unknown = sorted(set(raw) - _CONFIG_FIELDS)
    if unknown:
        raise ValueError(f"metaworld evaluation_config has unknown fields: {', '.join(unknown)}")
    cameras = raw.get("cameras")
    if not isinstance(cameras, list) or not cameras:
        raise ValueError("metaworld evaluation_config cameras must be a non-empty list")
    if len(cameras) > len(CAMERA_FEATURES):
        raise ValueError(f"metaworld evaluation_config supports at most {len(CAMERA_FEATURES)} cameras")
    resolved = [_resolve_camera(camera, index) for index, camera in enumerate(cameras)]
    names = [camera["camera_name"] for camera in resolved]
    if len(names) != len(set(names)):
        raise ValueError("metaworld evaluation_config camera_name values must be unique")
    return {"cameras": resolved}


def _resolve_camera(value: Any, index: int) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"metaworld camera {index} must be a JSON object")
    unknown = sorted(set(value) - _CAMERA_FIELDS)
    if unknown:
        raise ValueError(f"metaworld camera {index} has unknown fields: {', '.join(unknown)}")
    for required in ("feature", "camera_name"):
        if required not in value:
            raise ValueError(f"metaworld camera {index} requires {required!r}")
    camera = dict(DEFAULT_CAMERA)
    camera.update(value)
    camera["feature"] = _validate_feature(camera["feature"], index)
    camera["camera_name"] = _validate_camera_name(camera["camera_name"], index)
    for field in ("render_width", "render_height", "output_width", "output_height"):
        camera[field] = _validate_dimension(camera[field], field, index)
    camera["rotation_deg"] = _validate_rotation(camera["rotation_deg"], index)
    camera["center_crop_ratio"] = _validate_crop_ratio(camera["center_crop_ratio"], index)
    camera["camera_position"] = _validate_position(camera["camera_position"], index)
    return camera


def _validate_feature(value: Any, index: int) -> str:
    expected = CAMERA_FEATURES[index]
    if value != expected:
        raise ValueError(
            f"metaworld camera {index} feature must be {expected!r} "
            f"(features are primary/secondary/tertiary/quaternary in order); received {value!r}"
        )
    return str(value)


def _validate_camera_name(value: Any, index: int) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"metaworld camera {index} camera_name must be a non-empty string")
    return value


def _validate_dimension(value: Any, field: str, index: int) -> int:
    if type(value) is not int or not MIN_CAMERA_DIMENSION <= value <= MAX_CAMERA_DIMENSION:
        raise ValueError(
            f"metaworld camera {index} {field} must be an integer in "
            f"[{MIN_CAMERA_DIMENSION}, {MAX_CAMERA_DIMENSION}]; received {value!r}"
        )
    return value


def _validate_rotation(value: Any, index: int) -> int:
    if type(value) is not int or value not in ROTATION_DEGREES:
        raise ValueError(
            f"metaworld camera {index} rotation_deg must be one of {ROTATION_DEGREES} "
            f"(counter-clockwise); received {value!r}"
        )
    return value


def _validate_crop_ratio(value: Any, index: int) -> float:
    if type(value) not in (int, float) or not math.isfinite(value) or not 0.0 < value <= 1.0:
        raise ValueError(
            f"metaworld camera {index} center_crop_ratio must be a finite number in (0, 1]; "
            f"received {value!r}"
        )
    return float(value)


def _validate_position(value: Any, index: int) -> list[float] | None:
    if value is None:
        return None
    if (
        not isinstance(value, list)
        or len(value) != 3
        or any(type(component) not in (int, float) or not math.isfinite(component) for component in value)
    ):
        raise ValueError(
            f"metaworld camera {index} camera_position must be null or a finite [x, y, z] list; "
            f"received {value!r}"
        )
    return [float(component) for component in value]


__all__ = [
    "CAMERA_FEATURES",
    "DEFAULT_CAMERA",
    "MAX_CAMERA_DIMENSION",
    "MIN_CAMERA_DIMENSION",
    "ROTATION_DEGREES",
    "default_evaluation_config",
    "load_evaluation_config",
    "resolve_evaluation_config",
]

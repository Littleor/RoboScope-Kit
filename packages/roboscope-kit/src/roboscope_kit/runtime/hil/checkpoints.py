# Copyright 2026 RoboScope Team
# SPDX-License-Identifier: Apache-2.0

"""Allowlisted local checkpoint discovery and immutable provenance."""

from __future__ import annotations

import hashlib
import json
import os
import struct
import threading
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SUPPORTED_POLICY_TYPES = frozenset({"pi05"})
REQUIRED_FILES = ("config.json", "policy_preprocessor.json", "policy_postprocessor.json")
# The HIL loader replaces checkpoint-local pretrained_path and tokenizer_name
# with the verified checkpoint and configured tokenizer before loading policy
# weights or processors. Only adapter base-model references remain external.
DEPENDENCY_KEYS = frozenset({"base_model_name_or_path"})


@dataclass(frozen=True)
class CheckpointRoot:
    name: str
    path: Path

    @classmethod
    def parse(cls, value: str) -> CheckpointRoot:
        name, separator, raw_path = value.partition("=")
        if not separator or not name or not raw_path:
            raise ValueError("checkpoint roots must use NAME=/absolute/path")
        path = Path(raw_path).expanduser().resolve(strict=True)
        if not path.is_dir() or not path.is_absolute():
            raise ValueError(f"checkpoint root is not an absolute directory: {path}")
        return cls(name=name, path=path)


@dataclass(frozen=True)
class CheckpointEntry:
    catalog_id: str
    root_name: str
    relative_path: str
    path: Path
    policy_type: str
    checkpoint_kind: str
    step: int | None
    training_dataset: str | None
    size_bytes: int
    file_count: int
    modified_at: str
    compatibility: str
    reasons: tuple[str, ...]
    input_features: dict[str, Any]
    output_features: dict[str, Any]
    action_feature_names: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "catalog_id": self.catalog_id,
            "root_name": self.root_name,
            "relative_path": self.relative_path,
            "policy_type": self.policy_type,
            "checkpoint_kind": self.checkpoint_kind,
            "step": self.step,
            "training_dataset": self.training_dataset,
            "size_bytes": self.size_bytes,
            "file_count": self.file_count,
            "modified_at": self.modified_at,
            "compatibility": self.compatibility,
            "reasons": list(self.reasons),
            "input_features": self.input_features,
            "output_features": self.output_features,
            "action_feature_names": list(self.action_feature_names),
        }


@dataclass(frozen=True)
class VerifiedCheckpoint:
    entry: CheckpointEntry
    checkpoint_sha256: str
    checkpoint_tree_sha256: str
    dependency_sha256: tuple[str, ...]
    policy_config_sha256: str
    preprocessor_sha256: str
    postprocessor_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.entry.to_dict(),
            "checkpoint_sha256": self.checkpoint_sha256,
            "checkpoint_tree_sha256": self.checkpoint_tree_sha256,
            "dependency_sha256": list(self.dependency_sha256),
            "policy_config_sha256": self.policy_config_sha256,
            "preprocessor_sha256": self.preprocessor_sha256,
            "postprocessor_sha256": self.postprocessor_sha256,
        }


class CheckpointCatalog:
    def __init__(
        self,
        roots: tuple[CheckpointRoot, ...],
        dependency_roots: tuple[Path, ...],
        hash_cache_path: Path,
    ) -> None:
        if not roots:
            raise ValueError("at least one checkpoint root is required")
        root_names = [root.name for root in roots]
        if len(root_names) != len(set(root_names)):
            raise ValueError("checkpoint root names must be unique")
        self.roots = roots
        self.dependency_roots = tuple(path.expanduser().resolve(strict=True) for path in dependency_roots)
        self.hash_cache = FileHashCache(hash_cache_path)
        self._verify_lock = threading.Lock()

    def scan(self) -> tuple[CheckpointEntry, ...]:
        entries = [
            self._entry(root, path.parent) for root in self.roots for path in root.path.rglob("config.json")
        ]
        return tuple(sorted(entries, key=lambda item: (item.root_name, item.relative_path)))

    def get(self, catalog_id: str) -> CheckpointEntry:
        matches = [entry for entry in self.scan() if entry.catalog_id == catalog_id]
        if len(matches) != 1:
            raise ValueError(f"unknown local checkpoint catalog ID {catalog_id!r}")
        return matches[0]

    def verify(self, catalog_id: str) -> VerifiedCheckpoint:
        with self._verify_lock:
            return self._verify(catalog_id)

    def _verify(self, catalog_id: str) -> VerifiedCheckpoint:
        entry = self.get(catalog_id)
        if entry.compatibility == "blocked":
            raise ValueError("checkpoint is blocked: " + "; ".join(entry.reasons))
        dependencies = self._dependencies(entry.path)
        checkpoint_tree_sha = self.hash_cache.directory_hash(entry.path)
        dependency_sha = tuple(sorted(self.hash_cache.directory_hash(path) for path in dependencies))
        aggregate = hashlib.sha256()
        aggregate.update(checkpoint_tree_sha.encode())
        for digest in dependency_sha:
            aggregate.update(digest.encode())
        self.hash_cache.save()
        return VerifiedCheckpoint(
            entry=entry,
            checkpoint_sha256=aggregate.hexdigest(),
            checkpoint_tree_sha256=checkpoint_tree_sha,
            dependency_sha256=dependency_sha,
            policy_config_sha256=file_sha256(entry.path / "config.json"),
            preprocessor_sha256=file_sha256(entry.path / "policy_preprocessor.json"),
            postprocessor_sha256=file_sha256(entry.path / "policy_postprocessor.json"),
        )

    def compatibility(
        self,
        entry: CheckpointEntry,
        observation_schema: dict[str, Any],
        action_schema: dict[str, Any],
        camera_key_mapping: dict[str, str],
        target_fps: int,
    ) -> tuple[str, tuple[str, ...]]:
        reasons = list(entry.reasons)
        blocked = entry.compatibility == "blocked"
        blocked |= _validate_state(entry, observation_schema, reasons)
        blocked |= _validate_action(entry, action_schema, reasons)
        blocked |= _validate_images(entry, observation_schema, camera_key_mapping, reasons)
        if target_fps <= 0:
            reasons.append("target FPS must be positive")
            blocked = True
        else:
            reasons.append("checkpoint does not declare training FPS")
        return ("blocked" if blocked else "warning" if reasons else "compatible", tuple(reasons))

    def _entry(self, root: CheckpointRoot, path: Path) -> CheckpointEntry:
        resolved = path.resolve(strict=True)
        if not resolved.is_relative_to(root.path):
            raise ValueError(f"checkpoint path escapes allowlisted root: {resolved}")
        config = _read_json(resolved / "config.json")
        train_config = _optional_json(resolved / "train_config.json")
        files = _regular_files(resolved)
        reasons = _base_reasons(resolved, config, files)
        policy_type = str(config.get("type") or "")
        relative_path = resolved.relative_to(root.path).as_posix()
        catalog_id = hashlib.sha256(f"{root.name}\0{relative_path}".encode()).hexdigest()[:32]
        return CheckpointEntry(
            catalog_id=catalog_id,
            root_name=root.name,
            relative_path=relative_path,
            path=resolved,
            policy_type=policy_type,
            checkpoint_kind="lora" if (resolved / "adapter_config.json").is_file() else "full",
            step=_checkpoint_step(resolved),
            training_dataset=_training_dataset(train_config),
            size_bytes=sum(file.stat().st_size for file in files),
            file_count=len(files),
            modified_at=_modified_at(files),
            compatibility="blocked" if reasons else "compatible",
            reasons=tuple(reasons),
            input_features=dict(config.get("input_features") or {}),
            output_features=dict(config.get("output_features") or {}),
            action_feature_names=tuple(str(name) for name in config.get("action_feature_names") or ()),
        )

    def _dependencies(self, checkpoint: Path) -> tuple[Path, ...]:
        raw_paths: set[str] = set()
        for name in ("config.json", "adapter_config.json", "policy_preprocessor.json"):
            value = _optional_json(checkpoint / name)
            _collect_dependencies(value, raw_paths)
        paths = tuple(sorted((Path(value).expanduser().resolve(strict=True) for value in raw_paths), key=str))
        for path in paths:
            if not path.is_dir() or not any(path.is_relative_to(root) for root in self.dependency_roots):
                raise ValueError(f"checkpoint dependency is outside allowlisted model roots: {path}")
        return paths


class FileHashCache:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.values = _optional_json(path)
        if not isinstance(self.values, dict):
            self.values = {}

    def directory_hash(self, root: Path) -> str:
        files = _regular_files(root)
        digest = hashlib.sha256()
        for path in files:
            relative = path.relative_to(root).as_posix()
            stat = path.stat()
            cached = self.values.get(str(path))
            signature = [stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns]
            file_digest = None
            if isinstance(cached, dict) and cached.get("stat") == signature:
                file_digest = cached.get("sha256")
            if not file_digest:
                file_digest = file_sha256(path)
                self.values[str(path)] = {"stat": signature, "sha256": file_digest}
            digest.update(relative.encode())
            digest.update(str(stat.st_size).encode())
            digest.update(file_digest.encode())
        return digest.hexdigest()

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_name(f".{self.path.name}.{os.getpid()}.tmp")
        temporary.write_text(json.dumps(self.values, sort_keys=True, separators=(",", ":")))
        os.replace(temporary, self.path)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _regular_files(root: Path) -> tuple[Path, ...]:
    files: list[Path] = []
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError(f"checkpoint trees must not contain symlinks: {path}")
        if path.is_file():
            files.append(path)
    return tuple(sorted(files))


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _optional_json(path: Path) -> dict[str, Any]:
    return _read_json(path) if path.is_file() else {}


def _base_reasons(path: Path, config: dict[str, Any], files: tuple[Path, ...]) -> list[str]:
    reasons = [f"missing required file {name}" for name in REQUIRED_FILES if not (path / name).is_file()]
    policy_type = str(config.get("type") or "")
    if policy_type not in SUPPORTED_POLICY_TYPES:
        reasons.append(f"unsupported policy type {policy_type!r}")
    if not any(file.suffix == ".safetensors" for file in files):
        reasons.append("checkpoint contains no safetensors weights")
    return reasons


def _checkpoint_step(path: Path) -> int | None:
    for parent in (path, *path.parents):
        if parent.name.isdigit():
            return int(parent.name)
        if parent.name == "checkpoints":
            return None
    return None


def _training_dataset(config: dict[str, Any]) -> str | None:
    dataset = config.get("dataset")
    if not isinstance(dataset, dict):
        return None
    repo_id = dataset.get("repo_id")
    return str(repo_id) if repo_id else None


def _modified_at(files: tuple[Path, ...]) -> str:
    timestamp = max(file.stat().st_mtime for file in files)
    return datetime.fromtimestamp(timestamp, tz=UTC).isoformat()


def _collect_dependencies(value: Any, output: set[str], key: str = "") -> None:
    if isinstance(value, dict):
        for child_key, child in value.items():
            _collect_dependencies(child, output, str(child_key))
        return
    if isinstance(value, list):
        for child in value:
            _collect_dependencies(child, output, key)
        return
    if key in DEPENDENCY_KEYS and isinstance(value, str) and Path(value).is_absolute():
        output.add(value)


def _shape(value: Any) -> tuple[int, ...]:
    if not isinstance(value, dict) or not isinstance(value.get("shape"), list):
        return ()
    return tuple(int(dimension) for dimension in value["shape"])


def _validate_state(entry: CheckpointEntry, schema: dict[str, Any], reasons: list[str]) -> bool:
    expected = _normalizer_state_shape(entry.path)
    actual = _shape(schema.get("state"))
    blocked = expected != actual or not expected
    if blocked:
        reasons.append(f"state shape {actual} does not match checkpoint {expected}")
    blocked |= _validate_dtype(schema.get("state"), "float32", "state", reasons)
    blocked |= _validate_feature_names(entry, schema.get("state"), "state", reasons)
    return blocked


def _validate_action(entry: CheckpointEntry, schema: dict[str, Any], reasons: list[str]) -> bool:
    expected = _shape(entry.output_features.get("action"))
    actual = _shape(schema)
    blocked = expected != actual or not expected
    if blocked:
        reasons.append(f"action shape {actual} does not match checkpoint {expected}")
    blocked |= _validate_dtype(schema, "float32", "action", reasons)
    blocked |= _validate_feature_names(entry, schema, "action", reasons)
    return blocked


def _validate_images(
    entry: CheckpointEntry,
    schema: dict[str, Any],
    mapping: dict[str, str],
    reasons: list[str],
) -> bool:
    images = schema.get("images")
    if not isinstance(images, dict):
        reasons.append("observation schema does not declare images")
        return True
    raw_keys = {_raw_camera_key(source) for source in images}
    if set(mapping) != raw_keys:
        reasons.append("camera mapping keys must exactly match raw camera keys")
        return True
    mapped = {mapping[_raw_camera_key(source)]: value for source, value in images.items()}
    expected = {name: value for name, value in entry.input_features.items() if value.get("type") == "VISUAL"}
    missing = sorted(set(expected) - set(mapped))
    if missing:
        reasons.append("missing policy camera keys: " + ", ".join(missing))
        return True
    unexpected = sorted(set(mapped) - set(expected))
    if unexpected:
        reasons.append("unexpected policy camera keys: " + ", ".join(unexpected))
        return True
    for name in expected:
        actual_hwc = _shape(mapped[name])
        if _validate_dtype(mapped[name], "uint8", name, reasons):
            return True
        if len(actual_hwc) != 3 or actual_hwc[-1:] != (3,) or any(dimension <= 0 for dimension in actual_hwc):
            reasons.append(f"camera {name} must be a non-empty HWC RGB image, received {actual_hwc}")
            return True
    return False


def _normalizer_state_shape(checkpoint: Path) -> tuple[int, ...]:
    path = checkpoint / "policy_preprocessor_step_2_normalizer_processor.safetensors"
    with path.open("rb") as stream:
        header_size_bytes = stream.read(8)
        if len(header_size_bytes) != 8:
            raise ValueError(f"invalid safetensors header: {path}")
        (header_size,) = struct.unpack("<Q", header_size_bytes)
        if header_size <= 0 or header_size > 16 * 1024 * 1024:
            raise ValueError(f"invalid safetensors header size {header_size}: {path}")
        header = json.loads(stream.read(header_size))
    shapes = []
    for key in ("observation.state.q01", "observation.state.q99"):
        tensor = header.get(key)
        shape = tensor.get("shape") if isinstance(tensor, dict) else None
        if not isinstance(shape, list) or not shape:
            raise ValueError(f"normalizer stats do not declare {key}: {path}")
        shapes.append(tuple(int(dimension) for dimension in shape))
    if shapes[0] != shapes[1] or any(dimension <= 0 for dimension in shapes[0]):
        raise ValueError(f"normalizer state quantiles have inconsistent shapes: {path}")
    return shapes[0]


def _raw_camera_key(observation_key: str) -> str:
    prefix = "observation.images."
    if not observation_key.startswith(prefix) or len(observation_key) == len(prefix):
        raise ValueError(f"invalid image observation key {observation_key!r}")
    return observation_key[len(prefix) :]


def _validate_dtype(value: Any, expected: str, name: str, reasons: list[str]) -> bool:
    actual = value.get("dtype") if isinstance(value, dict) else None
    if actual == expected:
        return False
    reasons.append(f"{name} dtype {actual!r} does not match checkpoint input {expected!r}")
    return True


def _validate_feature_names(
    entry: CheckpointEntry,
    value: Any,
    feature: str,
    reasons: list[str],
) -> bool:
    actual = tuple(str(name) for name in value.get("names") or ()) if isinstance(value, dict) else ()
    if entry.action_feature_names and actual == entry.action_feature_names:
        return False
    reasons.append(f"{feature} feature names do not match checkpoint action ordering")
    return True

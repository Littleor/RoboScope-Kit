# Copyright 2026 RoboScope Team
# SPDX-License-Identifier: Apache-2.0

"""Load the immutable user Policy process contract from an evaluation image."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

POLICY_CONFIG_SCHEMA_VERSION = 2
POLICY_CONFIG_PATH = Path("/workspace/roboscope-policy.json")
MAX_POLICY_ARGUMENTS = 32
MAX_POLICY_ARGUMENT_LENGTH = 1024
MAX_POLICY_CWD_LENGTH = 1024

_POLICY_CONFIG_FIELDS = frozenset({"schema_version", "command", "cwd", "benchmark_config"})
_POLICY_CONFIG_REQUIRED_FIELDS = frozenset({"schema_version", "command", "cwd"})


class PolicyConfigError(ValueError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass(frozen=True)
class PolicyRuntimeConfig:
    path: Path
    command: tuple[str, ...]
    cwd: Path
    benchmark_config: dict[str, Any]

    def summary(self) -> dict[str, Any]:
        return {
            "schema_version": POLICY_CONFIG_SCHEMA_VERSION,
            "path": str(self.path),
            "command": list(self.command),
            "cwd": str(self.cwd),
            "benchmark_config": self.benchmark_config,
        }


def load_policy_runtime_config() -> PolicyRuntimeConfig:
    config_path = POLICY_CONFIG_PATH
    if not config_path.is_file():
        raise PolicyConfigError("missing_policy_config", f"Policy config not found: {config_path}")
    raw = config_path.read_bytes()
    value = _parse_config(raw, config_path)
    command = _validate_command(value.get("command"))
    cwd = _validate_cwd(value.get("cwd"))
    benchmark_config = _validate_benchmark_config(value.get("benchmark_config", {}))
    executable = Path(command[0])
    if not executable.is_file() or not os.access(executable, os.X_OK):
        raise PolicyConfigError(
            "invalid_policy_executable",
            f"Policy executable is not runnable: {executable}",
        )
    if not cwd.is_dir():
        raise PolicyConfigError("invalid_policy_cwd", f"Policy working directory not found: {cwd}")
    return PolicyRuntimeConfig(
        path=config_path,
        command=command,
        cwd=cwd,
        benchmark_config=benchmark_config,
    )


def _parse_config(raw: bytes, path: Path) -> dict[str, Any]:
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PolicyConfigError(
            "invalid_policy_config",
            f"Policy config is not valid UTF-8 JSON: {path}",
        ) from exc
    if not isinstance(value, dict):
        raise PolicyConfigError("invalid_policy_config", "Policy config root must be a JSON object")
    if not set(value) >= _POLICY_CONFIG_REQUIRED_FIELDS or not set(value) <= _POLICY_CONFIG_FIELDS:
        raise PolicyConfigError(
            "invalid_policy_config",
            "Policy config fields must be schema_version, command, cwd, "
            "and optional benchmark_config",
        )
    if type(value["schema_version"]) is not int or value["schema_version"] != POLICY_CONFIG_SCHEMA_VERSION:
        raise PolicyConfigError(
            "unsupported_policy_config_version",
            f"Policy config schema_version must be {POLICY_CONFIG_SCHEMA_VERSION}",
        )
    return value


def _validate_benchmark_config(value: object) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise PolicyConfigError(
            "invalid_policy_config",
            "Policy benchmark_config must be a JSON object",
        )
    for name, config in value.items():
        if not isinstance(name, str) or not name.strip() or not isinstance(config, dict):
            raise PolicyConfigError(
                "invalid_policy_config",
                "Policy benchmark_config entries must be named JSON objects",
            )
    return {str(name): dict(config) for name, config in value.items()}


def _validate_command(value: object) -> tuple[str, ...]:
    if not isinstance(value, list) or not value or not all(isinstance(item, str) for item in value):
        raise PolicyConfigError("invalid_policy_command", "Policy command must be a non-empty string array")
    if len(value) > MAX_POLICY_ARGUMENTS:
        raise PolicyConfigError(
            "invalid_policy_command",
            f"Policy command supports at most {MAX_POLICY_ARGUMENTS} arguments",
        )
    if any(not item.strip() or "\0" in item or len(item) > MAX_POLICY_ARGUMENT_LENGTH for item in value):
        raise PolicyConfigError(
            "invalid_policy_command",
            f"Policy command arguments must contain 1-{MAX_POLICY_ARGUMENT_LENGTH} characters",
        )
    if not Path(value[0]).is_absolute():
        raise PolicyConfigError("invalid_policy_command", "Policy executable must use an absolute path")
    return tuple(value)


def _validate_cwd(value: object) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise PolicyConfigError("invalid_policy_cwd", "Policy cwd must be a non-empty string")
    if "\0" in value or len(value) > MAX_POLICY_CWD_LENGTH:
        raise PolicyConfigError(
            "invalid_policy_cwd",
            f"Policy cwd must not contain NUL and must contain at most {MAX_POLICY_CWD_LENGTH} characters",
        )
    cwd = Path(value)
    if not cwd.is_absolute():
        raise PolicyConfigError("invalid_policy_cwd", "Policy cwd must use an absolute path")
    return cwd

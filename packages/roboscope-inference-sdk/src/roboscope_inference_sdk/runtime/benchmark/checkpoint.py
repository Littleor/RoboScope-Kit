# Copyright 2026 RoboScope Team
# SPDX-License-Identifier: Apache-2.0

"""Materialize a benchmark checkpoint from an absolute path or OSS prefix."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import time
from collections.abc import Callable
from pathlib import Path

DEFAULT_CHECKPOINT_DIR = Path("/root/benchmark/checkpoint")
PROGRESS_PATTERN = re.compile(r"(\d+(?:\.\d+)?)%")
CheckpointProgressCallback = Callable[[int], None]


def materialize_checkpoint(
    source: str,
    target: Path = DEFAULT_CHECKPOINT_DIR,
    progress_callback: CheckpointProgressCallback | None = None,
) -> Path:
    if source.startswith("/"):
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"checkpoint path does not exist: {path}")
        _report_progress(progress_callback, 100)
        return path
    if not source.startswith("oss://"):
        raise ValueError("checkpoint source must be an absolute path or oss:// prefix")
    binary = shutil.which(os.environ.get("OSSUTIL_BIN", "ossutil")) or shutil.which("ossutil64")
    if binary is None:
        raise RuntimeError("ossutil or ossutil64 is required to download an OSS checkpoint")
    endpoint = _required_env("BENCHMARK_CKPT_ENDPOINT")
    access_key_id = _required_env("BENCHMARK_CKPT_ACCESS_KEY_ID")
    access_key_secret = _required_env("BENCHMARK_CKPT_ACCESS_KEY_SECRET")
    security_token = os.environ.get("BENCHMARK_CKPT_SECURITY_TOKEN", "").strip()
    checkpoint_dir = target.parent / ".ossutil-checkpoints"
    if target.exists():
        shutil.rmtree(target)
    if checkpoint_dir.exists():
        shutil.rmtree(checkpoint_dir)
    target.mkdir(parents=True)
    checkpoint_dir.mkdir()
    command = [
        binary,
        "cp",
        source,
        f"{target}/",
        "-r",
        "-e",
        endpoint,
        "-i",
        access_key_id,
        "-k",
        access_key_secret,
        f"--checkpoint-dir={checkpoint_dir}",
    ]
    if security_token:
        command.extend(("-t", security_token))
    try:
        _run_oss_copy(command, progress_callback)
    finally:
        shutil.rmtree(checkpoint_dir, ignore_errors=True)
    return target


def _run_oss_copy(
    command: list[str],
    progress_callback: CheckpointProgressCallback | None,
) -> None:
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=0,
    )
    output: list[str] = []
    chunk = ""
    last_progress = -1
    if process.stdout is None:
        raise RuntimeError("ossutil output stream is unavailable")
    while True:
        character = process.stdout.read(1)
        if character == "" and process.poll() is not None:
            break
        if character == "":
            time.sleep(0.1)
            continue
        chunk += character
        if character not in {"\r", "\n"}:
            continue
        output.append(chunk)
        match = PROGRESS_PATTERN.search(chunk)
        if match is not None:
            progress = max(0, min(100, int(round(float(match.group(1))))))
            if progress != last_progress:
                _report_progress(progress_callback, progress)
                last_progress = progress
        chunk = ""
        if len(output) > 200:
            output = output[-200:]
    return_code = process.wait()
    if chunk:
        output.append(chunk)
    if return_code != 0:
        raise RuntimeError(f"ossutil checkpoint download failed: {''.join(output)[-2000:]}")
    if last_progress != 100:
        _report_progress(progress_callback, 100)


def _report_progress(
    progress_callback: CheckpointProgressCallback | None,
    progress: int,
) -> None:
    if progress_callback is not None:
        progress_callback(progress)


def _required_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"{name} is required to download an OSS checkpoint")
    return value


__all__ = ["CheckpointProgressCallback", "materialize_checkpoint"]

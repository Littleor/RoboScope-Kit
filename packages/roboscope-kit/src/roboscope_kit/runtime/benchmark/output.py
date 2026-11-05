# Copyright 2026 RoboScope Team
# SPDX-License-Identifier: Apache-2.0

"""Exclusive output-directory ownership for benchmark CLI invocations."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def reserve_output_directory(output_dir: Path) -> Iterator[None]:
    """Allow an empty directory or an error-only retry, preserving all previous artifacts."""

    output_dir.mkdir(parents=True, exist_ok=True)
    lock_path = output_dir / ".run.lock"
    message = (
        f"output directory contains run artifacts or is already in use: {output_dir}. "
        "Choose a new --output-dir; only directories containing errors.jsonl alone can be retried."
    )
    try:
        lock_path.touch(exist_ok=False)
    except FileExistsError as exc:
        raise FileExistsError(message) from exc
    try:
        for path in output_dir.iterdir():
            if path.name == ".run.lock":
                continue
            if path.name == "errors.jsonl" and path.is_file() and not path.is_symlink():
                continue
            raise FileExistsError(message)
        yield
    finally:
        lock_path.unlink()

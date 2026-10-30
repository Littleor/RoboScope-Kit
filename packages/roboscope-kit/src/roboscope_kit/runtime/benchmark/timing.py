# Copyright 2026 RoboScope Team
# SPDX-License-Identifier: Apache-2.0

"""Benchmark phase timing recording owned by the SDK runtime."""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from pathlib import Path
from typing import Callable  # noqa: UP035 - this alias is evaluated by Python 3.8

PhaseTimingCallback = Callable[[str, str], None]
LOGGER = logging.getLogger(__name__)


class BenchmarkTimingRecorder:
    """Continuously publish the current benchmark timing snapshot."""

    FIELDS = (
        "checkpoint_prepare_s",
        "autotune_s",
        "policy_load_s",
        "runtime_prepare_s",
        "evaluation_s",
    )

    def __init__(self, path: Path):
        self.path = path
        self.values = {field: 0.0 for field in self.FIELDS}
        self.completed_fields: set[str] = set()
        self.active_field: str | None = None
        self.active_started = 0.0
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.write()
        self.thread = threading.Thread(target=self._heartbeat, name="benchmark-timing", daemon=True)
        self.thread.start()

    def start(self, field: str) -> None:
        self._require_field(field)
        with self.lock:
            if self.active_field is not None:
                raise RuntimeError(f"benchmark timing stage already active: {self.active_field}")
            self.completed_fields.discard(field)
            self.active_field = field
            self.active_started = time.monotonic()
        self.write()

    def pause(self, field: str) -> None:
        self._stop_active_field(field, completed=False)

    def finish(self, field: str) -> None:
        self._stop_active_field(field, completed=True)

    def _stop_active_field(self, field: str, *, completed: bool) -> None:
        self._require_field(field)
        with self.lock:
            if self.active_field != field:
                raise RuntimeError(f"benchmark timing stage is not active: {field}")
            self.values[field] += time.monotonic() - self.active_started
            if completed:
                self.completed_fields.add(field)
            self.active_field = None
            self.active_started = 0.0
        self.write()

    def handle_phase(self, field: str, event: str) -> None:
        if event == "start":
            self.start(field)
            return
        if event == "pause":
            self.pause(field)
            return
        if event == "finish":
            self.finish(field)
            return
        raise ValueError(f"unknown benchmark timing event: {event}")

    def write(self) -> None:
        with self.lock:
            values = dict(self.values)
            if self.active_field is not None:
                values[self.active_field] += time.monotonic() - self.active_started
            payload = {
                **values,
                "active_stage": self.active_field,
                "completed_stages": [field for field in self.FIELDS if field in self.completed_fields],
            }
            tmp = self.path.with_name(f"{self.path.name}.{os.getpid()}.tmp")
            try:
                tmp.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
                os.replace(tmp, self.path)
            except OSError as exc:
                LOGGER.warning("benchmark timing snapshot write failed: path=%s error=%s", self.path, exc)
                with suppress(OSError):
                    tmp.unlink(missing_ok=True)

    def close(self) -> None:
        self.stop.set()
        self.thread.join(timeout=2)
        self.write()

    def _heartbeat(self) -> None:
        while not self.stop.wait(1):
            self.write()

    def _require_field(self, field: str) -> None:
        if field not in self.values:
            raise ValueError(f"unknown benchmark timing field: {field}")


@contextmanager
def benchmark_phase(callback: PhaseTimingCallback | None, field: str) -> Iterator[None]:
    """Report one SDK-owned phase and leave failures active for interruption capture."""

    if callback is not None:
        callback(field, "start")
    try:
        yield
    except Exception:
        raise
    else:
        if callback is not None:
            callback(field, "finish")


__all__ = ["BenchmarkTimingRecorder", "PhaseTimingCallback", "benchmark_phase"]

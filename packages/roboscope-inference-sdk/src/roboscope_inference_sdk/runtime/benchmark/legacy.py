# Copyright 2026 RoboScope Team
# SPDX-License-Identifier: Apache-2.0

"""Adapter for the legacy ``startbenchmark.sh`` progress protocol."""

from __future__ import annotations

import os
import threading
from collections.abc import Callable, Sequence
from pathlib import Path

from .schema import WorkResult


class LegacyContract:
    """Own the Cloud-facing ``testlog.txt`` status file."""

    def __init__(self, root: Path, benchmark: str, suite: str) -> None:
        self.root = root
        self.benchmark = benchmark
        self.suite = suite
        self.result_dir = root / "testresult"
        self.video_dir = root / "testvideo"
        self.testlog_path = root / "testlog.txt"
        self.total = 0
        self.successes = 0
        self.failures = 0
        self.checkpoint_progress = 0
        self.suite_progress = 0
        self.status = "preparing"
        self._lock = threading.Lock()

    def reset(self) -> None:
        self.result_dir.mkdir(parents=True, exist_ok=True)
        self.video_dir.mkdir(parents=True, exist_ok=True)
        self.write_log(0, 0, self.suite, 0, 0, 0, "preparing")

    def set_total(self, total: int) -> None:
        if total < 1:
            raise ValueError("legacy contract total must be positive")
        self.total = total
        self.write_log(100, 100, self.suite, total, 0, 0, "running")

    def set_checkpoint_progress(self, progress: int) -> None:
        normalized = max(self.checkpoint_progress, max(0, min(100, progress)))
        self.write_log(
            normalized,
            self.suite_progress,
            self.suite,
            self.total,
            self.successes,
            self.failures,
            self.status,
        )

    def write_log(
        self,
        checkpoint_progress: int,
        suite_progress: int,
        suite: str,
        total: int,
        successes: int,
        failures: int,
        status: str,
    ) -> None:
        with self._lock:
            self.successes = successes
            self.failures = failures
            self.checkpoint_progress = checkpoint_progress
            self.suite_progress = suite_progress
            self.status = status
            temporary = self.testlog_path.with_name(
                f"{self.testlog_path.name}.{os.getpid()}.{threading.get_ident()}.tmp"
            )
            temporary.write_text(
                f"用户模型权重下载进度 {checkpoint_progress}%\n"
                f"测试套件下载进度 {suite_progress}%\n"
                f"{self.benchmark} {suite} {total} {successes} {failures} {status}\n",
                encoding="utf-8",
            )
            temporary.replace(self.testlog_path)

    def checkpoint_ready(self) -> None:
        self.write_log(100, 100, self.suite, self.total, self.successes, self.failures, "running")

    def fail(self, code: str) -> None:
        normalized = code.strip().replace(" ", "_") or "runtime_failure"
        self.write_log(
            self.checkpoint_progress,
            self.suite_progress,
            self.suite,
            self.total,
            self.successes,
            self.failures,
            f"error:{normalized}",
        )


class LegacyProgressAdapter:
    """Mirror generic terminal records to the legacy status callback."""

    def __init__(
        self,
        *,
        suite: str,
        total: int,
        write_log: Callable[..., None],
    ) -> None:
        self.suite = suite
        self.total = total
        self.write_log = write_log
        self.result_count = 0
        self.successes = 0
        self.failures = 0
        self._condition = threading.Condition()
        self._latest_results: tuple[WorkResult, ...] | None = None
        self._stopping = False
        self._error: BaseException | None = None
        self._thread = threading.Thread(
            target=self._run,
            name="legacy-progress-writer",
            daemon=True,
        )
        self._thread.start()

    def update(self, _progress: dict, results: Sequence[WorkResult]) -> None:
        with self._condition:
            self._latest_results = tuple(results)
            self._condition.notify()

    def _process(self, results: Sequence[WorkResult]) -> None:
        records = sorted(
            (record for result in results for record in result.records),
            key=lambda record: record.record_id,
        )
        record_ids = [record.record_id for record in records]
        if len(record_ids) != len(set(record_ids)):
            raise RuntimeError("legacy progress received duplicate record ids")
        self.result_count = len(records)
        self.successes = sum(record.success for record in records)
        self.failures = len(records) - self.successes
        self.write_log(
            100,
            100,
            self.suite,
            self.total,
            self.successes,
            self.failures,
            "running",
        )

    def finish(self) -> None:
        with self._condition:
            self._stopping = True
            self._condition.notify()
        self._thread.join()
        if self._error is not None:
            raise RuntimeError("legacy progress writer failed") from self._error
        if self.result_count != self.total:
            raise RuntimeError(
                f"legacy protocol expected {self.total} results, received {self.result_count}"
            )
        self.write_log(
            100,
            100,
            self.suite,
            self.total,
            self.successes,
            self.failures,
            "finish",
        )

    def abort(self) -> None:
        with self._condition:
            self._stopping = True
            self._condition.notify()
        self._thread.join()

    def _run(self) -> None:
        try:
            while True:
                results = self._next_results()
                if results is None:
                    return
                self._process(results)
        except BaseException as exc:
            self._error = exc

    def _next_results(self) -> tuple[WorkResult, ...] | None:
        with self._condition:
            while self._latest_results is None and not self._stopping:
                self._condition.wait()
            results = self._latest_results
            self._latest_results = None
            if results is None and self._stopping:
                return None
            return results

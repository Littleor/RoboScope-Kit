# Copyright 2026 RoboScope Team
# SPDX-License-Identifier: Apache-2.0

"""Worker-side loop for adapting arbitrary evaluation scripts."""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path

from .queue import WorkQueue, atomic_json
from .schema import WorkItem, WorkResult


class WorkerLoop:
    """Claim SDK work items and invoke user evaluation code for each item."""

    def __init__(
        self,
        *,
        worker_id: int,
        queue_dir: Path,
        state_dir: Path,
        poll_interval_s: float = 0.05,
        recycle_after_records: int | None = None,
    ) -> None:
        if worker_id < 0:
            raise ValueError("worker_id must be non-negative")
        if poll_interval_s <= 0:
            raise ValueError("poll_interval_s must be positive")
        if recycle_after_records is not None and recycle_after_records <= 0:
            raise ValueError("recycle_after_records must be positive")
        self.worker_id = worker_id
        self.queue = WorkQueue(queue_dir)
        self.state_dir = state_dir
        self.poll_interval_s = poll_interval_s
        self.recycle_after_records = recycle_after_records
        self.state_path = state_dir / f"worker-{worker_id:03d}.json"

    def run(self, evaluate: Callable[[WorkItem], WorkResult]) -> None:
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self._write_state("ready")
        self._wait_for_start()
        preferred_group: str | None = None
        completed_records = 0
        while True:
            claimed = self.queue.claim(self.worker_id, preferred_group)
            if claimed is None:
                if (self.queue.root / "stop").exists():
                    return
                time.sleep(self.poll_interval_s)
                continue
            self._write_state("running", item_id=claimed.item.item_id)
            result = evaluate(claimed.item)
            if result.worker_id != self.worker_id:
                raise ValueError("evaluate returned a result for a different worker")
            self.queue.complete(claimed, result)
            completed_records += len(result.records)
            preferred_group = claimed.item.group_id
            self._write_state("idle", completed_item_id=claimed.item.item_id)
            if self.recycle_after_records is not None and completed_records >= self.recycle_after_records:
                self._write_state("recycling", completed_records=completed_records)
                return

    def _wait_for_start(self) -> None:
        while not (self.queue.root / "start").exists():
            time.sleep(self.poll_interval_s)

    def _write_state(self, state: str, **values: object) -> None:
        atomic_json(
            self.state_path,
            {
                "state": state,
                "worker_id": self.worker_id,
                "heartbeat_unix_ns": time.time_ns(),
                **values,
            },
        )

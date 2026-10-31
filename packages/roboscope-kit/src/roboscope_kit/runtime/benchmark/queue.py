# Copyright 2026 RoboScope Team
# SPDX-License-Identifier: Apache-2.0

"""Atomic filesystem queue used by benchmark worker processes."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from .schema import WorkItem, WorkResult


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def item_filename(item: WorkItem) -> str:
    return f"item-{item.ordinal:06d}-{group_token(item.group_id)}.json"


def group_token(group_id: str) -> str:
    return hashlib.blake2s(group_id.encode("utf-8"), digest_size=6).hexdigest()


@dataclass(frozen=True)
class ClaimedWork:
    path: Path
    item: WorkItem


class WorkQueue:
    """Durable work queue with atomic claims and idempotent completions."""

    def __init__(self, root: Path):
        self.root = root

    @property
    def pending_dir(self) -> Path:
        return self.root / "pending"

    @property
    def running_dir(self) -> Path:
        return self.root / "running"

    @property
    def completed_dir(self) -> Path:
        return self.root / "completed"

    @property
    def partial_dir(self) -> Path:
        return self.root / "partial"

    def initialize(self, items: list[WorkItem]) -> None:
        ordinals = [item.ordinal for item in items]
        item_ids = [item.item_id for item in items]
        if len(ordinals) != len(set(ordinals)):
            raise ValueError("work-item ordinals must be unique")
        if len(item_ids) != len(set(item_ids)):
            raise ValueError("work-item ids must be unique")
        for directory in (self.pending_dir, self.running_dir, self.partial_dir, self.completed_dir):
            directory.mkdir(parents=True, exist_ok=True)
            for existing in directory.glob("*.json"):
                existing.unlink()
        (self.root / "start").unlink(missing_ok=True)
        (self.root / "stop").unlink(missing_ok=True)
        for item in items:
            atomic_json(self.pending_dir / item_filename(item), item.to_dict())

    def start(self) -> None:
        (self.root / "start").touch()

    def request_stop(self) -> None:
        (self.root / "stop").touch()

    def claim(self, worker_id: int, preferred_group: str | None = None) -> ClaimedWork | None:
        for pending_path in self._pending_candidates(preferred_group):
            claimed_path = self.running_dir / f"worker-{worker_id:03d}-{pending_path.name}"
            try:
                pending_path.replace(claimed_path)
            except FileNotFoundError:
                continue
            payload = json.loads(claimed_path.read_text(encoding="utf-8"))
            return ClaimedWork(claimed_path, WorkItem.from_dict(payload))
        return None

    def complete(self, claimed: ClaimedWork, result: WorkResult) -> None:
        self._validate_result(claimed.item, result)
        if len(result.records) != claimed.item.expected_results:
            raise ValueError(
                f"work item {claimed.item.item_id} expected {claimed.item.expected_results} "
                f"records, received {len(result.records)}"
            )
        completed_path = self.completed_dir / item_filename(claimed.item)
        if completed_path.exists():
            existing = WorkResult.from_dict(json.loads(completed_path.read_text(encoding="utf-8")))
            if existing != result:
                raise RuntimeError(f"conflicting completion for {claimed.item.item_id}")
        else:
            atomic_json(completed_path, result.to_dict())
        (self.partial_dir / item_filename(claimed.item)).unlink(missing_ok=True)
        claimed.path.unlink(missing_ok=True)

    def publish_partial(self, item: WorkItem, result: WorkResult) -> None:
        self._validate_result(item, result)
        if not 0 < len(result.records) <= item.expected_results:
            raise ValueError(
                f"partial work result for {item.item_id} must contain between 1 and "
                f"{item.expected_results} records"
            )
        partial_path = self.partial_dir / item_filename(item)
        if partial_path.exists():
            existing = WorkResult.from_dict(json.loads(partial_path.read_text(encoding="utf-8")))
            existing_count = len(existing.records)
            if existing.records != result.records[:existing_count]:
                raise RuntimeError(f"partial work result for {item.item_id} is not append-only")
            if existing.records == result.records:
                return
        atomic_json(partial_path, result.to_dict())

    def requeue_worker(self, worker_id: int) -> int:
        count = 0
        pattern = f"worker-{worker_id:03d}-item-*.json"
        for claimed_path in sorted(self.running_dir.glob(pattern)):
            payload = json.loads(claimed_path.read_text(encoding="utf-8"))
            item = WorkItem.from_dict(payload)
            if (self.completed_dir / item_filename(item)).exists():
                (self.partial_dir / item_filename(item)).unlink(missing_ok=True)
                claimed_path.unlink()
                continue
            (self.partial_dir / item_filename(item)).unlink(missing_ok=True)
            claimed_path.replace(self.pending_dir / item_filename(item))
            count += 1
        return count

    def completed_results(self) -> list[WorkResult]:
        paths = sorted(self.completed_dir.glob("item-*.json"))
        return [WorkResult.from_dict(json.loads(path.read_text(encoding="utf-8"))) for path in paths]

    def observe(self) -> tuple[list[WorkResult], set[str]]:
        """Return the observable results and the ids of completed items, from one scan."""

        completed = {path.name for path in self.completed_dir.glob("item-*.json")}
        names = completed | {path.name for path in self.partial_dir.glob("item-*.json")}
        results: list[WorkResult] = []
        completed_ids: set[str] = set()
        for name in sorted(names):
            result = self._observable_result(name)
            if result is None:
                continue
            results.append(result)
            if name in completed:
                completed_ids.add(result.item_id)
        return results, completed_ids

    def observable_results(self) -> list[WorkResult]:
        return self.observe()[0]

    def is_drained(self) -> bool:
        return not any(self.pending_dir.glob("*.json")) and not any(self.running_dir.glob("*.json"))

    def _pending_candidates(self, preferred_group: str | None) -> list[Path]:
        paths = sorted(self.pending_dir.glob("item-*.json"))
        if preferred_group is None:
            return paths
        preferred = sorted(self.pending_dir.glob(f"item-*-{group_token(preferred_group)}.json"))
        return preferred if preferred else paths

    @staticmethod
    def _validate_result(item: WorkItem, result: WorkResult) -> None:
        if result.item_id != item.item_id:
            raise ValueError("work result item_id does not match the claimed item")
        if result.group_id != item.group_id:
            raise ValueError("work result group_id does not match the claimed item")

    def _observable_result(self, name: str) -> WorkResult | None:
        for path in (self.completed_dir / name, self.partial_dir / name, self.completed_dir / name):
            try:
                return WorkResult.from_dict(json.loads(path.read_text(encoding="utf-8")))
            except FileNotFoundError:
                continue
        return None

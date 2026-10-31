# Copyright 2026 RoboScope Team
# SPDX-License-Identifier: Apache-2.0

"""Adapter contracts for integrating inference and evaluation code."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from roboscope_kit.policy import PolicyExecutionContract

from .schema import WorkItem, WorkResult


@dataclass(frozen=True)
class ProcessSpec:
    """Description of an external process owned by the SDK."""

    command: tuple[str, ...]
    cwd: Path | None = None
    env: dict[str, str] = field(default_factory=dict)
    ready_port: int | None = None
    ready_socket: Path | None = None

    def __post_init__(self) -> None:
        if not self.command:
            raise ValueError("process command cannot be empty")
        if self.ready_port is not None and not 1 <= self.ready_port <= 65535:
            raise ValueError("ready_port must be a valid TCP port")
        if self.ready_port is not None and self.ready_socket is not None:
            raise ValueError("a process cannot declare both ready_port and ready_socket")


@dataclass(frozen=True)
class ReplicaContext:
    replica_id: int
    gpu_slot: int
    gpu_device: str
    port: int
    socket_path: Path
    output_dir: Path


@dataclass(frozen=True)
class WorkerContext:
    worker_id: int
    gpu_slot: int
    gpu_device: str
    replica_id: int
    replica_port: int
    replica_socket: Path
    queue_dir: Path
    state_dir: Path
    output_dir: Path
    execution_contract: PolicyExecutionContract


class InferenceAdapter(Protocol):
    """Launch arbitrary model inference code as independently placed replicas."""

    def baseline_replicas(self, contexts: Sequence[ReplicaContext]) -> None:
        """Capture resource baselines before any replica in the batch starts."""
        ...

    def replica_process(self, context: ReplicaContext) -> ProcessSpec | None: ...

    def initialize_replica(self, context: ReplicaContext) -> None: ...

    def finalize_replicas(self, contexts: Sequence[ReplicaContext]) -> None:
        """Observe resource usage after every replica in the batch is ready."""
        ...

    def healthcheck_replica(self, context: ReplicaContext) -> None: ...

    def release_replica(self, context: ReplicaContext) -> None: ...

    def maximum_workers_per_replica(self) -> int | None: ...

    def metadata(self) -> dict[str, Any]: ...

    def execution_contract(self) -> PolicyExecutionContract: ...


class BenchmarkAdapter(Protocol):
    """Connect an official evaluation protocol to the generic runtime."""

    def work_items(self) -> Sequence[WorkItem]: ...

    def worker_process(self, context: WorkerContext) -> ProcessSpec: ...

    def aggregate(self, results: Sequence[WorkResult]) -> dict[str, Any]: ...

    # Optional: ``aggregate_partial(results)`` for adapters whose official metric is undefined on a
    # subset of the workload. Sharded runs call it instead of ``aggregate``; see ``shard_reduce``.

    def metadata(self) -> dict[str, Any]: ...

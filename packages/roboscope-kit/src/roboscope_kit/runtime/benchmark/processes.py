# Copyright 2026 RoboScope Team
# SPDX-License-Identifier: Apache-2.0

"""Process lifecycle helpers used by the generic benchmark runtime."""

from __future__ import annotations

import hashlib
import os
import re
import signal
import socket
import subprocess
import tempfile
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, TextIO

from .adapters import InferenceAdapter, ProcessSpec, ReplicaContext

if TYPE_CHECKING:
    from .cuda_mps import CudaMpsSession


class ProcessFailureKind(str, Enum):  # noqa: UP042 -- imported by the Python 3.10 LIBERO environment
    OOM = "oom"
    OTHER = "other"


OOM_PATTERNS = (
    re.compile(r"(?im)^cuda out of memory\b"),
    re.compile(
        r"(?im)^.*(?:runtimeerror|outofmemoryerror|memoryerror|cuda error|"
        r"hip error|oidn error).*out of memory.*$"
    ),
    re.compile(r"(?im)^.*runtimeerror:\s*cannot create buffer\s*$"),
    re.compile(r"(?im)^.*failed to allocate device memory.*$"),
    re.compile(r"(?im)^.*mps backend out of memory.*$"),
)

FAILURE_LOG_READ_BYTES = 1024 * 1024
FAILURE_DETAIL_MAX_CHARS = 32 * 1024
ANSI_ESCAPE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


def stop_process(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    process_group = process.pid
    try:
        os.killpg(process_group, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        os.killpg(process_group, signal.SIGKILL)
        process.wait(timeout=10)
    try:
        os.killpg(process_group, signal.SIGKILL)
    except ProcessLookupError:
        return
    if not _wait_process_group_exit(process_group, timeout_s=10.0):
        raise TimeoutError(f"process group {process_group} did not exit after SIGKILL")


def _wait_process_group_exit(process_group: int, *, timeout_s: float) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            os.killpg(process_group, 0)
        except ProcessLookupError:
            return True
        time.sleep(0.05)
    return False


def wait_port(port: int, process: subprocess.Popen, timeout_s: float) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"process exited before TCP port {port} became ready")
        with socket.socket() as sock:
            sock.settimeout(0.5)
            if sock.connect_ex(("127.0.0.1", port)) == 0:
                return
        time.sleep(0.2)
    raise TimeoutError(f"TCP port {port} did not become ready")


def wait_socket(path: Path, process: subprocess.Popen, timeout_s: float) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"process exited before Unix socket {path} became ready")
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
            connection.settimeout(0.5)
            try:
                connection.connect(str(path))
            except (TimeoutError, FileNotFoundError, ConnectionRefusedError):
                time.sleep(0.05)
                continue
            return
    raise TimeoutError(f"Unix socket {path} did not become ready")


@dataclass
class ManagedProcess:
    process: subprocess.Popen
    log_file: TextIO
    log_path: Path
    spec: ProcessSpec

    def stop(self) -> None:
        stop_process(self.process)
        self.log_file.close()

    def failure_kind(self) -> ProcessFailureKind:
        failure_text = self._failure_tail().lower()
        if any(pattern.search(failure_text) for pattern in OOM_PATTERNS):
            return ProcessFailureKind.OOM
        return ProcessFailureKind.OTHER

    def failure_detail(self) -> str:
        tail = self._failure_tail()
        traceback_index = tail.lower().rfind("traceback (most recent call last)")
        detail = tail[traceback_index:] if traceback_index >= 0 else "\n".join(tail.splitlines()[-128:])
        detail = detail.strip()
        if len(detail) <= FAILURE_DETAIL_MAX_CHARS:
            return detail
        half = FAILURE_DETAIL_MAX_CHARS // 2
        return f"{detail[:half]}\n...[worker log truncated]...\n{detail[-half:]}"

    def _failure_tail(self) -> str:
        if not self.log_path.is_file():
            return ""
        with self.log_path.open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            size = handle.tell()
            handle.seek(max(0, size - FAILURE_LOG_READ_BYTES))
            tail = handle.read().decode("utf-8", errors="replace")
        return ANSI_ESCAPE.sub("", tail).replace("\r", "")


def start_process(
    spec: ProcessSpec,
    log_path: Path,
    ready_timeout_s: float,
    *,
    wait_until_ready: bool = True,
) -> ManagedProcess:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = log_path.open("a", encoding="utf-8")
    env = os.environ.copy()
    env.update(spec.env)
    try:
        process = subprocess.Popen(
            spec.command,
            cwd=spec.cwd,
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
    except OSError:
        log_file.close()
        raise
    managed = ManagedProcess(process, log_file, log_path, spec)
    try:
        if wait_until_ready and spec.ready_port is not None:
            wait_port(spec.ready_port, process, ready_timeout_s)
        if wait_until_ready and spec.ready_socket is not None:
            wait_socket(spec.ready_socket, process, ready_timeout_s)
        return managed
    except Exception:
        managed.stop()
        raise


class InferencePool:
    """Scale model inference replicas supplied by an :class:`InferenceAdapter`."""

    def __init__(
        self,
        *,
        adapter: InferenceAdapter,
        gpu_devices: tuple[str, ...],
        base_port: int,
        output_dir: Path,
        ready_timeout_s: float,
        cuda_mps: CudaMpsSession | None = None,
    ) -> None:
        self.adapter = adapter
        self.cuda_mps = cuda_mps
        if not gpu_devices or len(gpu_devices) != len(set(gpu_devices)):
            raise ValueError("gpu_devices must contain unique device identifiers")
        self.gpu_devices = gpu_devices
        self.base_port = base_port
        self.output_dir = output_dir
        self.ready_timeout_s = ready_timeout_s
        self.processes: dict[int, ManagedProcess | None] = {}
        self.contexts: dict[int, ReplicaContext] = {}

    @property
    def size(self) -> int:
        return len(self.processes)

    def scale(self, target: int, *, ready_timeout_s: float | None = None) -> None:
        if target < 1:
            raise ValueError("inference replica target must be positive")
        timeout_s = self.ready_timeout_s if ready_timeout_s is None else ready_timeout_s
        if timeout_s <= 0:
            raise ValueError("inference replica ready timeout must be positive")
        self._remove_replicas(replica_id for replica_id in self.processes if replica_id >= target)
        self._start_replicas(
            (replica_id for replica_id in range(target) if replica_id not in self.processes),
            ready_timeout_s=ready_timeout_s,
        )

    def ensure_healthy(self, *, ready_timeout_s: float | None = None) -> tuple[int, ...]:
        unhealthy = []
        for replica_id in sorted(self.processes):
            try:
                self._healthcheck(replica_id)
            except (OSError, RuntimeError, TimeoutError):
                unhealthy.append(replica_id)
        self._remove_replicas(unhealthy)
        self._start_replicas(unhealthy, ready_timeout_s=ready_timeout_s)
        return tuple(unhealthy)

    def ensure_replica_healthy(self, replica_id: int) -> bool:
        """Rebuild one replica if it fails its healthcheck, returning whether it was rebuilt."""
        try:
            self._healthcheck(replica_id)
        except (OSError, RuntimeError, TimeoutError):
            self._remove_replicas([replica_id])
            self._start_replicas([replica_id])
            return True
        return False

    def _start_replicas(
        self,
        replica_ids: Iterable[int],
        *,
        ready_timeout_s: float | None = None,
    ) -> None:
        contexts = [self._create_context(replica_id) for replica_id in replica_ids]
        if not contexts:
            return
        timeout_s = self.ready_timeout_s if ready_timeout_s is None else ready_timeout_s
        try:
            self.adapter.baseline_replicas(contexts)
            self._launch_replicas(
                contexts,
                timeout_s,
                shared_deadline=ready_timeout_s is not None,
            )
            self.adapter.finalize_replicas(contexts)
        except Exception:
            self._remove_replicas(context.replica_id for context in contexts)
            raise

    def _create_context(self, replica_id: int) -> ReplicaContext:
        context = ReplicaContext(
            replica_id=replica_id,
            gpu_slot=replica_id % len(self.gpu_devices),
            gpu_device=self.gpu_devices[replica_id % len(self.gpu_devices)],
            port=self.base_port + replica_id,
            socket_path=self._socket_path(replica_id),
            output_dir=self.output_dir,
        )
        context.socket_path.unlink(missing_ok=True)
        self.contexts[replica_id] = context
        return context

    def _launch_replicas(
        self,
        contexts: Sequence[ReplicaContext],
        ready_timeout_s: float,
        *,
        shared_deadline: bool,
    ) -> None:
        # Replicas on the same GPU load large checkpoints, so starting them
        # one at a time avoids a transient VRAM spike that fails replica start.
        deadline = time.monotonic() + ready_timeout_s
        for context in contexts:
            remaining_s = deadline - time.monotonic() if shared_deadline else ready_timeout_s
            if remaining_s <= 0:
                raise TimeoutError("inference replica scale operation timed out")
            self._launch_replica(context, remaining_s)

    def _launch_replica(self, context: ReplicaContext, ready_timeout_s: float) -> None:
        try:
            spec = self.adapter.replica_process(context)
        except Exception:
            self.contexts.pop(context.replica_id, None)
            context.socket_path.unlink(missing_ok=True)
            raise
        if spec is None:
            if self.cuda_mps is not None:
                self.contexts.pop(context.replica_id, None)
                raise ValueError("CUDA MPS requires inference replicas that run as separate processes")
            self.processes[context.replica_id] = None
            self.adapter.initialize_replica(context)
            return
        if self.cuda_mps is not None:
            spec = replace(spec, env={**spec.env, **self.cuda_mps.client_env(context.gpu_slot)})
        log_path = self.output_dir / f"replica-{context.replica_id:03d}.log"
        self.processes[context.replica_id] = start_process(
            spec,
            log_path,
            ready_timeout_s,
            wait_until_ready=False,
        )
        managed = self.processes[context.replica_id]
        if managed.spec.ready_port is not None:
            wait_port(managed.spec.ready_port, managed.process, ready_timeout_s)
        if managed.spec.ready_socket is not None:
            wait_socket(managed.spec.ready_socket, managed.process, ready_timeout_s)
        self.adapter.initialize_replica(context)
        if self.cuda_mps is not None:
            # The loaded Policy holds a CUDA context, so it must be an MPS client now.
            self.cuda_mps.require_attached(managed.process.pid)

    def _healthcheck(self, replica_id: int) -> None:
        managed = self.processes[replica_id]
        if managed is not None and managed.process.poll() is not None:
            raise RuntimeError(f"inference replica {replica_id} exited with {managed.process.returncode}")
        self.adapter.healthcheck_replica(self.contexts[replica_id])

    def _remove_replicas(self, replica_ids: Iterable[int]) -> None:
        errors: list[Exception] = []
        for replica_id in sorted(replica_ids, reverse=True):
            managed = self.processes.pop(replica_id, None)
            context = self.contexts.pop(replica_id, None)
            if context is None:
                continue
            try:
                self.adapter.release_replica(context)
            except Exception as exc:
                errors.append(exc)
            try:
                if managed is not None:
                    if self.cuda_mps is not None:
                        self.cuda_mps.terminate_clients(managed.process.pid)
                    managed.stop()
            except Exception as exc:
                errors.append(exc)
            context.socket_path.unlink(missing_ok=True)
        if errors:
            raise errors[0]

    def maximum_workers_per_replica(self) -> int | None:
        return self.adapter.maximum_workers_per_replica()

    def check(self) -> None:
        for replica_id, managed in self.processes.items():
            if managed is None:
                continue
            if managed.process.poll() is not None:
                raise RuntimeError(f"inference replica {replica_id} exited with {managed.process.returncode}")

    def stop(self) -> None:
        self._remove_replicas(tuple(self.processes))

    def __enter__(self) -> InferencePool:
        return self

    def __exit__(self, *_: object) -> None:
        self.stop()

    def _socket_path(self, replica_id: int) -> Path:
        token = hashlib.blake2s(str(self.output_dir.resolve()).encode(), digest_size=8).hexdigest()
        return Path(tempfile.gettempdir()) / f"roboscope-{token}-{replica_id}.sock"

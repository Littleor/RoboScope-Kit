# Copyright 2026 RoboScope Team
# SPDX-License-Identifier: Apache-2.0

"""Optional NVIDIA CUDA MPS session shared by the inference replicas of one run.

CUDA MPS (Multi-Process Service) lets kernels from several replica processes
run on one GPU at the same time instead of time-slicing their contexts. The
session runs a private MPS control daemon for the lifetime of an evaluation:

- it refuses to start next to another MPS daemon or server;
- replicas join it through ``client_env``; simulator workers do not;
- every replica must attach after loading, so a run never falls back to
  plain contexts while reporting MPS as enabled;
- a replica's MPS clients are terminated through the daemon before the
  replica process is signalled, as NVIDIA requires for clients that may still
  have GPU work, so stopping one replica does not fault the shared server.

References: https://docs.nvidia.com/deploy/mps/
"""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
import tempfile
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any, TextIO

# The kernel truncates process names to 15 bytes, which covers both
# nvidia-cuda-mps-control and nvidia-cuda-mps-server.
MPS_PROCESS_COMM = "nvidia-cuda-mps"
DEFAULT_CONTROL = "nvidia-cuda-mps-control"
ACTIVE_THREAD_PERCENTAGE = 100
READY_TIMEOUT_S = 10.0
ATTACH_TIMEOUT_S = 10.0
CONTROL_TIMEOUT_S = 20.0
QUIT_TIMEOUT_S = 15.0


class CudaMpsError(RuntimeError):
    """The MPS daemon or one of its clients is not in the required state."""


class CudaMpsSession:
    """Own one private MPS control daemon for the given GPUs."""

    def __init__(
        self,
        *,
        gpu_devices: Sequence[str],
        log_dir: Path,
        control: str = DEFAULT_CONTROL,
    ) -> None:
        if not gpu_devices or len(gpu_devices) != len(set(gpu_devices)):
            raise ValueError("gpu_devices must contain unique device identifiers")
        self.gpu_devices = tuple(gpu_devices)
        self.log_dir = log_dir
        self.control = control
        self.pipe_dir: Path | None = None
        self.max_clients = 0
        self._controller: subprocess.Popen | None = None
        self._controller_log: TextIO | None = None

    # Lifecycle -----------------------------------------------------------

    def start(self) -> None:
        if self._controller is not None:
            raise CudaMpsError("CUDA MPS session is already running")
        control = shutil.which(self.control)
        if control is None:
            raise CudaMpsError(f"CUDA MPS requires {self.control!r} on PATH")
        self.control = control
        existing = running_mps_processes()
        if existing:
            raise CudaMpsError(f"another CUDA MPS daemon or server is running: PIDs {existing}")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        # Named pipes need a local filesystem and a short path; TMPDIR may be neither.
        self.pipe_dir = Path(tempfile.mkdtemp(prefix="roboscope-mps-", dir="/tmp"))
        self._controller_log = (self.log_dir / "control-stdout.log").open("a", encoding="utf-8")
        self._controller = subprocess.Popen(
            [self.control, "-f"],
            env=self._daemon_env(),
            stdout=self._controller_log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            self._wait_ready()
        except BaseException:
            self.stop()
            raise

    def stop(self) -> None:
        controller = self._controller
        try:
            if controller is not None and controller.poll() is None:
                try:
                    for server, client in self.clients():
                        self._control(f"terminate_client {server} {client}")
                    self._control("quit")
                    controller.wait(timeout=QUIT_TIMEOUT_S)
                except (CudaMpsError, subprocess.TimeoutExpired):
                    _stop_group(controller)
        finally:
            self._controller = None
            if self._controller_log is not None:
                self._controller_log.close()
                self._controller_log = None
            if self.pipe_dir is not None:
                shutil.rmtree(self.pipe_dir, ignore_errors=True)

    def __enter__(self) -> CudaMpsSession:
        self.start()
        return self

    def __exit__(self, *_: object) -> None:
        self.stop()

    # Replica integration -------------------------------------------------

    def client_env(self, gpu_slot: int) -> dict[str, str]:
        """Environment that attaches one replica process to this session.

        MPS clients enumerate GPUs through the daemon, whose visible devices are
        ``gpu_devices``; the replica therefore selects its GPU by slot ordinal.
        """

        if self.pipe_dir is None or self._controller is None:
            raise CudaMpsError("CUDA MPS session is not running")
        if not 0 <= gpu_slot < len(self.gpu_devices):
            raise ValueError(f"GPU slot {gpu_slot} is outside the MPS session")
        return {
            "CUDA_MPS_PIPE_DIRECTORY": str(self.pipe_dir),
            "CUDA_MPS_LOG_DIRECTORY": str(self.log_dir),
            "CUDA_MPS_ACTIVE_THREAD_PERCENTAGE": str(ACTIVE_THREAD_PERCENTAGE),
            "CUDA_VISIBLE_DEVICES": str(gpu_slot),
        }

    def require_attached(self, process_group: int) -> None:
        """Fail unless a process in ``process_group`` is an MPS client."""

        deadline = time.monotonic() + ATTACH_TIMEOUT_S
        while True:
            if self._group_clients(process_group):
                return
            if time.monotonic() >= deadline:
                raise CudaMpsError(
                    f"inference replica process group {process_group} did not attach to CUDA MPS"
                )
            time.sleep(0.2)

    def terminate_clients(self, process_group: int) -> None:
        """Remove the MPS clients of one replica before it is signalled."""

        if self._controller is None or self._controller.poll() is not None:
            return
        for server, client in self._group_clients(process_group):
            self._control(f"terminate_client {server} {client}")

    def clients(self) -> list[tuple[int, int]]:
        pairs = [
            (server, client)
            for server in _pids(self._control("get_server_list"))
            for client in _pids(self._control(f"get_client_list {server}"))
        ]
        self.max_clients = max(self.max_clients, len(pairs))
        return pairs

    def metadata(self) -> dict[str, Any]:
        return {
            "enabled": True,
            "gpu_devices": list(self.gpu_devices),
            "active_thread_percentage": ACTIVE_THREAD_PERCENTAGE,
            "max_observed_clients": self.max_clients,
            "log_dir": str(self.log_dir),
        }

    # Internals -----------------------------------------------------------

    def _daemon_env(self) -> dict[str, str]:
        env = {key: value for key, value in os.environ.items() if not key.startswith("CUDA_MPS_")}
        env.update(
            {
                "CUDA_MPS_PIPE_DIRECTORY": str(self.pipe_dir),
                "CUDA_MPS_LOG_DIRECTORY": str(self.log_dir),
                "CUDA_MPS_ACTIVE_THREAD_PERCENTAGE": str(ACTIVE_THREAD_PERCENTAGE),
                "CUDA_VISIBLE_DEVICES": ",".join(self.gpu_devices),
            }
        )
        return env

    def _wait_ready(self) -> None:
        assert self._controller is not None
        deadline = time.monotonic() + READY_TIMEOUT_S
        while True:
            if self._controller.poll() is not None:
                raise CudaMpsError(
                    f"CUDA MPS control daemon exited with {self._controller.returncode}; "
                    f"see {self.log_dir}"
                )
            try:
                self.clients()
                return
            except CudaMpsError:
                if time.monotonic() >= deadline:
                    raise
            time.sleep(0.1)

    def _control(self, command: str) -> str:
        try:
            result = subprocess.run(
                [self.control],
                input=command + "\n",
                env=self._daemon_env(),
                capture_output=True,
                text=True,
                timeout=CONTROL_TIMEOUT_S,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise CudaMpsError(f"CUDA MPS command {command!r} failed: {exc}") from exc
        if result.returncode != 0:
            raise CudaMpsError(
                f"CUDA MPS command {command!r} exited {result.returncode}: "
                f"{(result.stdout + result.stderr).strip()}"
            )
        return result.stdout.strip()

    def _group_clients(self, process_group: int) -> list[tuple[int, int]]:
        return [
            (server, client)
            for server, client in self.clients()
            if _process_group(client) == process_group
        ]


def running_mps_processes(proc: Path = Path("/proc")) -> list[int]:
    """PIDs of MPS daemons or servers visible in this PID namespace."""

    found = []
    for entry in proc.iterdir():
        if not entry.name.isdecimal():
            continue
        try:
            comm = (entry / "comm").read_text(encoding="utf-8", errors="replace").strip()
        except OSError:
            continue
        if comm == MPS_PROCESS_COMM:
            found.append(int(entry.name))
    return sorted(found)


def _pids(output: str) -> list[int]:
    fields = output.split()
    if any(not field.isdecimal() for field in fields):
        raise CudaMpsError(f"unrecognized CUDA MPS PID list: {output!r}")
    return sorted({int(field) for field in fields})


def _process_group(pid: int) -> int | None:
    try:
        return os.getpgid(pid)
    except (ProcessLookupError, PermissionError):
        return None


def _stop_group(process: subprocess.Popen) -> None:
    for sig, timeout_s in ((signal.SIGTERM, 5.0), (signal.SIGKILL, 5.0)):
        try:
            os.killpg(process.pid, sig)
        except ProcessLookupError:
            return
        try:
            process.wait(timeout=timeout_s)
            return
        except subprocess.TimeoutExpired:
            continue


__all__ = ["CudaMpsError", "CudaMpsSession", "running_mps_processes"]

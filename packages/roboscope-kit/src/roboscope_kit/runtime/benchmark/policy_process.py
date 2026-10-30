# Copyright 2026 RoboScope Team
# SPDX-License-Identifier: Apache-2.0

"""Launch user policy entrypoints behind the RoboScope inference protocol."""

from __future__ import annotations

import subprocess
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from roboscope_kit.benchmark_specs import BenchmarkName
from roboscope_kit.client import PolicyClient
from roboscope_kit.contracts import BenchmarkSpec
from roboscope_kit.policy import (
    PolicyCapabilities,
    PolicyExecutionContract,
    SessionContext,
    negotiate_policy_action_contract,
)

from .adapters import ProcessSpec, ReplicaContext


class PolicyProcessAdapter:
    """Start one user-owned SDK policy process per inference replica."""

    def __init__(
        self,
        *,
        command: tuple[str, ...],
        checkpoint_path: str | Path,
        benchmark: BenchmarkName,
        suite: str,
        benchmark_spec: BenchmarkSpec,
        cwd: Path | None = None,
        env: dict[str, str] | None = None,
        session_metadata: dict[str, Any] | None = None,
        load_timeout_s: float = 30.0,
    ) -> None:
        if not command:
            raise ValueError("policy command cannot be empty")
        if not suite:
            raise ValueError("benchmark suite cannot be empty")
        if benchmark_spec.name != benchmark.value or benchmark_spec.suite != suite:
            raise ValueError("benchmark_spec does not match the policy process benchmark and suite")
        if load_timeout_s <= 0:
            raise ValueError("policy load timeout must be positive")
        self.command = command
        self.checkpoint_path = str(checkpoint_path)
        self.benchmark = benchmark
        self.suite = suite
        self.benchmark_spec = benchmark_spec
        self.cwd = cwd
        self.env = dict(env or {})
        self.session_metadata = dict(session_metadata or {})
        self.load_timeout_s = load_timeout_s
        self.capabilities: dict[int, PolicyCapabilities] = {}
        self.gpu_vram_baselines: dict[str, int] = {}
        self.replica_vram_bytes: dict[int, int] = {}
        self._execution_contract: PolicyExecutionContract | None = None

    def baseline_replicas(self, contexts: Sequence[ReplicaContext]) -> None:
        for gpu_device in sorted({context.gpu_device for context in contexts}):
            self.gpu_vram_baselines.pop(gpu_device, None)
            baseline = self._query_gpu_memory_bytes(gpu_device)
            if baseline is not None:
                self.gpu_vram_baselines[gpu_device] = baseline

    def replica_process(self, context: ReplicaContext) -> ProcessSpec:
        env = {
            **self.env,
            "ROBOSCOPE_INFERENCE_SOCKET": str(context.socket_path),
            "ROBOSCOPE_REPLICA_ID": str(context.replica_id),
            # Several simulator workers share one policy replica. torch.compile
            # (pytorch_compile_mode="default") records a CUDA Graph on the first
            # inference of a new input shape; CUDA Graph capture is device-level
            # exclusive, so concurrent first-inferences from different workers
            # collide with "beginAllocateToPool: already recording to mempool_id"
            # Disabling inductor cudagraph trees removes the capture
            # session entirely; the compiled Triton kernels are still used, so
            # there is no measurable throughput loss on eval workloads. Callers
            # may opt back in by setting TORCHINDUCTOR_CUDAGRAPHS explicitly in
            # the adapter env (this default is applied only when unset).
            "TORCHINDUCTOR_CUDAGRAPHS": self.env.get(
                "TORCHINDUCTOR_CUDAGRAPHS", "0"
            ),
        }
        env["CUDA_VISIBLE_DEVICES"] = context.gpu_device
        return ProcessSpec(
            command=self.command,
            cwd=self.cwd,
            env=env,
            ready_socket=context.socket_path,
        )

    def initialize_replica(self, context: ReplicaContext) -> None:
        session = SessionContext(
            session_id=f"{self.benchmark.value}-{self.suite}-replica-{context.replica_id}",
            checkpoint_path=self.checkpoint_path,
            benchmark=self.benchmark_spec,
            metadata={
                **self.session_metadata,
                "replica_id": context.replica_id,
                "gpu_slot": context.gpu_slot,
                "gpu_device": context.gpu_device,
            },
        )
        with PolicyClient(context.socket_path, timeout_s=self.load_timeout_s) as client:
            capabilities = client.load(session)
        contract = negotiate_policy_action_contract(session.benchmark, capabilities)
        if self._execution_contract is not None and contract != self._execution_contract:
            raise ValueError(
                f"policy replica {context.replica_id} execution contract {contract.to_dict()} "
                f"does not match {self._execution_contract.to_dict()}"
            )
        self.capabilities[context.replica_id] = capabilities
        self._execution_contract = contract

    def finalize_replicas(self, contexts: Sequence[ReplicaContext]) -> None:
        groups: dict[str, list[int]] = {}
        for context in contexts:
            groups.setdefault(context.gpu_device, []).append(context.replica_id)
        for gpu_device, replica_ids in groups.items():
            self._record_gpu_vram(gpu_device, replica_ids)

    def _record_gpu_vram(self, gpu_device: str, replica_ids: list[int]) -> None:
        baseline = self.gpu_vram_baselines.pop(gpu_device, None)
        current = self._query_gpu_memory_bytes(gpu_device)
        if baseline is None or current is None or current < baseline:
            return
        per_replica = (current - baseline) // len(replica_ids)
        for replica_id in replica_ids:
            self.replica_vram_bytes[replica_id] = per_replica

    def healthcheck_replica(self, context: ReplicaContext) -> None:
        with PolicyClient(context.socket_path, timeout_s=self.load_timeout_s) as client:
            health = client.health()
        if health.get("state") != "ready":
            raise RuntimeError(f"policy replica {context.replica_id} is not ready: {health.get('state')}")

    def release_replica(self, context: ReplicaContext) -> None:
        try:
            with PolicyClient(context.socket_path, timeout_s=self.load_timeout_s) as client:
                client.unload()
        except (
            BrokenPipeError,
            ConnectionAbortedError,
            ConnectionRefusedError,
            ConnectionResetError,
            FileNotFoundError,
            TimeoutError,
        ):
            pass
        finally:
            self.capabilities.pop(context.replica_id, None)
            self.gpu_vram_baselines.pop(context.gpu_device, None)
            self.replica_vram_bytes.pop(context.replica_id, None)
            if not self.capabilities:
                self._execution_contract = None

    def maximum_workers_per_replica(self) -> None:
        return None

    def vram_bytes_per_replica(self) -> int | None:
        """Estimated per-replica policy VRAM footprint, or ``None`` if unknown."""

        if not self.replica_vram_bytes:
            return None
        return max(self.replica_vram_bytes.values())

    def metadata(self) -> dict[str, Any]:
        return {
            "adapter": type(self).__name__,
            "benchmark": self.benchmark.value,
            "suite": self.suite,
            "checkpoint_path": self.checkpoint_path,
            "load_timeout_s": self.load_timeout_s,
            "replica_vram_bytes": {
                str(replica_id): vram for replica_id, vram in sorted(self.replica_vram_bytes.items())
            },
            "replicas": {
                str(replica_id): capabilities.to_dict()
                for replica_id, capabilities in sorted(self.capabilities.items())
            },
            "execution_contract": self.execution_contract().to_dict(),
        }

    def execution_contract(self) -> PolicyExecutionContract:
        if self._execution_contract is None:
            raise RuntimeError("policy action contract is unavailable before replica load")
        return self._execution_contract

    @staticmethod
    def _query_gpu_memory_bytes(gpu_device: str) -> int | None:
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    f"--id={gpu_device}",
                    "--query-gpu=memory.used",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                check=False,
                timeout=10,
            )
        except (OSError, subprocess.TimeoutExpired):
            return None
        if result.returncode != 0:
            return None
        try:
            mib = float(result.stdout.strip().splitlines()[0])
        except (IndexError, ValueError):
            return None
        return int(mib * 1024**2)


__all__ = ["PolicyProcessAdapter"]

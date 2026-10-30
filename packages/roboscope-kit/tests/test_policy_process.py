# Copyright 2026 RoboScope Team
# SPDX-License-Identifier: Apache-2.0

"""Policy process environment defaults (cudagraph capture conflict)."""

from __future__ import annotations

from pathlib import Path

from roboscope_kit.benchmark_specs import BenchmarkName, get_benchmark_spec
from roboscope_kit.runtime.benchmark.adapters import ReplicaContext
from roboscope_kit.runtime.benchmark.policy_process import (
    PolicyProcessAdapter,
)


def _adapter(*, env: dict[str, str] | None = None) -> PolicyProcessAdapter:
    return PolicyProcessAdapter(
        command=("/usr/bin/python", "policy_entry.py"),
        checkpoint_path=Path("/ckpt/2000"),
        benchmark=BenchmarkName.ROBOTWIN,
        suite="demo_clean",
        benchmark_spec=get_benchmark_spec(BenchmarkName.ROBOTWIN, "demo_clean"),
        cwd=Path("/work"),
        env=env,
    )


def _context(replica_id: int = 0) -> ReplicaContext:
    return ReplicaContext(
        replica_id=replica_id,
        gpu_slot=0,
        gpu_device="0",
        port=8000,
        socket_path=Path(f"/tmp/policy-{replica_id}.sock"),
        output_dir=Path("/tmp/out"),
    )


def test_policy_server_disables_inductor_cudagraphs_by_default() -> None:
    spec = _adapter().replica_process(_context())
    # Concurrent first-inference from several workers sharing one
    # torch.compile policy replica collides on CUDA Graph capture
    # ("beginAllocateToPool: already recording to mempool_id"). The SDK policy
    # server must default TORCHINDUCTOR_CUDAGRAPHS=0 to remove the capture
    # session; callers can opt back in explicitly.
    assert spec.env["TORCHINDUCTOR_CUDAGRAPHS"] == "0"


def test_policy_server_respects_explicit_cudagraph_env() -> None:
    spec = _adapter(env={"TORCHINDUCTOR_CUDAGRAPHS": "1"}).replica_process(_context())
    assert spec.env["TORCHINDUCTOR_CUDAGRAPHS"] == "1"


def test_policy_server_keeps_protocol_and_cuda_env() -> None:
    spec = _adapter().replica_process(_context(replica_id=3))
    assert spec.env["ROBOSCOPE_INFERENCE_SOCKET"] == "/tmp/policy-3.sock"
    assert spec.env["ROBOSCOPE_REPLICA_ID"] == "3"
    assert spec.env["CUDA_VISIBLE_DEVICES"] == "0"

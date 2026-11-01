# Copyright 2026 RoboScope Team
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import io
import json
import os
import signal
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from roboscope_kit import PolicyExecutionContract
from roboscope_kit.runtime.benchmark.adapters import ProcessSpec, WorkerContext
from roboscope_kit.runtime.benchmark.autotune import (
    AutotuneBounds,
    AutotuneLimits,
    BenchmarkAutotuneController,
    ProbeCapacityReached,
    ProbeLimits,
    ProbeTimedOut,
)
from roboscope_kit.runtime.benchmark.orchestrator import (
    ParallelBenchmarkRuntime,
    RuntimeConfig,
)
from roboscope_kit.runtime.benchmark.processes import (
    ManagedProcess,
    ProcessFailureKind,
    start_process,
)
from roboscope_kit.runtime.benchmark.schema import EvaluationRecord, WorkItem
from roboscope_kit.runtime.benchmarks.robotwin.worker import (
    _EvalSession,
    _failed_task_result,
)


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def _bounds(*, max_workers: int = 16, max_replicas: int = 8) -> AutotuneBounds:
    return AutotuneBounds(
        gpu_count=4,
        cpu_cores=80.0,
        max_workers=max_workers,
        max_replicas=max_replicas,
        worker_cpu_cores=2.0,
        vram_per_gpu_bytes=48 * 1024**3,
    )


def _health(k: int, *, final: int | None = None, failures: int = 0, oom: int = 0) -> dict[str, int]:
    return {
        "requested_workers": k,
        "initial_workers": k,
        "final_workers": k if final is None else final,
        "worker_failures": failures,
        "oom_events": oom,
        "completed_episodes": k,
        "expected_episodes": k,
    }


def _output(k: int, throughput: float, *, gpu_utilization: float = 0.6) -> dict:
    return {
        "benchmark": {"bound_hint": "gpu_infer"},
        "result": {
            "steps_per_s": throughput,
            "episodes": k,
            "successes": k,
            "success_rate": 1.0,
            "total_episodes": k,
        },
        "resources": {
            "cpu": {"cpu_cores": k * 0.5, "throttle_time_ratio": 0.0},
            "gpu": {
                "mean_utilization": gpu_utilization,
                "peak_memory_bytes": 12 * 1024**3,
            },
        },
        "recovery": {
            "worker_failures": [],
            "oom_events": 0,
            "initial_workers_per_gpu": {"0": k},
            "final_workers_per_gpu": {"0": k},
        },
        "inference_metrics": {"queue_wait_ms": {"p95": 5.0}},
    }


def _selection(path: Path) -> dict:
    return json.loads((path / "selection.json").read_text(encoding="utf-8"))


def test_controller_persists_each_candidate_and_falls_back_after_oom(tmp_path: Path) -> None:
    clock = FakeClock()
    calls = []

    def measure(k: int, g: int, limits: ProbeLimits, progress):
        running = _selection(tmp_path)
        assert running["candidates"][-1]["status"] == "running"
        calls.append((k, g, limits.timeout_s))
        clock.advance(1.0)
        progress(
            {
                "episodes": 1,
                "total_episodes": k,
                "successes": 1,
                "failures": 0,
            }
        )
        if (k, g) == (16, 4):
            raise ProbeCapacityReached(
                "simulator worker cannot create buffer",
                stop_reason="worker_oom",
                health=_health(k, final=15, failures=1, oom=1),
            )
        throughput = {(4, 4): 40.0, (8, 4): 60.0, (8, 8): 70.0}[k, g]
        return _output(k, throughput), 9 * 1024**3

    result = BenchmarkAutotuneController(
        measure=measure,
        output_dir=tmp_path,
        bounds=_bounds(),
        limits=AutotuneLimits(candidate_timeout_s=30.0, total_budget_s=120.0, strategy="two-phase"),
        clock=clock,
    ).run()

    assert result["k"] == 8
    assert result["g"] == 8
    assert calls == [(4, 4, 30.0), (8, 4, 30.0), (16, 4, 30.0), (8, 8, 30.0)]
    selection = _selection(tmp_path)
    assert selection["status"] == "complete"
    assert [candidate["status"] for candidate in selection["candidates"]] == [
        "accepted",
        "accepted",
        "rejected",
        "accepted",
    ]
    rejected = selection["candidates"][2]
    assert rejected["stop_reason"] == "worker_oom"
    assert rejected["health"]["completed_episodes"] == 16
    assert selection["candidates"][0]["progress"]["completed_episodes"] == 1
    assert "worker_oom" in selection["k_stop_reason"]


def test_controller_times_out_candidate_and_keeps_last_healthy_k(tmp_path: Path) -> None:
    clock = FakeClock()

    def measure(k: int, _g: int, _limits: ProbeLimits, _progress):
        clock.advance(10.0 if k == 8 else 1.0)
        return _output(k, 40.0 if k == 4 else 70.0, gpu_utilization=0.9), 9 * 1024**3

    result = BenchmarkAutotuneController(
        measure=measure,
        output_dir=tmp_path,
        bounds=_bounds(max_workers=8),
        limits=AutotuneLimits(candidate_timeout_s=10.0, total_budget_s=100.0, strategy="two-phase"),
        clock=clock,
    ).run()

    assert (result["k"], result["g"]) == (4, 4)
    selection = _selection(tmp_path)
    assert selection["candidates"][1]["status"] == "timed_out"
    assert selection["candidates"][1]["stop_reason"] == "candidate_timeout"


def test_controller_caps_last_candidate_at_remaining_total_budget(tmp_path: Path) -> None:
    clock = FakeClock()
    observed_limits = []

    def measure(k: int, _g: int, limits: ProbeLimits, _progress):
        observed_limits.append(limits)
        clock.advance(10.0 if k == 4 else limits.timeout_s)
        return _output(k, 40.0 if k == 4 else 70.0, gpu_utilization=0.9), 9 * 1024**3

    result = BenchmarkAutotuneController(
        measure=measure,
        output_dir=tmp_path,
        bounds=_bounds(max_workers=16),
        limits=AutotuneLimits(candidate_timeout_s=100.0, total_budget_s=15.0, strategy="two-phase"),
        clock=clock,
    ).run()

    assert (result["k"], result["g"]) == (4, 4)
    assert observed_limits[1].timeout_s == 5.0
    assert observed_limits[1].timeout_reason == "total_budget_exhausted"
    timed_out = _selection(tmp_path)["candidates"][1]
    assert timed_out["status"] == "timed_out"
    assert timed_out["stop_reason"] == "total_budget_exhausted"


def test_controller_keeps_failed_audit_when_first_candidate_is_unhealthy(tmp_path: Path) -> None:
    def measure(k: int, _g: int, _limits: ProbeLimits, _progress):
        raise ProbeCapacityReached(
            "minimum concurrency failed",
            stop_reason="worker_failure",
            health=_health(k, final=3, failures=1),
        )

    controller = BenchmarkAutotuneController(
        measure=measure,
        output_dir=tmp_path,
        bounds=_bounds(max_workers=4),
        limits=AutotuneLimits(candidate_timeout_s=10.0, total_budget_s=20.0, strategy="two-phase"),
    )
    with pytest.raises(ProbeCapacityReached, match="minimum concurrency failed"):
        controller.run()

    selection = _selection(tmp_path)
    assert selection["status"] == "failed"
    assert selection["candidates"][0]["status"] == "rejected"
    assert selection["candidates"][0]["health"]["final_workers"] == 3


class _FakeProcess:
    returncode = 1

    @staticmethod
    def poll() -> int:
        return 1


class _FakeManagedProcess:
    def __init__(self, log_path: Path) -> None:
        self.process = _FakeProcess()
        self.log_path = log_path

    @staticmethod
    def failure_kind() -> ProcessFailureKind:
        return ProcessFailureKind.OOM

    @staticmethod
    def failure_detail() -> str:
        return "RuntimeError: cannot create buffer"


class _FakeQueue:
    @staticmethod
    def observable_results() -> list:
        return []

    @staticmethod
    def is_drained() -> bool:
        return False


class _FakeServerPool:
    @staticmethod
    def check() -> None:
        return None


def test_probe_runtime_rejects_oom_before_recovery(tmp_path: Path) -> None:
    runtime = ParallelBenchmarkRuntime.__new__(ParallelBenchmarkRuntime)
    runtime.config = RuntimeConfig(
        output_dir=tmp_path,
        gpu_devices=("0",),
        replicas=1,
        concurrency=1,
        probe_limits=ProbeLimits(timeout_s=10.0),
    )
    managed = _FakeManagedProcess(tmp_path / "worker.log")
    runtime.server_pool = _FakeServerPool()
    runtime.workers = {0: managed}
    runtime.worker_contexts = {
        0: WorkerContext(
            worker_id=0,
            gpu_slot=0,
            gpu_device="0",
            replica_id=0,
            replica_port=16694,
            replica_socket=tmp_path / "replica.sock",
            queue_dir=tmp_path / "queue",
            state_dir=tmp_path / "state",
            output_dir=tmp_path,
            execution_contract=PolicyExecutionContract(
                benchmark="libero",
                action_representation="libero.delta-eef-v1",
            ),
        )
    }
    runtime.state_dir = tmp_path / "state"
    runtime.state_dir.mkdir()
    runtime.initial_workers_per_gpu = {"0": 1}
    runtime.failure_events = []
    runtime.queue = _FakeQueue()

    with pytest.raises(ProbeCapacityReached) as raised:
        runtime._check_processes(tmp_path / "events.jsonl", total_records=1)

    assert raised.value.stop_reason == "worker_oom"
    assert raised.value.health["final_workers"] == 0
    assert raised.value.health["oom_events"] == 1
    event = json.loads((tmp_path / "events.jsonl").read_text(encoding="utf-8"))
    assert event["action"] == "abort-probe"
    assert event["stop_reason"] == "worker_oom"


def test_probe_runtime_reports_deadline_with_partial_health(tmp_path: Path) -> None:
    runtime = ParallelBenchmarkRuntime.__new__(ParallelBenchmarkRuntime)
    runtime.config = RuntimeConfig(
        output_dir=tmp_path,
        gpu_devices=("0",),
        replicas=1,
        concurrency=1,
        probe_limits=ProbeLimits(timeout_s=5.0, timeout_reason="total_budget_exhausted"),
    )
    runtime.workers = {}
    runtime.initial_workers_per_gpu = {"0": 1}
    runtime.failure_events = []
    runtime.queue = _FakeQueue()

    with pytest.raises(ProbeTimedOut) as raised:
        runtime._raise_probe_timeout(-1.0, total_records=3, events_path=tmp_path / "events.jsonl")

    assert raised.value.stop_reason == "total_budget_exhausted"
    assert raised.value.health["completed_episodes"] == 0
    assert raised.value.health["expected_episodes"] == 3


def test_managed_process_stops_its_descendant_process_group(tmp_path: Path) -> None:
    child_pid_path = tmp_path / "child.pid"
    parent_code = (
        "import subprocess,sys,time; "
        f"child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)']); "
        f"open({str(child_pid_path)!r},'w').write(str(child.pid)); "
        "time.sleep(60)"
    )
    managed = start_process(
        ProcessSpec(command=(sys.executable, "-c", parent_code)),
        tmp_path / "managed.log",
        ready_timeout_s=1.0,
        wait_until_ready=False,
    )
    child_pid: int | None = None
    try:
        deadline = time.monotonic() + 5.0
        while not child_pid_path.is_file() and time.monotonic() < deadline:
            time.sleep(0.05)
        child_pid = int(child_pid_path.read_text(encoding="utf-8"))
        assert os.getpgid(managed.process.pid) == managed.process.pid
        assert os.getpgid(child_pid) == managed.process.pid

        managed.stop()

        deadline = time.monotonic() + 5.0
        while _process_is_running(child_pid) and time.monotonic() < deadline:
            time.sleep(0.05)
        assert not _process_is_running(child_pid)
    finally:
        if managed.process.poll() is None:
            managed.stop()
        if child_pid is not None and _process_is_running(child_pid):
            os.kill(child_pid, signal.SIGKILL)


def test_robotwin_eval_session_inherits_worker_process_group(tmp_path: Path) -> None:
    launcher = tmp_path / "launcher.py"
    environment_path = tmp_path / "launcher-environment.txt"
    launcher.write_text(
        "import os\n"
        "import time\n"
        "from pathlib import Path\n"
        f"Path({str(environment_path)!r}).write_text("
        "os.environ['ROBOTWIN_DISABLE_VIDEO'] + ' ' + "
        "os.environ['ROBOTWIN_SEED_CACHE_ENABLED'], encoding='utf-8')\n"
        "time.sleep(60)\n",
        encoding="utf-8",
    )
    args = SimpleNamespace(
        gpu_id=0,
        policy_socket=tmp_path / "policy.sock",
        robotwin_python=sys.executable,
        launcher=launcher,
        robotwin_dir=tmp_path,
        disable_video=True,
        seed_cache_dir=tmp_path / "seed-cache",
    )
    session = _EvalSession(args)
    process = session._spawn()
    try:
        deadline = time.monotonic() + 5.0
        while not environment_path.is_file() and time.monotonic() < deadline:
            time.sleep(0.05)
        assert os.getpgid(process.pid) == os.getpgrp()
        assert environment_path.read_text(encoding="utf-8") == "1 1"
    finally:
        session.stop()


def test_robotwin_failed_window_fills_only_its_global_episode_indices(tmp_path: Path) -> None:
    item = WorkItem(
        ordinal=0,
        item_id="demo-randomized-adjust-bottle-episodes-0010-0013",
        group_id="demo-randomized-adjust-bottle-episodes-0010-0013",
        payload={},
        expected_results=3,
    )
    existing = EvaluationRecord(
        record_id="demo_randomized-adjust_bottle-episode-0011",
        success=True,
        steps=42,
        elapsed_s=1.0,
        metadata={"episode_index": 11},
    )

    result = _failed_task_result(
        item,
        2,
        task="adjust_bottle",
        suite="demo_randomized",
        episode_start=10,
        episodes=3,
        records=[existing],
        returncode=1,
        log_path=tmp_path / "native.log",
        detail="setup failed",
    )

    assert [record.metadata["episode_index"] for record in result.records] == [10, 11, 12]
    assert [record.success for record in result.records] == [False, True, False]
    assert result.metadata["episode_start"] == 10
    assert result.metadata["episode_stop"] == 13


def test_managed_process_oom_detection_reads_before_last_traceback(tmp_path: Path) -> None:
    log_path = tmp_path / "worker.log"
    log_path.write_text(
        "Traceback (most recent call last):\n"
        "RuntimeError: CUDA error: out of memory\n"
        "Traceback (most recent call last):\n"
        "ImportError: cannot import name 'CuroboPlanner'\n",
        encoding="utf-8",
    )
    managed = ManagedProcess(
        process=_FakeProcess(),
        log_file=io.StringIO(),
        log_path=log_path,
        spec=ProcessSpec(command=("unused",)),
    )

    assert managed.failure_kind() == ProcessFailureKind.OOM
    assert managed.failure_detail().startswith("Traceback (most recent call last):\nImportError")


def _process_is_running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True

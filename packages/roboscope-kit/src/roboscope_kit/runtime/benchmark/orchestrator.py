# Copyright 2026 RoboScope Team
# SPDX-License-Identifier: Apache-2.0

"""Dataset- and model-independent parallel benchmark orchestration."""

from __future__ import annotations

import json
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence  # noqa: UP035 - this alias is evaluated by Python 3.8

from .adapters import BenchmarkAdapter, InferenceAdapter, WorkerContext
from .autotune import ProbeCapacityReached, ProbeLimits, ProbeTimedOut
from .cuda_mps import CudaMpsSession
from .metrics import read_jsonl_since, summarize_events
from .nsys import profiled_process
from .processes import InferencePool, ManagedProcess, ProcessFailureKind, start_process
from .queue import WorkQueue, atomic_json
from .resources import GpuSampler, cpu_quota, cpu_window, gpu_window, read_cpu_stat
from .schema import EvaluationRecord, WorkItem, WorkResult, validate_terminal_accounting
from .sharding import (
    DEFAULT_EPISODES_PER_SHARD,
    plan_shard,
)
from .timing import PhaseTimingCallback, benchmark_phase

MIN_WORKERS_PER_GPU = 1
MAX_OOM_RESTARTS_PER_GPU = 1
MAX_OTHER_RESTARTS = 1
OOM_RESTART_DELAY_S = 5.0


@dataclass(frozen=True)
class RuntimeConfig:
    output_dir: Path
    gpu_devices: tuple[str, ...]
    replicas: int
    concurrency: int
    base_port: int = 16694
    ready_timeout_s: float = 900
    poll_interval_s: float = 0.2
    probe_limits: ProbeLimits | None = None
    shard_index: int | None = None
    shard_count: int | None = None
    shard_episodes: int = DEFAULT_EPISODES_PER_SHARD
    shard_build: dict[str, str] | None = None
    profile: bool = False

    def __post_init__(self) -> None:
        if not self.gpu_devices or len(self.gpu_devices) != len(set(self.gpu_devices)):
            raise ValueError("gpu_devices must contain unique device identifiers")
        if self.replicas < 1:
            raise ValueError("replicas must be positive")
        if self.concurrency < 1:
            raise ValueError("concurrency must be positive")
        if self.ready_timeout_s <= 0:
            raise ValueError("ready_timeout_s must be positive")
        if self.poll_interval_s <= 0:
            raise ValueError("poll_interval_s must be positive")
        if self.shard_count is not None and self.shard_index is None:
            raise ValueError("shard_count requires shard_index")
        if self.shard_index is not None and self.shard_index < 0:
            raise ValueError("shard_index must be non-negative")
        if self.shard_count is not None and self.shard_count < 1:
            raise ValueError("shard_count must be positive")
        if self.shard_episodes < 1:
            raise ValueError("shard_episodes must be positive")

    @property
    def gpu_count(self) -> int:
        return len(self.gpu_devices)


ProgressCallback = Callable[["dict[str, Any]", Sequence[WorkResult]], None]


def aggregate_standard_results(results: Sequence[WorkResult], total_records: int) -> dict[str, Any]:
    records = [record for result in results for record in result.records]
    successes = sum(record.success for record in records)
    total_steps = sum(record.steps for record in records)
    return {
        "state": "complete" if len(records) == total_records else "running",
        "episodes": len(records),
        "successes": successes,
        "failures": len(records) - successes,
        "total_steps": total_steps,
        "total_episodes": total_records,
        "success_rate": successes / len(records) if records else 0.0,
        "completed_items": len(results),
    }


def terminal_records(results: Sequence[WorkResult]) -> list[EvaluationRecord]:
    return [record for result in results for record in result.records]


class ParallelBenchmarkRuntime:
    """Coordinate inference replicas, evaluator workers, queueing, and result accounting."""

    def __init__(
        self,
        *,
        config: RuntimeConfig,
        inference: InferenceAdapter,
        benchmark: BenchmarkAdapter,
        server_pool: InferencePool | None = None,
        cuda_mps: CudaMpsSession | None = None,
    ) -> None:
        self.config = config
        self.inference = inference
        self.benchmark = benchmark
        self.queue = WorkQueue(config.output_dir / "queue")
        self.state_dir = config.output_dir / "worker-state"
        if server_pool is not None and server_pool.adapter is not inference:
            raise ValueError("shared inference pool must use the runtime inference adapter")
        if server_pool is not None and server_pool.gpu_devices != config.gpu_devices:
            raise ValueError("shared inference pool GPU devices must match RuntimeConfig")
        if server_pool is not None and server_pool.base_port != config.base_port:
            raise ValueError("shared inference pool base port must match RuntimeConfig")
        if server_pool is not None and cuda_mps is not None and server_pool.cuda_mps is not cuda_mps:
            raise ValueError("shared inference pool must use the runtime CUDA MPS session")
        self.owns_server_pool = server_pool is None
        self.server_pool = server_pool or InferencePool(
            adapter=inference,
            gpu_devices=config.gpu_devices,
            base_port=config.base_port,
            output_dir=config.output_dir / "servers",
            ready_timeout_s=config.ready_timeout_s,
            cuda_mps=cuda_mps,
        )
        self.workers: dict[int, ManagedProcess] = {}
        self.worker_contexts: dict[int, WorkerContext] = {}
        self.worker_attempts: Counter[int] = Counter()
        self.worker_start_order: dict[int, int] = {}
        self.next_worker_start_order = 0
        self.oom_restarts: Counter[int] = Counter()
        self.other_restarts: Counter[int] = Counter()
        self.failure_events: list[dict[str, Any]] = []
        self.recycle_events: list[dict[str, Any]] = []
        self.initial_workers_per_gpu: dict[str, int] = {}
        # Seconds from evaluation start at which the controller first saw each work item completed.
        self.item_completion_s: dict[str, float] = {}
        self.evaluation_started_s = 0.0

    def run(
        self,
        progress_callback: ProgressCallback | None = None,
        phase_timing_callback: PhaseTimingCallback | None = None,
    ) -> dict[str, Any]:
        return self.run_workers(progress_callback, phase_timing_callback)

    def run_workers(
        self,
        progress_callback: ProgressCallback | None = None,
        phase_timing_callback: PhaseTimingCallback | None = None,
    ) -> dict[str, Any]:
        """Run one evaluation pass, scaling the pool to ``config.replicas``.

        The caller keeps ownership of an injected shared pool and may invoke this
        repeatedly to probe several worker concurrencies against one policy load.
        When the pool is owned by the runtime, prefer :meth:`run`, which also
        stops the pool on completion.
        """

        probe_deadline = (
            time.monotonic() + self.config.probe_limits.timeout_s
            if self.config.probe_limits is not None
            else None
        )
        items = list(self.benchmark.work_items())
        if not items:
            raise ValueError("benchmark adapter produced no work items")
        shard = self._select_shard(items)
        if shard is not None:
            items, shard_metadata = shard
        else:
            shard_metadata = None
        total_records = sum(item.expected_results for item in items)
        benchmark_metadata = self.benchmark.metadata()
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        events_path = self.config.output_dir / "events.jsonl"
        events_path.write_text("", encoding="utf-8")
        self.queue.initialize(items)
        self._clear_worker_state()
        append_event(events_path, {"type": "phase", "phase": "server-starting"})
        previous_replicas = set(self.server_pool.contexts)
        policy_load_started = time.monotonic()
        with benchmark_phase(phase_timing_callback, "policy_load_s"):
            try:
                self._raise_probe_timeout(probe_deadline, total_records, events_path)
                self.server_pool.scale(
                    self.config.replicas,
                    ready_timeout_s=self._probe_ready_timeout(probe_deadline),
                )
                self._raise_probe_timeout(probe_deadline, total_records, events_path)
                restarted_replicas = self.server_pool.ensure_healthy(
                    ready_timeout_s=self._probe_ready_timeout(probe_deadline)
                )
                policy_load_s = time.monotonic() - policy_load_started
            except (ProbeCapacityReached, ProbeTimedOut):
                self._stop_owned_server_pool()
                raise
            except (OSError, RuntimeError, TimeoutError) as exc:
                self._record_policy_server_error(events_path, exc)
                self._stop_owned_server_pool()
                if isinstance(exc, TimeoutError):
                    self._raise_probe_timeout(probe_deadline, total_records, events_path)
                if self.config.probe_limits is not None:
                    raise ProbeCapacityReached(
                        f"probe g={self.config.replicas} could not start its policy replicas: {exc}",
                        stop_reason="policy_replica_failure",
                        health=self._probe_health(total_records),
                    ) from exc
                raise
            except Exception as exc:
                self._record_policy_server_error(events_path, exc)
                self._stop_owned_server_pool()
                raise
        active_replicas = set(self.server_pool.contexts)
        restarted_replica_ids = set(restarted_replicas)
        if restarted_replicas:
            append_event(
                events_path.with_name("errors.jsonl"),
                {
                    "type": "policy-server-restart",
                    "component": "policy-server",
                    "replica_ids": list(restarted_replicas),
                    "action": "restart",
                },
            )
        append_event(
            events_path,
            {
                "type": "inference-pool",
                "target_replicas": self.config.replicas,
                "reused_replicas": len((previous_replicas & active_replicas) - restarted_replica_ids),
                "started_replicas": len(active_replicas - previous_replicas),
                "stopped_replicas": len(previous_replicas - active_replicas),
                "restarted_replicas": list(restarted_replicas),
            },
        )
        policy_io = self.inference.execution_contract().to_dict()
        atomic_json(
            self.config.output_dir / "manifest.json",
            {
                "sdk_version": 1,
                "benchmark": benchmark_metadata,
                "policy_io": policy_io,
                "items": [item.to_dict() for item in items],
                "total_records": total_records,
                **({"profile": True} if self.config.profile else {}),
                **({"shard": shard_metadata} if shard_metadata is not None else {}),
            },
        )
        maximum_workers = self.server_pool.maximum_workers_per_replica()
        if maximum_workers is not None and self.config.concurrency > self.config.replicas * maximum_workers:
            self._stop_owned_server_pool()
            raise ValueError(
                f"concurrency {self.config.concurrency} exceeds policy capacity "
                f"{self.config.replicas * maximum_workers}"
            )
        inference_offsets = self._inference_metric_offsets()
        sampler = GpuSampler(gpu_devices=self.config.gpu_devices)
        before_cpu = read_cpu_stat()
        sampler.start()
        started = time.monotonic()
        surviving_workers_per_gpu: dict[str, int] = {}
        try:
            append_event(events_path, {"type": "phase", "phase": "workers-starting"})
            runtime_prepare_started = time.monotonic()
            with benchmark_phase(phase_timing_callback, "runtime_prepare_s"):
                self._start_workers(min(self.config.concurrency, len(items)))
                self.initial_workers_per_gpu = self._workers_per_gpu()
                self._wait_workers_ready(events_path, total_records, probe_deadline)
                runtime_prepare_s = time.monotonic() - runtime_prepare_started
            append_event(events_path, {"type": "phase", "phase": "evaluation"})
            self.queue.start()
            self.evaluation_started_s = time.monotonic()
            with benchmark_phase(phase_timing_callback, "evaluation_s"):
                results = self._monitor(
                    total_records,
                    progress_callback,
                    events_path,
                    probe_deadline,
                )
                evaluation_s = time.monotonic() - self.evaluation_started_s
            surviving_workers_per_gpu = self._workers_per_gpu()
            self.queue.request_stop()
            self._wait_workers_exit()
        finally:
            self.queue.request_stop()
            samples = sampler.stop()
            self._stop_workers()
            inference_metadata = self.inference.metadata()
            self._stop_owned_server_pool()
        elapsed_s = time.monotonic() - started
        # A shard only holds part of the workload, so an adapter whose official metric needs the
        # whole run reports raw partials here and leaves the metric to the merge step.
        aggregate_partial = getattr(self.benchmark, "aggregate_partial", None)
        if shard_metadata is not None and aggregate_partial is not None:
            aggregate = aggregate_partial(results)
        else:
            aggregate = self.benchmark.aggregate(results)
        validate_terminal_accounting(results, total_records)
        standard = aggregate_standard_results(results, total_records)
        standard["elapsed_s"] = elapsed_s
        standard["steps_per_s"] = standard["total_steps"] / max(elapsed_s, 1e-9)
        standard["episodes_per_s"] = standard["episodes"] / max(elapsed_s, 1e-9)
        steady = steady_throughput(results, self.item_completion_s)
        if steady is not None:
            standard["steady_steps_per_s"] = steady["steps_per_s"]
            standard["steady_window"] = steady
        output = {
            "sdk_version": 1,
            "runtime": asdict(self.config),
            **({"shard": shard_metadata} if shard_metadata is not None else {}),
            "phase_timings": {
                "policy_load_s": policy_load_s,
                "runtime_prepare_s": runtime_prepare_s,
                "evaluation_s": evaluation_s,
            },
            "item_timeline": {
                "completion_s": dict(self.item_completion_s),
                "worker_id": {result.item_id: result.worker_id for result in results},
            },
            "record_success": {
                record.record_id: record.success for result in results for record in result.records
            },
            "inference": inference_metadata,
            "policy_io": policy_io,
            "benchmark": benchmark_metadata,
            "result": standard,
            "official_result": aggregate,
            "resources": {
                "elapsed_s": elapsed_s,
                "cpu": cpu_window(before_cpu, read_cpu_stat(), elapsed_s, cpu_quota()),
                "gpu": gpu_window(samples, self.config.gpu_count),
                "gpu_samples": len(samples),
            },
            "recovery": {
                "worker_failures": self.failure_events,
                "planned_recycles": len(self.recycle_events),
                "oom_events": sum(
                    event["failure_kind"] == ProcessFailureKind.OOM for event in self.failure_events
                ),
                "initial_workers_per_gpu": self.initial_workers_per_gpu,
                "final_workers_per_gpu": surviving_workers_per_gpu,
            },
        }
        inference_events = [
            event
            for path in sorted(self.server_pool.output_dir.glob("inference-metrics-*.jsonl"))
            for event in read_jsonl_since(path, inference_offsets.get(path, 0))
        ]
        if inference_events:
            output["inference_metrics"] = summarize_events(inference_events)
        output["runtime"]["output_dir"] = str(self.config.output_dir)
        atomic_json(self.config.output_dir / "result.json", output)
        append_event(events_path, {"type": "complete", **standard})
        return output

    def _select_shard(
        self, items: Sequence[WorkItem]
    ) -> tuple[list[WorkItem], dict[str, Any]] | None:
        """Apply the deterministic shard partition for the final (non-probe) pass.

        Autotune probes always run unsharded so every instance measures K/g
        throughput against the same probe workload; only the final pass carries
        the shard metadata that the merge tool validates.
        """

        if self.config.probe_limits is not None or self.config.shard_index is None:
            return None
        return plan_shard(
            items,
            self.config.shard_index,
            self.config.shard_count,
            self.config.shard_episodes,
            build=self.config.shard_build,
        )

    def _stop_owned_server_pool(self) -> None:
        if self.owns_server_pool:
            self.server_pool.stop()

    def _probe_ready_timeout(self, probe_deadline: float | None) -> float | None:
        if probe_deadline is None:
            return None
        remaining_s = max(probe_deadline - time.monotonic(), 1e-6)
        return min(self.config.ready_timeout_s, remaining_s)

    @staticmethod
    def _record_policy_server_error(events_path: Path, exc: Exception) -> None:
        append_event(
            events_path.with_name("errors.jsonl"),
            {
                "type": "policy-server-error",
                "component": "policy-server",
                "error_type": type(exc).__name__,
                "message": str(exc),
                "action": "abort",
            },
        )

    def _inference_metric_offsets(self) -> dict[Path, int]:
        return {
            path: path.stat().st_size
            for path in self.server_pool.output_dir.glob("inference-metrics-*.jsonl")
        }

    def _start_workers(self, worker_count: int) -> None:
        execution_contract = self.inference.execution_contract()
        for worker_id in range(worker_count):
            replica_id = worker_id % self.config.replicas
            context = WorkerContext(
                worker_id=worker_id,
                gpu_slot=worker_id % self.config.gpu_count,
                gpu_device=self.config.gpu_devices[worker_id % self.config.gpu_count],
                replica_id=replica_id,
                replica_port=self.config.base_port + replica_id,
                replica_socket=self.server_pool.contexts[replica_id].socket_path,
                queue_dir=self.queue.root,
                state_dir=self.state_dir,
                output_dir=self.config.output_dir,
                execution_contract=execution_contract,
            )
            self.worker_contexts[worker_id] = context
            self._start_worker(worker_id)

    def _start_worker(self, worker_id: int) -> None:
        context = self.worker_contexts[worker_id]
        state_path = self.state_dir / f"worker-{worker_id:03d}.json"
        state_path.unlink(missing_ok=True)
        attempt = self.worker_attempts[worker_id]
        spec = self.benchmark.worker_process(context)
        if self.config.profile:
            report = self.config.output_dir / "profile" / f"worker-{worker_id:03d}-attempt-{attempt:03d}"
            spec = profiled_process(spec, report)
        log_path = self.config.output_dir / "workers" / f"worker-{worker_id:03d}-attempt-{attempt:03d}.log"
        self.workers[worker_id] = start_process(spec, log_path, self.config.ready_timeout_s)
        self.worker_attempts[worker_id] += 1
        self.worker_start_order[worker_id] = self.next_worker_start_order
        self.next_worker_start_order += 1

    def _wait_workers_ready(
        self,
        events_path: Path,
        total_records: int,
        probe_deadline: float | None,
    ) -> None:
        deadline = time.monotonic() + self.config.ready_timeout_s
        while time.monotonic() < deadline:
            state_paths = (self.state_dir / f"worker-{worker_id:03d}.json" for worker_id in self.workers)
            if all(path.exists() for path in state_paths):
                return
            self._check_processes(events_path, total_records)
            self._raise_probe_timeout(probe_deadline, total_records, events_path)
            time.sleep(self.config.poll_interval_s)
        self._raise_probe_timeout(probe_deadline, total_records, events_path)
        raise TimeoutError("benchmark workers did not become ready")

    def _monitor(
        self,
        total_records: int,
        progress_callback: ProgressCallback | None,
        events_path: Path,
        probe_deadline: float | None,
    ) -> list[WorkResult]:
        reported = -1
        while True:
            self._check_processes(events_path, total_records)
            observable_results, completed_ids = self.queue.observe()
            self._record_completions(completed_ids)
            record_count = sum(len(result.records) for result in observable_results)
            if record_count != reported:
                progress = aggregate_standard_results(observable_results, total_records)
                atomic_json(self.config.output_dir / "progress.json", progress)
                append_event(events_path, {"type": "progress", **progress})
                if progress_callback is not None:
                    progress_callback(progress, observable_results)
                reported = record_count
            if record_count == total_records and self.queue.is_drained():
                results = self.queue.completed_results()
                self._record_completions(result.item_id for result in results)
                return results
            self._raise_probe_timeout(probe_deadline, total_records, events_path)
            time.sleep(self.config.poll_interval_s)

    def _record_completions(self, item_ids: Iterable[str]) -> None:
        """Note when each work item first appears completed; partial results do not count."""

        seen_s = time.monotonic() - self.evaluation_started_s
        for item_id in item_ids:
            self.item_completion_s.setdefault(item_id, seen_s)

    def _check_processes(self, events_path: Path, total_records: int) -> None:
        self.server_pool.check()
        exited = [
            (worker_id, managed)
            for worker_id, managed in self.workers.items()
            if managed.process.poll() is not None
        ]
        self._reject_probe_worker_exits(exited, total_records, events_path)
        exited_by_gpu: dict[int, list[tuple[int, ManagedProcess]]] = defaultdict(list)
        for worker_id, managed in exited:
            gpu_slot = self.worker_contexts[worker_id].gpu_slot
            exited_by_gpu[gpu_slot].append((worker_id, managed))
        for gpu_slot, failed_workers in exited_by_gpu.items():
            if any(self._is_oom_exit(worker_id, managed) for worker_id, managed in failed_workers):
                self._handle_gpu_oom(gpu_slot, failed_workers, events_path)
                continue
            for worker_id, managed in failed_workers:
                self._handle_worker_exit(worker_id, managed, events_path)
        self._reject_probe_worker_loss(total_records, events_path)

    def _reject_probe_worker_exits(
        self,
        exited: list[tuple[int, ManagedProcess]],
        total_records: int,
        events_path: Path,
    ) -> None:
        limits = self.config.probe_limits
        failures = [
            (worker_id, managed)
            for worker_id, managed in exited
            if not self._is_planned_recycle(worker_id, managed)
        ]
        if limits is None or not failures:
            return
        oom_failures = [
            (worker_id, managed)
            for worker_id, managed in failures
            if self._is_oom_exit(worker_id, managed)
        ]
        if oom_failures and limits.abort_on_oom:
            reason = "worker_oom"
            rejected = oom_failures
            failure_kind = ProcessFailureKind.OOM
        elif limits.abort_on_worker_failure:
            reason = "worker_failure"
            rejected = failures
            failure_kind = ProcessFailureKind.OTHER
        else:
            return
        event = {
            "type": "probe-rejected",
            "component": "worker",
            "action": "abort-probe",
            "stop_reason": reason,
            "failure_kind": failure_kind,
            "failed_worker_ids": [worker_id for worker_id, _ in rejected],
            "failures": [
                self._worker_failure_payload(worker_id, managed)
                for worker_id, managed in rejected
            ],
        }
        self._record_failure(event, events_path)
        health = self._probe_health(total_records)
        raise ProbeCapacityReached(
            f"probe K={self.config.concurrency} stopped on {reason}",
            stop_reason=reason,
            health=health,
        )

    def _reject_probe_worker_loss(self, total_records: int, events_path: Path) -> None:
        limits = self.config.probe_limits
        if limits is None or not limits.abort_on_worker_loss:
            return
        health = self._probe_health(total_records)
        if health["completed_episodes"] == total_records and self.queue.is_drained():
            return
        if health["final_workers"] >= health["initial_workers"]:
            return
        event = {
            "type": "probe-rejected",
            "component": "worker",
            "action": "abort-probe",
            "stop_reason": "worker_loss",
            "failure_kind": ProcessFailureKind.OTHER,
            "initial_workers": health["initial_workers"],
            "final_workers": health["final_workers"],
        }
        self._record_failure(event, events_path)
        health = self._probe_health(total_records)
        raise ProbeCapacityReached(
            f"probe K={self.config.concurrency} lost workers "
            f"({health['initial_workers']} -> {health['final_workers']})",
            stop_reason="worker_loss",
            health=health,
        )

    def _raise_probe_timeout(
        self,
        deadline: float | None,
        total_records: int,
        events_path: Path,
    ) -> None:
        limits = self.config.probe_limits
        if deadline is None or limits is None or time.monotonic() < deadline:
            return
        health = self._probe_health(total_records)
        event = {
            "type": "probe-timeout",
            "component": "runtime",
            "action": "abort-probe",
            "stop_reason": limits.timeout_reason,
            "timeout_s": limits.timeout_s,
            "completed_episodes": health["completed_episodes"],
            "expected_episodes": health["expected_episodes"],
        }
        append_event(events_path, event)
        append_event(events_path.with_name("errors.jsonl"), event)
        raise ProbeTimedOut(
            f"probe K={self.config.concurrency}, g={self.config.replicas} "
            f"exceeded {limits.timeout_s:.3f}s",
            stop_reason=limits.timeout_reason,
            health=health,
        )

    def _probe_health(self, total_records: int) -> dict[str, int]:
        observable_results = self.queue.observable_results()
        completed_records = sum(len(result.records) for result in observable_results)
        return {
            "requested_workers": self.config.concurrency,
            "initial_workers": sum(self.initial_workers_per_gpu.values()),
            "final_workers": sum(
                managed.process.poll() is None for managed in self.workers.values()
            ),
            "worker_failures": len(self.failure_events),
            "oom_events": sum(
                event.get("failure_kind") == ProcessFailureKind.OOM
                for event in self.failure_events
            ),
            "completed_episodes": completed_records,
            "expected_episodes": total_records,
        }

    def _is_planned_recycle(self, worker_id: int, managed: ManagedProcess) -> bool:
        state_path = self.state_dir / f"worker-{worker_id:03d}.json"
        worker_state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.is_file() else {}
        return managed.process.returncode == 0 and worker_state.get("state") == "recycling"

    def _is_oom_exit(self, worker_id: int, managed: ManagedProcess) -> bool:
        if self._is_planned_recycle(worker_id, managed):
            return False
        return managed.failure_kind() == ProcessFailureKind.OOM

    def _handle_gpu_oom(
        self,
        gpu_slot: int,
        failed_workers: list[tuple[int, ManagedProcess]],
        events_path: Path,
    ) -> None:
        active_worker_ids = self._worker_ids_on_gpu(gpu_slot)
        active_before = len(active_worker_ids)
        if active_before <= MIN_WORKERS_PER_GPU:
            worker_id, managed = failed_workers[0]
            self._handle_minimum_gpu_oom(worker_id, managed, events_path)
            return

        retired_worker_id = max(active_worker_ids, key=self.worker_start_order.__getitem__)
        failed_by_id = dict(failed_workers)
        failure_details = []
        requeued = 0
        if retired_worker_id not in failed_by_id:
            requeued += self._remove_worker(retired_worker_id, self.workers[retired_worker_id])
        for worker_id, managed in failed_by_id.items():
            failure_details.append(self._worker_failure_payload(worker_id, managed))
            requeued += self._remove_worker(worker_id, managed)

        time.sleep(OOM_RESTART_DELAY_S)
        late_failures = [
            (worker_id, managed)
            for worker_id, managed in self.workers.items()
            if self.worker_contexts[worker_id].gpu_slot == gpu_slot
            and managed.process.poll() is not None
        ]
        for worker_id, managed in late_failures:
            failed_by_id[worker_id] = managed
            failure_details.append(self._worker_failure_payload(worker_id, managed))
            requeued += self._remove_worker(worker_id, managed)

        failed_worker_ids = set(failed_by_id)
        restarted_worker_ids = sorted(failed_worker_ids - {retired_worker_id})
        for worker_id in restarted_worker_ids:
            self._start_worker(worker_id)
        event = {
            "type": "worker-oom-scale-down",
            "component": "worker",
            "gpu_id": self.config.gpu_devices[gpu_slot],
            "gpu_slot": gpu_slot,
            "failure_kind": ProcessFailureKind.OOM,
            "failed_worker_ids": sorted(failed_worker_ids),
            "retired_worker_id": retired_worker_id,
            "restarted_worker_ids": restarted_worker_ids,
            "requeued_items": requeued,
            "active_workers_before": active_before,
            "active_workers_after": self._workers_on_gpu(gpu_slot),
            "action": "reduce-gpu-concurrency",
            "failures": failure_details,
        }
        self._record_failure(event, events_path)

    def _handle_minimum_gpu_oom(
        self,
        worker_id: int,
        managed: ManagedProcess,
        events_path: Path,
    ) -> None:
        context = self.worker_contexts[worker_id]
        failure = self._worker_failure_payload(worker_id, managed)
        requeued = self._remove_worker(worker_id, managed)
        event = {
            **failure,
            "requeued_items": requeued,
            "active_workers_before": 1,
        }
        restarts = self.oom_restarts[context.gpu_slot]
        if restarts >= MAX_OOM_RESTARTS_PER_GPU:
            event["action"] = "abort-minimum-concurrency-oom"
            event["active_workers_after"] = 0
            self._record_failure(event, events_path)
            detail = str(failure.get("detail") or "")
            raise RuntimeError(
                f"benchmark worker {worker_id} OOM at minimum GPU concurrency; "
                f"requeued {requeued} items{self._failure_detail_message(detail)}"
            )
        self.oom_restarts[context.gpu_slot] += 1
        time.sleep(OOM_RESTART_DELAY_S)
        self._start_worker(worker_id)
        event["action"] = "restart-minimum-concurrency"
        event["active_workers_after"] = self._workers_on_gpu(context.gpu_slot)
        self._record_failure(event, events_path)

    def _worker_failure_payload(self, worker_id: int, managed: ManagedProcess) -> dict[str, Any]:
        context = self.worker_contexts[worker_id]
        detail = managed.failure_detail()
        payload = {
            "type": "worker-failure",
            "component": "worker",
            "worker_id": worker_id,
            "gpu_id": context.gpu_device,
            "gpu_slot": context.gpu_slot,
            "replica_id": context.replica_id,
            "returncode": managed.process.returncode,
            "failure_kind": managed.failure_kind(),
            "log_path": str(managed.log_path),
        }
        if detail:
            payload["detail"] = detail
        return payload

    def _remove_worker(self, worker_id: int, managed: ManagedProcess) -> int:
        managed.stop()
        requeued = self.queue.requeue_worker(worker_id)
        self.workers.pop(worker_id)
        return requeued

    def _handle_worker_exit(self, worker_id: int, managed: ManagedProcess, events_path: Path) -> None:
        context = self.worker_contexts[worker_id]
        returncode = managed.process.returncode
        active_before = self._workers_on_gpu(context.gpu_slot)
        state_path = self.state_dir / f"worker-{worker_id:03d}.json"
        worker_state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.is_file() else {}
        requeued = self._remove_worker(worker_id, managed)
        if returncode == 0 and worker_state.get("state") == "recycling":
            event = {
                "type": "worker-recycle",
                "worker_id": worker_id,
                "gpu_id": context.gpu_device,
                "gpu_slot": context.gpu_slot,
                "replica_id": context.replica_id,
                "completed_records": worker_state["completed_records"],
                "requeued_items": requeued,
                "active_workers_before": active_before,
                "log_path": str(managed.log_path),
            }
            if any(self.queue.pending_dir.glob("*.json")):
                self._start_worker(worker_id)
                event["action"] = "restart"
            else:
                event["action"] = "retire"
            event["active_workers_after"] = self._workers_on_gpu(context.gpu_slot)
            self.recycle_events.append(event)
            append_event(events_path, event)
            return
        failure_kind = managed.failure_kind()
        failure_detail = managed.failure_detail()
        event = {
            "type": "worker-failure",
            "component": "worker",
            "worker_id": worker_id,
            "gpu_id": context.gpu_device,
            "gpu_slot": context.gpu_slot,
            "replica_id": context.replica_id,
            "returncode": returncode,
            "failure_kind": failure_kind,
            "requeued_items": requeued,
            "active_workers_before": active_before,
            "log_path": str(managed.log_path),
        }
        if failure_detail:
            event["detail"] = failure_detail
        if failure_kind != ProcessFailureKind.OOM:
            if self.other_restarts[worker_id] >= MAX_OTHER_RESTARTS:
                event["action"] = "abort-other-retry-exhausted"
                event["active_workers_after"] = self._workers_on_gpu(context.gpu_slot)
                self._record_failure(event, events_path)
                raise RuntimeError(
                    f"benchmark worker {worker_id} repeatedly exited with {returncode}; "
                    f"failure_kind={failure_kind}; requeued {requeued} items"
                    f"{self._failure_detail_message(failure_detail)}"
                )
            self.other_restarts[worker_id] += 1
            event["action"] = "restart-other"
            try:
                # A policy replica that entered a failed state cannot serve
                # reset/act; restarting only the worker would retry against the
                # same failed replica and always fail.
                replica_rebuilt = self.server_pool.ensure_replica_healthy(context.replica_id)
                event["replica_rebuilt"] = replica_rebuilt
                self._start_worker(worker_id)
            finally:
                event["active_workers_after"] = self._workers_on_gpu(context.gpu_slot)
                self._record_failure(event, events_path)
            return
        raise RuntimeError("OOM worker exits must be handled as a GPU scale-down event")

    def _record_failure(self, event: dict[str, Any], events_path: Path) -> None:
        self.failure_events.append(event)
        append_event(events_path, event)
        append_event(events_path.with_name("errors.jsonl"), event)

    @staticmethod
    def _failure_detail_message(detail: str) -> str:
        return f"\nOriginal worker error from the final attempt:\n{detail}" if detail else ""

    def _workers_on_gpu(self, gpu_slot: int) -> int:
        return len(self._worker_ids_on_gpu(gpu_slot))

    def _worker_ids_on_gpu(self, gpu_slot: int) -> list[int]:
        return [
            worker_id
            for worker_id in self.workers
            if self.worker_contexts[worker_id].gpu_slot == gpu_slot
        ]

    def _workers_per_gpu(self) -> dict[str, int]:
        return {
            device: self._workers_on_gpu(gpu_slot) for gpu_slot, device in enumerate(self.config.gpu_devices)
        }

    def _wait_workers_exit(self) -> None:
        for managed in self.workers.values():
            managed.process.wait(timeout=30)

    def _stop_workers(self) -> None:
        for worker_id in sorted(self.workers, reverse=True):
            self.workers[worker_id].stop()
        self.workers.clear()

    def _clear_worker_state(self) -> None:
        for state_path in self.state_dir.glob("worker-*.json"):
            state_path.unlink()



def steady_throughput(
    results: Sequence[WorkResult],
    completion_s: dict[str, float],
) -> dict[str, Any] | None:
    """Steady-state steps/s of K concurrently busy workers, or ``None`` without a window.

    Workers run items back to back, so an item spans from its worker's previous
    completion (or evaluation start) to its own completion. Each worker's first
    item carries environment creation and Policy warm-up and is dropped. The window
    ends when the first worker finishes its last item, after which fewer than K
    workers compete for CPU and inference, so later items are dropped too. A
    worker's rate is the steps of its remaining items over their spans; the
    throughput is K times the mean rate of the workers that have such items.
    Completion times are rounded up to the controller poll interval.
    """

    by_worker: dict[int, list[tuple[float, int]]] = {}
    for result in results:
        done_s = completion_s.get(result.item_id)
        if done_s is None:
            return None
        steps = sum(record.steps for record in result.records)
        by_worker.setdefault(result.worker_id, []).append((done_s, steps))
    if not by_worker:
        return None
    for items in by_worker.values():
        items.sort()
    window_end_s = min(items[-1][0] for items in by_worker.values())
    window_start_s = max(items[0][0] for items in by_worker.values())
    rates: list[float] = []
    measured_items = 0
    for items in by_worker.values():
        warm_s = items[0][0]
        counted = [(done_s, steps) for done_s, steps in items[1:] if done_s <= window_end_s]
        if not counted:
            continue
        span_s = counted[-1][0] - warm_s
        if span_s <= 0:
            continue
        rates.append(sum(steps for _, steps in counted) / span_s)
        measured_items += len(counted)
    if not rates:
        return None
    return {
        "steps_per_s": len(by_worker) * sum(rates) / len(rates),
        "workers": len(by_worker),
        "measured_workers": len(rates),
        "measured_items": measured_items,
        "window_s": max(0.0, window_end_s - window_start_s),
    }


def append_event(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")

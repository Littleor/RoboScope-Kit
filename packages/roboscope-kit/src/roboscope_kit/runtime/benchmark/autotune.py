# Copyright 2026 RoboScope Team
# SPDX-License-Identifier: Apache-2.0

"""Benchmark-agnostic K/g autotune runner built on the shared selection rules.

Each candidate is measured by running a short real benchmark probe, then the
shared :mod:`roboscope_kit.runtime.autotune` rules pick worker
concurrency ``K`` and inference replica count ``g`` from the measured
throughput and resource signals.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict  # noqa: UP035 - CALVIN worker uses Python 3.8

from roboscope_kit.runtime.autotune import (
    BoundHint,
    ClientMetrics,
    KTrial,
    ServerMetrics,
    build_g_candidates,
    build_k_candidates,
    select_k,
    should_ramp_g,
)

from .queue import atomic_json
from .resources import cpu_quota
from .timing import PhaseTimingCallback, benchmark_phase

RESERVED_VRAM_PER_GPU_BYTES = 2 * 1024**3

DEFAULT_CANDIDATE_TIMEOUT_S = 15 * 60.0
DEFAULT_TOTAL_BUDGET_S = 30 * 60.0
DEFAULT_MAXIMUM_SUCCESS_RATE_DROP = 0.15

ProbeProgressFn = Callable[[Dict[str, Any]], None]  # noqa: UP006 - evaluated by Python 3.8
MeasureFn = Callable[
    [int, int, "ProbeLimits", ProbeProgressFn],
    "tuple[dict[str, Any], int | None]",
]
ClockFn = Callable[[], float]

# Estimated CPU cores one simulator worker needs. Used only to derive a sane
# initial K ceiling before any measurement; the manifest may override it via
# ``defaults.worker_cpu_cores`` for heavy simulators.
DEFAULT_WORKER_CPU_CORES = 2.0
# Safety caps so a bad cores-per-worker estimate can never produce an absurd
# ramp; the adaptive CPU-derived ceiling is normally the binding constraint.
ABSOLUTE_MAX_WORKERS = 64
ABSOLUTE_MAX_REPLICAS = 16


class ProbeStopped(RuntimeError):
    """A candidate stopped without invalidating earlier healthy candidates."""

    status = "rejected"

    def __init__(
        self,
        message: str,
        *,
        stop_reason: str,
        health: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.stop_reason = stop_reason
        self.health = dict(health or {})


class ProbeCapacityReached(ProbeStopped):
    """Raised when a probe cannot sustain its requested worker concurrency."""

    def __init__(
        self,
        message: str,
        *,
        stop_reason: str = "capacity_reached",
        health: dict[str, int] | None = None,
    ) -> None:
        super().__init__(message, stop_reason=stop_reason, health=health)


class ProbeTimedOut(ProbeStopped):
    """Raised when a probe reaches its candidate or total-budget deadline."""

    status = "timed_out"


class ProbeCorrectnessRegression(ProbeStopped):
    """Raised when concurrency degrades probe success relative to the baseline."""

    def __init__(self, message: str, *, health: dict[str, Any]) -> None:
        super().__init__(
            message,
            stop_reason="correctness_regression",
            health=health,
        )


class AutotuneBudgetExhausted(RuntimeError):
    """Raised when no autotune budget remains for another candidate."""


@dataclass(frozen=True)
class ProbeLimits:
    """Fail-fast and deadline controls applied only to a probe runtime."""

    timeout_s: float
    timeout_reason: str = "candidate_timeout"
    abort_on_oom: bool = True
    abort_on_worker_failure: bool = True
    abort_on_worker_loss: bool = True

    def __post_init__(self) -> None:
        if self.timeout_s <= 0:
            raise ValueError("probe timeout_s must be positive")
        if self.timeout_reason not in {"candidate_timeout", "total_budget_exhausted"}:
            raise ValueError("unsupported probe timeout_reason")


@dataclass(frozen=True)
class AutotuneLimits:
    """Hard wall-clock limits for candidate probes and the complete tune."""

    candidate_timeout_s: float = DEFAULT_CANDIDATE_TIMEOUT_S
    total_budget_s: float = DEFAULT_TOTAL_BUDGET_S
    maximum_success_rate_drop: float = DEFAULT_MAXIMUM_SUCCESS_RATE_DROP

    def __post_init__(self) -> None:
        if self.candidate_timeout_s <= 0:
            raise ValueError("autotune candidate_timeout_s must be positive")
        if self.total_budget_s <= 0:
            raise ValueError("autotune total_budget_s must be positive")
        if self.maximum_success_rate_drop < 0 or self.maximum_success_rate_drop >= 1:
            raise ValueError("autotune maximum_success_rate_drop must be in [0, 1)")


@dataclass(frozen=True)
class AutotuneBounds:
    """Hardware and manifest caps that bound the K/g candidate ramps."""

    gpu_count: int
    cpu_cores: float
    max_workers: int
    max_replicas: int
    worker_cpu_cores: float
    vram_per_gpu_bytes: int | None


def resolve_bounds(gpu_count: int, manifest_defaults: dict[str, Any]) -> AutotuneBounds:
    """Derive resource-aware K/g ceilings from the actual machine.

    ``max_workers`` is estimated from the CPU budget and the per-worker core
    cost (``worker_cpu_cores``) rather than a fixed constant, so a single-GPU
    heavy-simulator host does not get a meaningless 64-worker ramp. The
    estimate is refined once a real single-worker probe reports actual usage.
    """

    if gpu_count < 1:
        raise ValueError("autotune requires at least one visible GPU")
    cpu_cores = cpu_quota()
    worker_cpu_cores = float(manifest_defaults.get("worker_cpu_cores", DEFAULT_WORKER_CPU_CORES))
    if worker_cpu_cores <= 0:
        raise ValueError("worker_cpu_cores must be positive")
    # Reserve ~1 core for the controller, then floor at one worker per GPU.
    cpu_bound = max(1, int((cpu_cores - 1.0) // worker_cpu_cores))
    ceiling = min(ABSOLUTE_MAX_WORKERS, cpu_bound)
    # An explicit manifest cap tightens the adaptive ceiling further.
    if "maximum_workers" in manifest_defaults:
        ceiling = min(ceiling, int(manifest_defaults["maximum_workers"]))
    max_workers = max(gpu_count, ceiling)
    max_replicas = max(
        gpu_count,
        min(int(manifest_defaults.get("maximum_replicas", ABSOLUTE_MAX_REPLICAS)), int(cpu_cores)),
    )
    vram_raw = manifest_defaults.get("vram_per_gpu_bytes")
    return AutotuneBounds(
        gpu_count=gpu_count,
        cpu_cores=cpu_cores,
        max_workers=max_workers,
        max_replicas=max_replicas,
        worker_cpu_cores=worker_cpu_cores,
        vram_per_gpu_bytes=int(vram_raw) if vram_raw else None,
    )


def resolve_limits(manifest_defaults: dict[str, Any]) -> AutotuneLimits:
    """Resolve immutable benchmark-image limits for the autotune controller."""

    return AutotuneLimits(
        candidate_timeout_s=float(
            manifest_defaults.get("autotune_candidate_timeout_s", DEFAULT_CANDIDATE_TIMEOUT_S)
        ),
        total_budget_s=float(
            manifest_defaults.get("autotune_total_budget_s", DEFAULT_TOTAL_BUDGET_S)
        ),
        maximum_success_rate_drop=float(
            manifest_defaults.get(
                "autotune_maximum_success_rate_drop",
                DEFAULT_MAXIMUM_SUCCESS_RATE_DROP,
            )
        ),
    )


def k_candidate_plan(bounds: AutotuneBounds) -> tuple[int, ...]:
    """Worker-concurrency ramp: geometric from one per GPU, CPU-capped.

    Geometric growth (not +1 steps) keeps the candidate count logarithmic in
    the ceiling, so a high CPU-derived bound does not translate into dozens of
    full benchmark probes.
    """

    return build_k_candidates(bounds.gpu_count, bounds.max_workers)


def g_candidate_plan(bounds: AutotuneBounds, replica_vram_bytes: int) -> tuple[int, ...]:
    """Replica ramp capped by measured policy footprint and GPU capacity."""

    maximum = bounds.max_replicas
    if bounds.vram_per_gpu_bytes is None:
        raise ValueError("GPU VRAM capacity is required to plan inference replicas")
    available = max(bounds.vram_per_gpu_bytes - RESERVED_VRAM_PER_GPU_BYTES, 0)
    per_gpu = max(1, available // max(replica_vram_bytes, 1))
    maximum = max(bounds.gpu_count, min(maximum, bounds.gpu_count * per_gpu))
    return build_g_candidates(bounds.gpu_count, maximum)


def run_autotune(
    *,
    measure: MeasureFn,
    output_dir: Path,
    bounds: AutotuneBounds,
    limits: AutotuneLimits,
    phase_timing_callback: PhaseTimingCallback | None = None,
) -> dict[str, Any]:
    """Measure and select K/g while reporting the complete autotune phase."""

    with autotune_phase(phase_timing_callback):
        return BenchmarkAutotuneController(
            measure=measure,
            output_dir=output_dir,
            bounds=bounds,
            limits=limits,
        ).run()


@contextmanager
def autotune_phase(callback: PhaseTimingCallback | None) -> Iterator[None]:
    """Report the complete SDK autotune operation as one benchmark phase."""

    with benchmark_phase(callback, "autotune_s"):
        yield


class BenchmarkAutotuneController:
    """Own candidate deadlines, audit state, fallback, and final K/g selection."""

    def __init__(
        self,
        *,
        measure: MeasureFn,
        output_dir: Path,
        bounds: AutotuneBounds,
        limits: AutotuneLimits,
        clock: ClockFn | None = None,
    ) -> None:
        self.measure = measure
        self.output_dir = output_dir
        self.bounds = bounds
        self.limits = limits
        self.clock = clock or time.monotonic
        self.cache: dict[tuple[int, int], tuple[KTrial, int | None, dict[str, Any]]] = {}
        self.candidates: list[dict[str, Any]] = []
        self.bound_hint = BoundHint.GPU_INFER
        self.started_s = 0.0
        self.deadline_s = 0.0

    def run(self) -> dict[str, Any]:
        """Tune within budget and atomically persist every candidate transition."""

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.started_s = self.clock()
        self.deadline_s = self.started_s + self.limits.total_budget_s
        self._persist("running")
        try:
            result = self._tune()
        except Exception as exc:
            self._persist("failed", stop_reason=f"{type(exc).__name__}: {exc}")
            raise
        self._persist("complete", result=result)
        return result

    def _tune(self) -> dict[str, Any]:
        k_candidates = k_candidate_plan(self.bounds)
        k_trials, k_stop_reason = self._run_k_ramp(k_candidates)
        selected_k = select_k(k_trials, cpu_quota=self.bounds.cpu_cores).k
        selected_k_trial = next(trial for trial in k_trials if trial.k == selected_k)
        base_g = self.bounds.gpu_count
        replica_vram = self.cache[(selected_k, base_g)][1]
        g_candidates, g_trials, selected_g, g_ramp_reason = self._run_g_ramp(
            selected_k=selected_k,
            selected_k_trial=selected_k_trial,
            k_trials=k_trials,
            replica_vram=replica_vram,
        )
        return {
            "k": selected_k,
            "g": selected_g,
            "mode": "dynamic-probe",
            "bound_hint": self.bound_hint.value,
            "k_candidates": list(k_candidates),
            "g_candidates": list(g_candidates),
            "g_ramp_reason": g_ramp_reason,
            "k_stop_reason": k_stop_reason,
            "k_trials": [_trial_dict(trial) for trial in k_trials],
            "g_trials": [_trial_dict(trial) for trial in g_trials],
            "concurrency_health": self.cache[(selected_k, selected_g)][2],
            "bounds": {
                "gpu_count": self.bounds.gpu_count,
                "cpu_cores": self.bounds.cpu_cores,
                "max_workers": self.bounds.max_workers,
                "max_replicas": self.bounds.max_replicas,
                "worker_cpu_cores": self.bounds.worker_cpu_cores,
                "vram_per_gpu_bytes": self.bounds.vram_per_gpu_bytes,
                "replica_vram_bytes": replica_vram,
            },
        }

    def _run_k_ramp(self, candidates: tuple[int, ...]) -> tuple[list[KTrial], str | None]:
        trials: list[KTrial] = []
        stop_reason = None
        for k in candidates:
            try:
                trial = self._measure_trial(k, self.bounds.gpu_count, ramp="k")
            except (ProbeStopped, AutotuneBudgetExhausted) as exc:
                if not trials:
                    raise
                stop_reason = _stop_reason(exc)
                break
            trials.append(trial)
            selected = select_k(trials, cpu_quota=self.bounds.cpu_cores)
            if selected.k != k:
                stop_reason = f"K={k} did not improve safe throughput; keeping K={selected.k}"
                break
        return trials, stop_reason

    def _run_g_ramp(
        self,
        *,
        selected_k: int,
        selected_k_trial: KTrial,
        k_trials: list[KTrial],
        replica_vram: int | None,
    ) -> tuple[tuple[int, ...], list[KTrial], int, str]:
        base_g = self.bounds.gpu_count
        if replica_vram is None:
            reason = "policy replica VRAM footprint unknown; keeping one replica per GPU"
            return (base_g,), [], base_g, reason
        if self.bounds.vram_per_gpu_bytes is None:
            reason = "GPU VRAM capacity unknown; keeping one replica per GPU"
            return (base_g,), [], base_g, reason
        g_candidates = g_candidate_plan(self.bounds, replica_vram)
        single_throughput = k_trials[0].clients.throughput / max(k_trials[0].k, 1)
        if not should_ramp_g(
            bound_hint=self.bound_hint,
            k_trial=selected_k_trial,
            single_client_throughput=single_throughput,
        ):
            reason = "replica-ramp conditions not met; keeping one replica per GPU"
            return g_candidates, [], base_g, reason
        return self._measure_g_candidates(selected_k, g_candidates)

    def _measure_g_candidates(
        self,
        selected_k: int,
        candidates: tuple[int, ...],
    ) -> tuple[tuple[int, ...], list[KTrial], int, str]:
        trials: list[KTrial] = []
        selected_g = self.bounds.gpu_count
        reason = "GPU-inference bound; ramping replicas"
        for g in candidates:
            if g == self.bounds.gpu_count:
                continue
            try:
                trial = self._measure_trial(selected_k, g, ramp="g")
            except (ProbeStopped, AutotuneBudgetExhausted) as exc:
                reason = _stop_reason(exc)
                break
            trials.append(trial)
            previous = self.cache[(selected_k, selected_g)][0]
            gain = trial.clients.throughput / max(previous.clients.throughput, 1e-9) - 1
            if gain < 0.01:
                reason = f"replica throughput gain {gain:.1%} below 1% at g={g}"
                break
            selected_g = g
        return candidates, trials, selected_g, reason

    def _measure_trial(self, k: int, g: int, *, ramp: str) -> KTrial:
        key = (k, g)
        if key in self.cache:
            return self.cache[key][0]
        probe_limits = self._probe_limits()
        candidate = {
            "k": k,
            "g": g,
            "ramp": ramp,
            "status": "running",
            "elapsed_s": 0.0,
            "timeout_s": probe_limits.timeout_s,
        }
        self.candidates.append(candidate)
        self._persist("running")
        candidate_started = self.clock()
        try:
            output, replica_vram = self.measure(
                k,
                g,
                probe_limits,
                lambda progress: self._update_candidate_progress(
                    candidate,
                    candidate_started,
                    progress,
                ),
            )
            health = _concurrency_health(k, output)
            elapsed_s = self.clock() - candidate_started
            if elapsed_s >= probe_limits.timeout_s:
                raise ProbeTimedOut(
                    f"probe K={k}, g={g} exceeded {probe_limits.timeout_s:.3f}s",
                    stop_reason=probe_limits.timeout_reason,
                    health=health,
                )
            trial = _as_trial(k, g, output, health)
            self._check_correctness(k, g, health)
        except ProbeStopped as exc:
            self._finish_candidate(
                candidate,
                status=exc.status,
                started_s=candidate_started,
                stop_reason=exc.stop_reason,
                health=exc.health,
            )
            raise
        except Exception as exc:
            self._finish_candidate(
                candidate,
                status="error",
                started_s=candidate_started,
                stop_reason="measurement_error",
                error_type=type(exc).__name__,
                message=str(exc),
            )
            raise
        self.cache[key] = (trial, replica_vram, health)
        self.bound_hint = _bound_hint(output)
        self._finish_candidate(
            candidate,
            status="accepted",
            started_s=candidate_started,
            health=health,
            metrics=_trial_dict(trial),
        )
        return trial

    def _check_correctness(self, k: int, g: int, health: dict[str, Any]) -> None:
        baseline_key = (self.bounds.gpu_count, self.bounds.gpu_count)
        baseline = self.cache.get(baseline_key)
        if baseline is None or (k, g) == baseline_key:
            return
        baseline_health = baseline[2]
        baseline_rate = float(baseline_health["success_rate"])
        minimum_rate = baseline_rate * (1.0 - self.limits.maximum_success_rate_drop)
        health["baseline_k"] = baseline_key[0]
        health["baseline_successes"] = baseline_health["successes"]
        health["baseline_episodes"] = baseline_health["completed_episodes"]
        health["baseline_success_rate"] = baseline_rate
        health["minimum_success_rate"] = minimum_rate
        if float(health["success_rate"]) >= minimum_rate:
            return
        raise ProbeCorrectnessRegression(
            f"probe K={k}, g={g} success rate {health['success_rate']:.1%} fell below "
            f"the K={baseline_key[0]} baseline {baseline_rate:.1%} with "
            f"{self.limits.maximum_success_rate_drop:.1%} tolerance",
            health=health,
        )

    def _probe_limits(self) -> ProbeLimits:
        remaining_s = self.deadline_s - self.clock()
        if remaining_s <= 0:
            raise AutotuneBudgetExhausted(
                f"autotune total budget {self.limits.total_budget_s:.3f}s exhausted"
            )
        timeout_s = min(self.limits.candidate_timeout_s, remaining_s)
        timeout_reason = (
            "total_budget_exhausted"
            if remaining_s < self.limits.candidate_timeout_s
            else "candidate_timeout"
        )
        return ProbeLimits(timeout_s=timeout_s, timeout_reason=timeout_reason)

    def _update_candidate_progress(
        self,
        candidate: dict[str, Any],
        started_s: float,
        progress: dict[str, Any],
    ) -> None:
        candidate["elapsed_s"] = max(0.0, self.clock() - started_s)
        candidate["progress"] = {
            "completed_episodes": int(progress["episodes"]),
            "expected_episodes": int(progress["total_episodes"]),
            "successes": int(progress["successes"]),
            "failures": int(progress["failures"]),
        }
        self._persist("running")

    def _finish_candidate(
        self,
        candidate: dict[str, Any],
        *,
        status: str,
        started_s: float,
        stop_reason: str | None = None,
        health: dict[str, Any] | None = None,
        metrics: dict[str, Any] | None = None,
        error_type: str | None = None,
        message: str | None = None,
    ) -> None:
        candidate["status"] = status
        candidate["elapsed_s"] = max(0.0, self.clock() - started_s)
        if stop_reason is not None:
            candidate["stop_reason"] = stop_reason
        if health:
            candidate["health"] = dict(health)
        if metrics is not None:
            candidate["metrics"] = metrics
        if error_type is not None:
            candidate["error_type"] = error_type
        if message is not None:
            candidate["message"] = message
        self._persist("running")

    def _persist(
        self,
        status: str,
        *,
        stop_reason: str | None = None,
        result: dict[str, Any] | None = None,
    ) -> None:
        snapshot: dict[str, Any] = {
            "schema_version": 2,
            "status": status,
            "mode": "dynamic-probe",
            "elapsed_s": max(0.0, self.clock() - self.started_s),
            "limits": {
                "candidate_timeout_s": self.limits.candidate_timeout_s,
                "total_budget_s": self.limits.total_budget_s,
                "maximum_success_rate_drop": self.limits.maximum_success_rate_drop,
            },
            "candidates": self.candidates,
        }
        if stop_reason is not None:
            snapshot["stop_reason"] = stop_reason
        if result is not None:
            snapshot.update(result)
        atomic_json(self.output_dir / "selection.json", snapshot)


def _stop_reason(exc: ProbeStopped | AutotuneBudgetExhausted) -> str:
    if isinstance(exc, ProbeStopped):
        return f"{exc.stop_reason}: {exc}"
    return str(exc)


def _concurrency_health(k: int, output: dict[str, Any]) -> dict[str, Any]:
    recovery = output["recovery"]
    result = output["result"]
    return {
        "requested_workers": k,
        "initial_workers": sum(
            int(value) for value in recovery["initial_workers_per_gpu"].values()
        ),
        "final_workers": sum(
            int(value) for value in recovery["final_workers_per_gpu"].values()
        ),
        "worker_failures": len(recovery["worker_failures"]),
        "oom_events": int(recovery["oom_events"]),
        "completed_episodes": int(result["episodes"]),
        "expected_episodes": int(result["total_episodes"]),
        "successes": int(result["successes"]),
        "success_rate": float(result["success_rate"]),
    }


def _as_trial(
    k: int,
    g: int,
    output: dict[str, Any],
    health: dict[str, Any],
) -> KTrial:
    result = output["result"]
    resources = output["resources"]
    cpu = resources["cpu"]
    gpu = resources["gpu"]
    inference = output.get("inference_metrics", {})
    queue_wait = inference.get("queue_wait_ms") or {}
    if (
        health["initial_workers"] != k
        or health["final_workers"] != k
        or health["worker_failures"]
        or health["oom_events"]
        or health["completed_episodes"] != health["expected_episodes"]
    ):
        raise ProbeCapacityReached(
            f"probe K={k} ran with initial={health['initial_workers']}, "
            f"final={health['final_workers']}, worker_failures={health['worker_failures']}, "
            f"oom_events={health['oom_events']}, "
            f"completed_episodes={health['completed_episodes']}, "
            f"expected_episodes={health['expected_episodes']}",
            stop_reason=_health_stop_reason(k, health),
            health=health,
        )
    throughput = float(result["steps_per_s"])
    return KTrial(
        k=k,
        clients=ClientMetrics(
            active_clients=k,
            throughput=throughput,
            cpu_cores=float(cpu["cpu_cores"]),
            throttle_time_ratio=float(cpu["throttle_time_ratio"]),
            inference_wait_ratio=0.0,
        ),
        servers=ServerMetrics(
            replicas=g,
            throughput=throughput,
            queue_wait_p95_ms=float(queue_wait.get("p95") or 0.0),
            gpu_utilization=float(gpu["mean_utilization"]),
            vram_bytes=int(gpu["peak_memory_bytes"]),
        ),
    )


def _health_stop_reason(k: int, health: dict[str, Any]) -> str:
    if health["oom_events"]:
        return "worker_oom"
    if health["worker_failures"]:
        return "worker_failure"
    if health["initial_workers"] != k:
        return "worker_shortfall"
    if health["final_workers"] != k:
        return "worker_loss"
    return "incomplete_probe"


def _trial_dict(trial: KTrial) -> dict[str, Any]:
    return {
        "k": trial.k,
        "g": trial.servers.replicas,
        "throughput": trial.clients.throughput,
        "cpu_cores": trial.clients.cpu_cores,
        "throttle_time_ratio": trial.clients.throttle_time_ratio,
        "gpu_utilization": trial.servers.gpu_utilization,
        "vram_bytes": trial.servers.vram_bytes,
        "queue_wait_p95_ms": trial.servers.queue_wait_p95_ms,
    }


def _bound_hint(output: dict[str, Any]) -> BoundHint:
    raw = output.get("benchmark", {}).get("bound_hint", "gpu_infer")
    try:
        return BoundHint(str(raw))
    except ValueError:
        return BoundHint.GPU_INFER


__all__ = [
    "AutotuneBudgetExhausted",
    "AutotuneBounds",
    "AutotuneLimits",
    "BenchmarkAutotuneController",
    "ProbeCapacityReached",
    "ProbeLimits",
    "ProbeProgressFn",
    "ProbeStopped",
    "ProbeTimedOut",
    "autotune_phase",
    "g_candidate_plan",
    "k_candidate_plan",
    "resolve_bounds",
    "resolve_limits",
    "run_autotune",
]

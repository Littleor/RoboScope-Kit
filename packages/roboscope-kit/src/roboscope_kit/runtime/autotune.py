# Copyright 2026 RoboScope Team
# SPDX-License-Identifier: Apache-2.0

"""Platform-owned adaptive benchmark deployment and selection rules.

User policies do not call this module. RoboScope uses it to select evaluator
concurrency ``K`` and inference replica count ``g`` from measured end-to-end
throughput and resource signals.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Protocol  # noqa: UP035 - runtime also supports the CALVIN Python 3.8 environment


class BoundHint(str, Enum):  # noqa: UP042 - imported by the Python 3.10 LIBERO environment
    """Workload bottleneck classification supplied by the benchmark adapter."""

    CPU = "cpu"
    CPU_PHYSICS = "cpu_physics"
    GPU_RENDER = "gpu_render"
    GPU_INFER = "gpu_infer"


@dataclass(frozen=True)
class ScaleResult:
    """Result of scaling a server or client pool."""

    requested: int
    running: int


@dataclass(frozen=True)
class DrainResult:
    """Result of gracefully reducing a server or client pool."""

    requested: int
    drained: int
    timed_out: int = 0


@dataclass(frozen=True)
class ProbeMetrics:
    """Resource and throughput baseline measured with one client."""

    throughput: float
    cpu_cores: float
    vram_bytes: int
    inference_wait_ratio: float


@dataclass(frozen=True)
class ServerMetrics:
    """Aggregated policy-server metrics for one measurement window."""

    replicas: int
    throughput: float
    queue_wait_p95_ms: float
    gpu_utilization: float
    vram_bytes: int
    error_count: int = 0


@dataclass(frozen=True)
class ClientMetrics:
    """Aggregated evaluation-client metrics for one measurement window.

    ``throttle_time_ratio`` is cgroup throttled duration normalized by the
    available CPU core-seconds and is the signal used for K backoff.
    ``throttle_period_ratio`` is ``nr_throttled / nr_periods`` and is diagnostic
    only; it must not be substituted for the time ratio.
    """

    active_clients: int
    throughput: float
    cpu_cores: float
    throttle_time_ratio: float | None
    inference_wait_ratio: float
    throttle_period_ratio: float | None = None
    error_count: int = 0


@dataclass(frozen=True)
class KTrial:
    """Metrics collected for one client-concurrency candidate."""

    k: int
    clients: ClientMetrics
    servers: ServerMetrics


@dataclass(frozen=True)
class GTrial:
    """Metrics collected for one policy-server replica candidate."""

    g: int
    clients: ClientMetrics
    servers: ServerMetrics


@dataclass(frozen=True)
class KLimits:
    """Independent upper bounds used to calculate the safe K ceiling."""

    vram: int
    cpu: int
    queue: int

    @property
    def maximum(self) -> int:
        """Return the smallest upper bound."""

        return min(self.vram, self.cpu, self.queue)


class ServerPool(Protocol):
    """Control-plane contract required from a policy-server pool."""

    async def scale(self, target: int) -> ScaleResult:
        """Scale the pool to ``target`` replicas, which sets adaptive parameter g."""

        ...

    async def wait_ready(self, timeout_s: float) -> Sequence[str]:
        """Wait until all model replicas are loaded and accepting inference requests."""

        ...

    async def drain(self, server_ids: Sequence[str], timeout_s: float) -> DrainResult:
        """Stop routing new requests and wait for in-flight inference to finish."""

        ...

    async def stop(self, server_ids: Sequence[str]) -> None:
        """Stop the selected replicas and release their GPU resources."""

        ...

    async def metrics(self, window_s: float) -> ServerMetrics:
        """Measure inference throughput, queueing, GPU utilization, and errors."""

        ...


class ClientPool(Protocol):
    """Control-plane contract required from an evaluation-client pool."""

    async def scale(self, target: int) -> ScaleResult:
        """Scale the pool to ``target`` clients, which sets adaptive parameter K."""

        ...

    async def wait_ready(self, target: int, timeout_s: float) -> Sequence[str]:
        """Wait until ``target`` clients have initialized their simulation environments."""

        ...

    async def drain_to(self, target: int, timeout_s: float) -> DrainResult:
        """Let excess clients finish their current chunks before exiting."""

        ...

    async def stop(self, client_ids: Sequence[str], requeue_unfinished: bool = True) -> None:
        """Force-stop clients and optionally return unfinished chunks to the queue."""

        ...

    async def metrics(self, window_s: float) -> ClientMetrics:
        """Measure aggregate throughput, CPU throttle, inference wait, and errors."""

        ...

    async def probe_one(self, chunk_count: int) -> ProbeMetrics:
        """Measure the single-client throughput and resource baseline."""

        ...


def calculate_k_limits(
    *,
    gpu_count: int,
    vram_per_gpu_bytes: int,
    reserved_vram_per_gpu_bytes: int,
    server_replicas: int,
    server_vram_bytes: int,
    cpu_quota: float,
    reserved_cpu_cores: float,
    probe: ProbeMetrics,
    chunk_count: int,
) -> KLimits:
    """Calculate the VRAM, CPU, and available-work upper bounds for K."""

    available_vram = gpu_count * (vram_per_gpu_bytes - reserved_vram_per_gpu_bytes)
    available_vram -= server_replicas * server_vram_bytes
    k_vram = max(1, available_vram // max(probe.vram_bytes, 1))
    available_cpu_cores = max(cpu_quota - reserved_cpu_cores, 0.0)
    cpu_bound = available_cpu_cores / max(probe.cpu_cores, 1e-9)
    k_cpu = max(1, math.floor(cpu_bound + 1e-9))
    return KLimits(vram=int(k_vram), cpu=k_cpu, queue=max(1, chunk_count))


def build_k_candidates(base: int, maximum: int, *, growth_factor: float = 2.0) -> tuple[int, ...]:
    """Build a geometric K ramp that always reaches the calculated ceiling."""

    return _build_candidates(base, maximum, growth_factor=growth_factor)


def build_adaptive_k_candidates(base: int, hard_maximum: int) -> tuple[int, ...]:
    """Build a K ramp that probes beyond the single-client CPU estimate.

    VRAM and remaining work are hard limits. Probe ``base`` and ``2 * base``,
    then continue in half-base increments so live throughput and resource
    measurements decide where to stop. The single-client CPU estimate is
    reported for diagnostics but never used as a hard ramp boundary.
    """

    if hard_maximum < 1:
        raise ValueError("hard_maximum must be positive")
    first = min(base, hard_maximum)
    candidates = [first]
    second = min(2 * base, hard_maximum)
    if second > first:
        candidates.append(second)
    step = max(1, math.ceil(base / 2))
    current = candidates[-1]
    while current < hard_maximum:
        current = min(current + step, hard_maximum)
        candidates.append(current)
    return tuple(candidates)


def calculate_g_limit(
    *,
    gpu_count: int,
    vram_per_gpu_bytes: int,
    reserved_vram_per_gpu_bytes: int,
    server_vram_bytes: int,
) -> int:
    """Return the maximum replica count allowed by per-GPU VRAM placement."""

    if gpu_count < 1:
        raise ValueError("gpu_count must be positive")
    if server_vram_bytes < 1:
        raise ValueError("server_vram_bytes must be positive")
    available_per_gpu = max(vram_per_gpu_bytes - reserved_vram_per_gpu_bytes, 0)
    replicas_per_gpu = available_per_gpu // server_vram_bytes
    if replicas_per_gpu < 1:
        raise ValueError("a policy-server replica does not fit in the available per-GPU VRAM")
    return gpu_count * replicas_per_gpu


def build_g_candidates(base: int, maximum: int, *, growth_factor: float = 2.0) -> tuple[int, ...]:
    """Build a geometric policy-replica ramp up to the VRAM ceiling."""

    return _build_candidates(base, maximum, growth_factor=growth_factor)


def _build_candidates(base: int, maximum: int, *, growth_factor: float) -> tuple[int, ...]:
    if base < 1:
        raise ValueError("base must be positive")
    if maximum < 1:
        raise ValueError("maximum must be positive")
    if growth_factor <= 1:
        raise ValueError("growth_factor must be greater than one")

    current = min(base, maximum)
    candidates = [current]
    while current < maximum:
        next_candidate = max(current + 1, math.ceil(current * growth_factor))
        current = min(next_candidate, maximum)
        candidates.append(current)
    return tuple(candidates)


def select_k(
    trials: Sequence[KTrial],
    *,
    cpu_quota: float | None = None,
    reserved_cpu_cores: float = 1.0,
    minimum_gain: float = 0.01,
    gpu_limit_ratio: float = 0.95,
    throttle_limit_ratio: float = 0.40,
) -> KTrial:
    """Scale while throughput gains at least one percent and resources remain safe."""

    if not trials:
        raise ValueError("At least one K trial is required")

    selected = trials[0]
    for current in trials[1:]:
        previous_throughput = selected.clients.throughput
        gain = current.clients.throughput / max(previous_throughput, 1e-9) - 1
        cpu_limit_reached = cpu_quota is not None and current.clients.cpu_cores > max(
            cpu_quota - reserved_cpu_cores, 0.0
        )
        resource_limit_reached = (
            cpu_limit_reached
            or current.servers.gpu_utilization > gpu_limit_ratio
            or (
                current.clients.throttle_time_ratio is not None
                and current.clients.throttle_time_ratio > throttle_limit_ratio
            )
        )
        if gain < minimum_gain or resource_limit_reached:
            return selected
        selected = current
    return selected


def should_ramp_g(
    *,
    bound_hint: BoundHint,
    k_trial: KTrial,
    single_client_throughput: float,
    minimum_gpu_utilization: float = 0.40,
    maximum_gpu_utilization: float = 0.70,
    minimum_linear_efficiency: float = 0.80,
) -> bool:
    """Return whether server replicas should be tested after K has been selected."""

    gpu_utilization = k_trial.servers.gpu_utilization
    ideal_throughput = k_trial.k * single_client_throughput
    return (
        bound_hint is BoundHint.GPU_INFER
        and minimum_gpu_utilization <= gpu_utilization <= maximum_gpu_utilization
        and k_trial.clients.throughput < minimum_linear_efficiency * ideal_throughput
    )


def select_g(trials: Sequence[GTrial], *, minimum_gain: float = 0.01) -> GTrial:
    """Scale while each additional replica improves throughput by at least one percent."""

    if not trials:
        raise ValueError("At least one g trial is required")

    selected = trials[0]
    for current in trials[1:]:
        gain = current.clients.throughput / max(selected.clients.throughput, 1e-9) - 1
        if gain < minimum_gain:
            return selected
        selected = current
    return selected


@dataclass(frozen=True)
class AutotunePlan:
    """Ordered concurrency and inference-replica candidates to measure."""

    k_candidates: tuple[int, ...]
    g_candidates: tuple[int, ...]
    cpu_quota: float | None = None
    reserved_cpu_cores: float = 1.0
    minimum_gain: float = 0.01

    def __post_init__(self) -> None:
        _validate_candidates("k", self.k_candidates)
        _validate_candidates("g", self.g_candidates)
        if self.reserved_cpu_cores < 0:
            raise ValueError("reserved_cpu_cores cannot be negative")
        if self.minimum_gain < 0:
            raise ValueError("minimum_gain cannot be negative")


@dataclass(frozen=True)
class AutotuneSelection:
    """Measured trials and the selected parallel runtime configuration."""

    k: int
    g: int
    k_trials: tuple[KTrial, ...]
    g_trials: tuple[GTrial, ...]


KMeasurement = Callable[[int, int], KTrial]


class AutotuneController:
    """Select platform concurrency and inference replicas from measured trials."""

    def __init__(self, plan: AutotunePlan):
        self.plan = plan

    def tune(self, measure: KMeasurement) -> AutotuneSelection:
        initial_g = self.plan.g_candidates[0]
        k_trials = self._measure_k_candidates(measure, initial_g)
        selected_k = self._selected_k(k_trials).k
        g_trials = self._measure_g_candidates(measure, selected_k)
        selected_g = select_g(g_trials, minimum_gain=self.plan.minimum_gain).g
        return AutotuneSelection(
            k=selected_k,
            g=selected_g,
            k_trials=tuple(k_trials),
            g_trials=tuple(g_trials),
        )

    def _measure_k_candidates(self, measure: KMeasurement, initial_g: int) -> list[KTrial]:
        trials: list[KTrial] = []
        for k in self.plan.k_candidates:
            trial = measure(k, initial_g)
            if trial.k != k or trial.servers.replicas != initial_g:
                raise ValueError("measurement callback returned metrics for a different K/g candidate")
            trials.append(trial)
            if self._selected_k(trials).k != k:
                break
        return trials

    def _measure_g_candidates(self, measure: KMeasurement, selected_k: int) -> list[GTrial]:
        trials: list[GTrial] = []
        for g in self.plan.g_candidates:
            measured = measure(selected_k, g)
            if measured.k != selected_k or measured.servers.replicas != g:
                raise ValueError("measurement callback returned metrics for a different K/g candidate")
            trials.append(GTrial(g=g, clients=measured.clients, servers=measured.servers))
            if select_g(trials, minimum_gain=self.plan.minimum_gain).g != g:
                break
        return trials

    def _selected_k(self, trials: Sequence[KTrial]) -> KTrial:
        return select_k(
            trials,
            cpu_quota=self.plan.cpu_quota,
            reserved_cpu_cores=self.plan.reserved_cpu_cores,
            minimum_gain=self.plan.minimum_gain,
        )


def _validate_candidates(name: str, candidates: tuple[int, ...]) -> None:
    if not candidates:
        raise ValueError(f"{name}_candidates cannot be empty")
    if any(candidate < 1 for candidate in candidates):
        raise ValueError(f"{name}_candidates must be positive")
    if list(candidates) != sorted(set(candidates)):
        raise ValueError(f"{name}_candidates must be unique and increasing")

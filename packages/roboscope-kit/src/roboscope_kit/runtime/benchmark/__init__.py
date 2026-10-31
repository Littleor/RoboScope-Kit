# Copyright 2026 RoboScope Team
# SPDX-License-Identifier: Apache-2.0

"""Benchmark runtime for autotuned, parallel benchmark execution."""

from .adapters import (
    BenchmarkAdapter,
    InferenceAdapter,
    ProcessSpec,
    ReplicaContext,
    WorkerContext,
)
from .autotune import autotune_phase
from .factories import BenchmarkFactory, BenchmarkRunRequest
from .manifest import BenchmarkManifest, load_benchmark_manifest
from .orchestrator import ParallelBenchmarkRuntime, RuntimeConfig, aggregate_standard_results
from .policy_process import PolicyProcessAdapter
from .processes import InferencePool
from .queue import ClaimedWork, WorkQueue, atomic_json
from .resources import CpuStat, GpuSampler, cpu_quota, cpu_window, gpu_window, read_cpu_stat
from .schema import EvaluationRecord, WorkItem, WorkResult
from .timing import BenchmarkTimingRecorder, PhaseTimingCallback, benchmark_phase
from .worker import WorkerLoop

__all__ = [
    "BenchmarkAdapter",
    "BenchmarkFactory",
    "BenchmarkManifest",
    "BenchmarkRunRequest",
    "BenchmarkTimingRecorder",
    "ClaimedWork",
    "CpuStat",
    "EvaluationRecord",
    "InferenceAdapter",
    "GpuSampler",
    "InferencePool",
    "ParallelBenchmarkRuntime",
    "PhaseTimingCallback",
    "PolicyProcessAdapter",
    "ProcessSpec",
    "ReplicaContext",
    "RuntimeConfig",
    "WorkItem",
    "WorkQueue",
    "WorkResult",
    "WorkerContext",
    "WorkerLoop",
    "aggregate_standard_results",
    "autotune_phase",
    "benchmark_phase",
    "cpu_quota",
    "cpu_window",
    "gpu_window",
    "read_cpu_stat",
    "atomic_json",
    "load_benchmark_manifest",
]

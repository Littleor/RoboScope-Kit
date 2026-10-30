# Copyright 2026 RoboScope Team
# SPDX-License-Identifier: Apache-2.0

"""Host resource measurements shared by all benchmark adapters."""

from __future__ import annotations

import os
import resource
import statistics
import subprocess
import threading
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CpuStat:
    usage_usec: int
    nr_periods: int
    nr_throttled: int
    throttled_usec: int


class GpuSampler:
    def __init__(self, interval_s: float = 0.5, gpu_devices: tuple[str, ...] | None = None):
        self._interval_s = interval_s
        self._gpu_devices = set(gpu_devices) if gpu_devices is not None else None
        self._samples: list[list[dict[str, float | str]]] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="gpu-sampler", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> list[list[dict[str, float | str]]]:
        self._stop.set()
        self._thread.join(timeout=5)
        return self._samples

    def _run(self) -> None:
        while not self._stop.is_set():
            self._sample_once()
            self._stop.wait(self._interval_s)

    def _sample_once(self) -> None:
        command = [
            "nvidia-smi",
            "--query-gpu=index,uuid,utilization.gpu,memory.used",
            "--format=csv,noheader,nounits",
        ]
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                check=False,
                timeout=10,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return
        if result.returncode != 0:
            return
        sample = []
        for line in result.stdout.splitlines():
            index, uuid, utilization, memory_mib = [item.strip() for item in line.split(",")]
            if self._gpu_devices is not None and not {index, uuid} & self._gpu_devices:
                continue
            sample.append(
                {
                    "device": index if self._gpu_devices is None or index in self._gpu_devices else uuid,
                    "utilization": float(utilization) / 100,
                    "memory_bytes": float(memory_mib) * 1024**2,
                }
            )
        if sample:
            self._samples.append(sample)


def read_cpu_stat(path: Path = Path("/sys/fs/cgroup/cpu.stat")) -> CpuStat:
    if not path.is_file():
        usage = resource.getrusage(resource.RUSAGE_SELF)
        usage_usec = int((usage.ru_utime + usage.ru_stime) * 1e6)
        return CpuStat(usage_usec, 0, 0, 0)
    values = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        key, value = line.split()
        values[key] = int(value)
    return CpuStat(
        usage_usec=values.get("usage_usec", 0),
        nr_periods=values.get("nr_periods", 0),
        nr_throttled=values.get("nr_throttled", 0),
        throttled_usec=values.get("throttled_usec", 0),
    )


def cpu_quota(path: Path = Path("/sys/fs/cgroup/cpu.max")) -> float:
    if not path.is_file():
        return float(os.cpu_count() or 1)
    quota, period = path.read_text(encoding="utf-8").split()
    if quota == "max":
        return float(os.cpu_count() or 1)
    return int(quota) / int(period)


def cpu_window(before: CpuStat, after: CpuStat, elapsed_s: float, quota_cores: float) -> dict:
    usage_s = (after.usage_usec - before.usage_usec) / 1e6
    throttled_s = (after.throttled_usec - before.throttled_usec) / 1e6
    periods = after.nr_periods - before.nr_periods
    throttled_periods = after.nr_throttled - before.nr_throttled
    entitlement = elapsed_s * quota_cores
    return {
        "cpu_cores": usage_s / max(elapsed_s, 1e-9),
        "throttle_time_ratio": throttled_s / max(entitlement, 1e-9),
        "throttle_period_ratio": throttled_periods / max(periods, 1),
    }


def gpu_window(samples: list[list[dict[str, float | str]]], replicas: int) -> dict[str, object]:
    active_samples = [sample[:replicas] for sample in samples if sample]
    utilizations = [gpu["utilization"] for sample in active_samples for gpu in sample]
    memory_totals = [sum(gpu["memory_bytes"] for gpu in sample) for sample in active_samples]
    utilization_by_gpu: dict[str, list[float]] = {}
    memory_by_gpu: dict[str, list[float]] = {}
    for sample in active_samples:
        for gpu in sample:
            device = str(gpu["device"])
            utilization_by_gpu.setdefault(device, []).append(gpu["utilization"])
            memory_by_gpu.setdefault(device, []).append(gpu["memory_bytes"])
    return {
        "mean_utilization": statistics.fmean(utilizations) if utilizations else 0.0,
        "peak_memory_bytes": max(memory_totals, default=0.0),
        "gpu_util_avg": {
            str(index): 100 * statistics.fmean(values) for index, values in utilization_by_gpu.items()
        },
        "gpu_util_max": {str(index): 100 * max(values) for index, values in utilization_by_gpu.items()},
        "gpu_mem_mib_max": {str(index): max(values) / 1024**2 for index, values in memory_by_gpu.items()},
    }

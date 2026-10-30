# Copyright 2026 RoboScope Team
# SPDX-License-Identifier: Apache-2.0

import numpy as np
from benchmarks.libero_official.starvla_runtime.libero_client import (
    episode_terminated,
    reset_episode,
)
from benchmarks.libero_official.starvla_runtime.libero_controller import (
    aggregate_client_profile,
    percentile,
)

from roboscope_kit.runtime.benchmark.resources import (
    CpuStat,
    cpu_window,
    gpu_window,
)


class FakeClient:
    def __init__(self) -> None:
        self.reset_calls = 0

    def reset(self, *, task_description: str) -> None:
        assert task_description == "pick up the object"
        self.reset_calls += 1


class FakeEnv:
    def __init__(self) -> None:
        self.env = self
        self.done = False
        self.active_state = -1
        self.steps = 0

    def reset(self) -> None:
        self.done = False
        self.steps = 0

    def set_init_state(self, initial_state: np.ndarray) -> dict:
        self.active_state = int(initial_state[0])
        return {"state": self.active_state}

    def step(self, action: list[float]) -> tuple[dict, float, bool, dict]:
        assert action == [0.0] * 6 + [-1.0]
        self.steps += 1
        terminated = self.active_state == 1
        return {"state": self.active_state, "step": self.steps}, 0.0, terminated, {}


def test_cpu_window_separates_throttled_time_and_period_ratios() -> None:
    before = CpuStat(usage_usec=1_000_000, nr_periods=100, nr_throttled=10, throttled_usec=200_000)
    after = CpuStat(usage_usec=21_000_000, nr_periods=200, nr_throttled=60, throttled_usec=4_200_000)

    metrics = cpu_window(before, after, elapsed_s=10, quota_cores=1)

    assert metrics == {
        "cpu_cores": 2.0,
        "throttle_time_ratio": 0.4,
        "throttle_period_ratio": 0.5,
    }


def test_cpu_window_normalizes_throttled_time_by_cpu_entitlement() -> None:
    before = CpuStat(usage_usec=0, nr_periods=0, nr_throttled=0, throttled_usec=0)
    after = CpuStat(
        usage_usec=560_000_000,
        nr_periods=100,
        nr_throttled=56,
        throttled_usec=320_000_000,
    )

    metrics = cpu_window(before, after, elapsed_s=10, quota_cores=80)

    assert metrics["cpu_cores"] == 56
    assert metrics["throttle_time_ratio"] == 0.4
    assert metrics["throttle_period_ratio"] == 0.56


def test_gpu_window_uses_active_replicas_only() -> None:
    samples = [
        [
            {"device": "0", "utilization": 0.4, "memory_bytes": 10.0},
            {"device": "1", "utilization": 0.6, "memory_bytes": 11.0},
            {"device": "2", "utilization": 0.9, "memory_bytes": 99.0},
        ],
        [
            {"device": "0", "utilization": 0.6, "memory_bytes": 12.0},
            {"device": "1", "utilization": 0.8, "memory_bytes": 13.0},
            {"device": "2", "utilization": 1.0, "memory_bytes": 99.0},
        ],
    ]

    metrics = gpu_window(samples, replicas=2)

    assert metrics == {
        "mean_utilization": 0.6,
        "peak_memory_bytes": 25.0,
        "gpu_util_avg": {"0": 50.0, "1": 70.0},
        "gpu_util_max": {"0": 60.0, "1": 80.0},
        "gpu_mem_mib_max": {
            "0": 12.0 / 1024**2,
            "1": 13.0 / 1024**2,
        },
    }


def test_percentile_handles_short_measurement_windows() -> None:
    assert percentile([], 0.95) == 0
    assert percentile([30, 10, 20], 0.50) == 20
    assert percentile([30, 10, 20], 0.95) == 30


def test_aggregate_client_profile_separates_worker_cpu_from_cgroup_cpu() -> None:
    results = [
        {
            "elapsed_s": 10.0,
            "process_user_cpu_s": 1.5,
            "process_system_cpu_s": 0.5,
            "observation_build_s": 1.0,
            "client_step_s": 2.0,
            "action_decode_s": 0.5,
            "env_step_s": 5.0,
            "reset_s": 1.5,
        },
        {
            "elapsed_s": 10.0,
            "process_user_cpu_s": 1.5,
            "process_system_cpu_s": 0.5,
            "observation_build_s": 1.0,
            "client_step_s": 2.0,
            "action_decode_s": 0.5,
            "env_step_s": 5.0,
            "reset_s": 1.5,
        },
    ]

    profile = aggregate_client_profile(results, elapsed_s=10.0, cgroup_cpu_cores=8.0)

    assert profile["process_cpu_s"] == 4.0
    assert profile["process_cpu_cores"] == 0.4
    assert profile["process_cpu_cores_per_worker"] == 0.2
    assert profile["non_client_cgroup_cpu_cores"] == 7.6
    assert profile["stage_time_s"]["env_step_s"] == 10.0
    assert profile["unaccounted_client_wall_s"] == 0.0


def test_reset_episode_skips_initial_state_that_terminates_during_settling() -> None:
    env = FakeEnv()
    client = FakeClient()

    observation, episode_index = reset_episode(
        env,
        client,
        "pick up the object",
        np.asarray([[1], [2]]),
        episode_index=0,
        settle_steps=2,
    )

    assert episode_index == 1
    assert observation == {"state": 2, "step": 2}
    assert client.reset_calls == 2


def test_episode_terminated_includes_masked_robosuite_horizon() -> None:
    env = FakeEnv()
    env.done = True

    assert episode_terminated(env, task_done=False)

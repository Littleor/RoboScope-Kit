# Copyright 2026 RoboScope Team
# SPDX-License-Identifier: Apache-2.0

from roboscope_kit.runtime.autotune import (
    BoundHint,
    ClientMetrics,
    GTrial,
    KTrial,
    ProbeMetrics,
    ServerMetrics,
    build_adaptive_k_candidates,
    build_g_candidates,
    build_k_candidates,
    calculate_g_limit,
    calculate_k_limits,
    select_g,
    select_k,
    should_ramp_g,
)


def _k_trial(
    k: int,
    throughput: float,
    *,
    cpu_cores: float = 0.5,
    throttle: float = 0.0,
    throttle_period: float = 0.0,
    gpu_utilization: float = 0.6,
) -> KTrial:
    return KTrial(
        k=k,
        clients=ClientMetrics(
            active_clients=k,
            throughput=throughput,
            cpu_cores=cpu_cores,
            throttle_time_ratio=throttle,
            throttle_period_ratio=throttle_period,
            inference_wait_ratio=0.4,
        ),
        servers=ServerMetrics(
            replicas=4,
            throughput=throughput,
            queue_wait_p95_ms=10,
            gpu_utilization=gpu_utilization,
            vram_bytes=36 * 1024**3,
        ),
    )


def _g_trial(g: int, throughput: float) -> GTrial:
    k_trial = _k_trial(8, throughput)
    return GTrial(g=g, clients=k_trial.clients, servers=k_trial.servers)


def test_calculate_k_limits_uses_smallest_bound() -> None:
    probe = ProbeMetrics(throughput=20, cpu_cores=6, vram_bytes=1024**3, inference_wait_ratio=0.2)

    limits = calculate_k_limits(
        gpu_count=4,
        vram_per_gpu_bytes=48 * 1024**3,
        reserved_vram_per_gpu_bytes=2 * 1024**3,
        server_replicas=4,
        server_vram_bytes=9 * 1024**3,
        cpu_quota=80,
        reserved_cpu_cores=1,
        probe=probe,
        chunk_count=100,
    )

    assert limits.cpu == 13
    assert limits.maximum == 13


def test_calculate_k_limits_does_not_round_exact_cpu_bound_down() -> None:
    probe = ProbeMetrics(throughput=23, cpu_cores=2.8, vram_bytes=512 * 1024**2, inference_wait_ratio=0.3)

    limits = calculate_k_limits(
        gpu_count=4,
        vram_per_gpu_bytes=48 * 1024**3,
        reserved_vram_per_gpu_bytes=2 * 1024**3,
        server_replicas=4,
        server_vram_bytes=9 * 1024**3,
        cpu_quota=48,
        reserved_cpu_cores=1,
        probe=probe,
        chunk_count=128,
    )

    assert limits.cpu == 16


def test_build_k_candidates_reaches_high_non_power_of_two_maximum() -> None:
    assert build_k_candidates(4, 48) == (4, 8, 16, 32, 48)


def test_build_k_candidates_clamps_base_to_small_maximum() -> None:
    assert build_k_candidates(4, 3) == (3,)


def test_adaptive_k_candidates_continue_beyond_soft_cpu_limit() -> None:
    assert build_adaptive_k_candidates(4, 14) == (4, 8, 10, 12, 14)


def test_g_candidates_use_per_gpu_vram_limit() -> None:
    maximum = calculate_g_limit(
        gpu_count=4,
        vram_per_gpu_bytes=48 * 1024**3,
        reserved_vram_per_gpu_bytes=2 * 1024**3,
        server_vram_bytes=9 * 1024**3,
    )

    assert maximum == 20
    assert build_g_candidates(4, maximum) == (4, 8, 16, 20)


def test_select_k_keeps_scaling_above_one_percent_gain() -> None:
    selected = select_k([_k_trial(4, 100), _k_trial(8, 160), _k_trial(12, 170)])

    assert selected.k == 12


def test_select_k_returns_previous_candidate_when_throughput_drops() -> None:
    selected = select_k([_k_trial(4, 100), _k_trial(8, 160), _k_trial(12, 159)])

    assert selected.k == 8


def test_select_k_returns_previous_candidate_below_one_percent_gain() -> None:
    selected = select_k([_k_trial(4, 100), _k_trial(8, 160), _k_trial(12, 161)])

    assert selected.k == 8


def test_select_k_returns_previous_candidate_before_throttle_limit() -> None:
    selected = select_k([_k_trial(4, 100), _k_trial(8, 150), _k_trial(12, 190, throttle=0.5)])

    assert selected.k == 8


def test_select_k_ignores_throttle_period_ratio_for_backoff() -> None:
    selected = select_k(
        [_k_trial(4, 100), _k_trial(8, 150, throttle_period=0.9)],
    )

    assert selected.k == 8


def test_select_k_returns_previous_candidate_before_gpu_limit() -> None:
    selected = select_k(
        [_k_trial(4, 100), _k_trial(8, 160), _k_trial(16, 220, gpu_utilization=0.97)],
    )

    assert selected.k == 8


def test_select_k_reserves_one_cpu_core() -> None:
    selected = select_k(
        [_k_trial(4, 100, cpu_cores=60), _k_trial(8, 160, cpu_cores=79.5)],
        cpu_quota=80,
    )

    assert selected.k == 4


def test_should_ramp_g_for_gpu_inference_supply_bottleneck() -> None:
    assert should_ramp_g(
        bound_hint=BoundHint.GPU_INFER,
        k_trial=_k_trial(8, 120),
        single_client_throughput=25,
    )


def test_select_g_keeps_scaling_above_one_percent_gain() -> None:
    selected = select_g([_g_trial(4, 120), _g_trial(8, 170), _g_trial(12, 176)])

    assert selected.g == 12


def test_select_g_returns_previous_replica_count_when_throughput_drops() -> None:
    selected = select_g([_g_trial(4, 120), _g_trial(8, 170), _g_trial(12, 169)])

    assert selected.g == 8


def test_select_g_returns_previous_replica_count_below_one_percent_gain() -> None:
    selected = select_g([_g_trial(4, 120), _g_trial(8, 170), _g_trial(12, 171)])

    assert selected.g == 8

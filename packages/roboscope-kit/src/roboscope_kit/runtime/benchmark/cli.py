# Copyright 2026 RoboScope Team
# SPDX-License-Identifier: Apache-2.0

"""Unified command-line runtime for every RoboScope benchmark."""

from __future__ import annotations

import argparse
import os
import subprocess
from dataclasses import replace
from pathlib import Path
from typing import Any

from roboscope_kit.benchmark_specs import (
    BenchmarkName,
    BenchmarkSpec,
    MetaworldBenchmarkDefinition,
    get_benchmark_spec,
)
from roboscope_kit.policy_config import PolicyRuntimeConfig, load_policy_runtime_config
from roboscope_kit.runtime.benchmarks.metaworld.config import (
    load_evaluation_config,
    resolve_evaluation_config,
)

from .autotune import (
    ProbeLimits,
    ProbeProgressFn,
    autotune_phase,
    autotune_strategies,
    resolve_bounds,
    resolve_limits,
    run_autotune,
)
from .cuda_mps import CudaMpsSession
from .factories import BenchmarkFactory, BenchmarkRunRequest
from .manifest import (
    BenchmarkManifest,
    load_benchmark_manifest,
)
from .orchestrator import ParallelBenchmarkRuntime, RuntimeConfig, append_event
from .policy_process import PolicyProcessAdapter
from .queue import atomic_json
from .registry import benchmark_factory
from .sharding import DEFAULT_EPISODES_PER_SHARD, shard_build
from .timing import BenchmarkTimingRecorder, PhaseTimingCallback, benchmark_phase


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", choices=tuple(item.value for item in BenchmarkName), required=True)
    parser.add_argument("--suite")
    parser.add_argument("--checkpoint-source", type=Path, help="absolute path to a local checkpoint")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--episodes", type=int)
    parser.add_argument("--quick-test", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument(
        "--cuda-mps",
        action="store_true",
        help="run the Policy replicas under a private NVIDIA CUDA MPS daemon (default: off)",
    )
    parser.add_argument(
        "--shard-index",
        type=int,
        help="run only the work items owned by shard INDEX (cross-instance sharding)",
    )
    parser.add_argument(
        "--shard-count",
        type=int,
        help="total shard count; omit to derive it from the workload size (one shard per "
        "--shard-episodes episodes)",
    )
    parser.add_argument(
        "--shard-episodes",
        type=int,
        default=DEFAULT_EPISODES_PER_SHARD,
        help="episode budget per shard when --shard-count is omitted",
    )
    parser.add_argument("--benchmark-manifest", type=Path, required=True)
    parser.add_argument("--policy-config", type=Path, help="Policy process contract (roboscope-policy.json)")
    parser.add_argument("--evaluation-config", type=Path)
    parser.add_argument(
        "--autotune-strategy",
        help=f"K/g autotune strategy, one of {', '.join(autotune_strategies())}; overrides the "
        "manifest's autotune_strategy",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        metavar="K",
        help="evaluate with K benchmark Workers instead of autotuning K/g; requires --replicas",
    )
    parser.add_argument(
        "--replicas",
        type=int,
        metavar="G",
        help="evaluate with G Policy replicas instead of autotuning K/g; requires --concurrency",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    errors_path = args.output_dir / "errors.jsonl"
    errors_path.unlink(missing_ok=True)
    timing_recorder = None
    try:
        benchmark = BenchmarkName(args.benchmark)
        if args.autotune_strategy not in (None, *autotune_strategies()):
            raise ValueError(f"--autotune-strategy must be one of {autotune_strategies()}")
        manifest = load_benchmark_manifest(args.benchmark_manifest)
        if args.autotune_strategy is not None:
            manifest = replace(
                manifest,
                defaults={**manifest.defaults, "autotune_strategy": args.autotune_strategy},
            )
        factory = benchmark_factory(benchmark)
        if args.self_test:
            manifest.validate_request(benchmark, None)
            result = factory.self_test(manifest, args.output_dir)
            atomic_json(args.output_dir / "self-test.json", result)
            return 0
        timing_recorder = BenchmarkTimingRecorder(args.output_dir / "metrics.json")
        return _run_evaluation(
            args,
            benchmark,
            manifest,
            factory,
            timing_recorder.handle_phase,
        )
    except Exception as exc:
        append_event(
            errors_path,
            {
                "type": "evaluation-error",
                "component": "runtime",
                "error_type": type(exc).__name__,
                "message": str(exc),
            },
        )
        raise
    finally:
        if timing_recorder is not None:
            timing_recorder.close()


def _run_evaluation(
    args: argparse.Namespace,
    benchmark: BenchmarkName,
    manifest,
    factory: BenchmarkFactory,
    phase_timing_callback: PhaseTimingCallback,
) -> int:
    if not args.suite:
        raise ValueError("--suite is required for benchmark evaluation")
    if not args.checkpoint_source:
        raise ValueError("--checkpoint-source is required for benchmark evaluation")
    if args.policy_config is None:
        raise ValueError("--policy-config is required for benchmark evaluation")
    if args.episodes is not None and args.episodes < 1:
        raise ValueError("--episodes must be positive")
    if args.quick_test and args.shard_index is not None:
        raise ValueError("--quick-test cannot be combined with sharding")
    fixed = args.concurrency is not None or args.replicas is not None
    if fixed:
        if args.concurrency is None or args.replicas is None:
            raise ValueError("--concurrency and --replicas must be given together")
        if args.concurrency < 1 or args.replicas < 1:
            raise ValueError("--concurrency and --replicas must be positive")
        if args.quick_test:
            raise ValueError("--quick-test runs at K=1/g=1 and cannot be combined with --concurrency")
        if args.autotune_strategy is not None:
            raise ValueError("--autotune-strategy selects a search and cannot be combined with a fixed K/g")
    if args.shard_count is not None and args.shard_index is None:
        raise ValueError("--shard-count requires --shard-index")
    if args.shard_index is not None and args.shard_index < 0:
        raise ValueError("--shard-index must be non-negative")
    if args.shard_count is not None and args.shard_count < 1:
        raise ValueError("--shard-count must be positive")
    if args.shard_episodes < 1:
        raise ValueError("--shard-episodes must be positive")
    manifest.validate_request(benchmark, args.suite)
    factory.validate_suite(args.suite)
    with benchmark_phase(phase_timing_callback, "checkpoint_prepare_s"):
        checkpoint = args.checkpoint_source
        if not checkpoint.is_absolute():
            raise ValueError("checkpoint source must be an absolute local path")
        if not checkpoint.exists():
            raise FileNotFoundError(f"checkpoint path does not exist: {checkpoint}")
    policy = load_policy_runtime_config(args.policy_config)
    evaluation_config, evaluation_config_path = _resolve_evaluation_config(
        benchmark,
        args.evaluation_config,
        policy,
        args.output_dir,
    )
    benchmark_spec = _build_benchmark_spec(benchmark, args.suite, evaluation_config)
    request = BenchmarkRunRequest(
        benchmark=benchmark,
        suite=args.suite,
        checkpoint_path=checkpoint,
        output_dir=args.output_dir,
        episodes=args.episodes,
        quick_test=args.quick_test,
        evaluation_config_path=evaluation_config_path,
    )
    gpu_devices = detect_gpu_devices()
    if not gpu_devices:
        raise RuntimeError("benchmark runtime requires at least one visible GPU")
    shard_kwargs = _shard_kwargs(args, manifest)
    cuda_mps: CudaMpsSession | None = None
    try:
        if args.cuda_mps:
            # Autotune probes and the formal pass both run under MPS, so K/g is
            # selected for the configuration that is evaluated.
            cuda_mps = CudaMpsSession(gpu_devices=gpu_devices, log_dir=args.output_dir / "cuda-mps")
            cuda_mps.start()
        if args.quick_test or fixed:
            if args.quick_test:
                selected = {"k": 1, "g": 1, "mode": "quick-test"}
            else:
                selected = {"k": args.concurrency, "g": args.replicas, "mode": "fixed"}
            with autotune_phase(phase_timing_callback):
                pass
            output = run_candidate(
                factory=factory,
                request=request,
                manifest=manifest,
                policy=policy,
                benchmark_spec=benchmark_spec,
                gpu_devices=gpu_devices,
                concurrency=int(selected["k"]),
                replicas=int(selected["g"]),
                output_dir=args.output_dir,
                probe=False,
                phase_timing_callback=phase_timing_callback,
                shard_kwargs=shard_kwargs,
                cuda_mps=cuda_mps,
            )
        else:
            output, selected = _autotuned_run(
                factory,
                request,
                manifest,
                policy,
                benchmark_spec,
                gpu_devices,
                args.output_dir,
                phase_timing_callback,
                shard_kwargs=shard_kwargs,
                cuda_mps=cuda_mps,
            )
    finally:
        if cuda_mps is not None:
            cuda_mps.stop()
    output["autotune"] = selected
    output["cuda_mps"] = cuda_mps.metadata() if cuda_mps is not None else {"enabled": False}
    atomic_json(args.output_dir / "result.json", output)
    return 0


def _resolve_evaluation_config(
    benchmark: BenchmarkName,
    path: Path | None,
    policy: PolicyRuntimeConfig,
    output_dir: Path,
) -> tuple[dict[str, Any] | None, Path | None]:
    """Resolve the benchmark evaluation config once and materialize it for workers.

    Priority is an explicit ``--evaluation-config`` file, then the Policy file's
    ``benchmark_config`` declaration, then the default. Every worker and
    the Policy ``load`` spec therefore consume the same resolved object.
    """

    if benchmark is not BenchmarkName.METAWORLD:
        if path is not None:
            raise ValueError("--evaluation-config is only supported for the metaworld benchmark")
        return None, None
    declared = policy.benchmark_config.get("metaworld")
    config = load_evaluation_config(path) if path is not None else resolve_evaluation_config(declared)
    target = output_dir / "evaluation-config.json"
    atomic_json(target, config)
    return config, target


def _build_benchmark_spec(
    benchmark: BenchmarkName,
    suite: str,
    evaluation_config: dict[str, Any] | None,
) -> BenchmarkSpec:
    if benchmark is BenchmarkName.METAWORLD:
        if evaluation_config is None:
            raise ValueError("metaworld evaluation config was not resolved")
        return MetaworldBenchmarkDefinition().spec(suite, tuple(evaluation_config["cameras"]))
    return get_benchmark_spec(benchmark, suite)


def _autotuned_run(
    factory: BenchmarkFactory,
    request: BenchmarkRunRequest,
    manifest,
    policy,
    benchmark_spec: BenchmarkSpec,
    gpu_devices: tuple[str, ...],
    output_dir: Path,
    phase_timing_callback: PhaseTimingCallback,
    progress_callback=None,
    shard_kwargs: dict[str, Any] | None = None,
    cuda_mps: CudaMpsSession | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load the policy pool once, probe candidates against it, then run for real.

    The pool starts with one replica per GPU. K probes reuse that pool, while g
    probes grow it incrementally so every candidate measures the requested
    replica count without reloading replicas that are already running.
    Autotune probes always run the unsharded workload; sharding applies to the
    final pass only.
    """

    gpu_count = len(gpu_devices)
    defaults = dict(manifest.defaults)
    if "vram_per_gpu_bytes" not in defaults:
        vram_per_gpu_bytes = detect_gpu_memory_bytes(gpu_devices)
        if vram_per_gpu_bytes is not None:
            defaults["vram_per_gpu_bytes"] = vram_per_gpu_bytes
    bounds = resolve_bounds(gpu_count, defaults)
    limits = resolve_limits(defaults)
    shared = _build_inference(policy, request, manifest, benchmark_spec)
    from .processes import InferencePool

    pool = InferencePool(
        adapter=shared,
        gpu_devices=gpu_devices,
        base_port=int(manifest.defaults.get("base_port", 16694)),
        output_dir=output_dir / "autotune" / "policy-pool",
        ready_timeout_s=float(manifest.defaults.get("ready_timeout_s", 900)),
        cuda_mps=cuda_mps,
    )
    try:
        with autotune_phase(phase_timing_callback):
            selected = run_autotune(
                measure=lambda k, g, probe_limits, probe_progress: _measure_shared(
                    factory,
                    request,
                    manifest,
                    pool,
                    gpu_devices,
                    k,
                    g,
                    probe_limits,
                    probe_progress,
                    output_dir,
                ),
                output_dir=output_dir / "autotune",
                bounds=bounds,
                limits=limits,
            )
        final = _run_with_pool(
            factory=factory,
            request=request,
            manifest=manifest,
            pool=pool,
            gpu_devices=gpu_devices,
            concurrency=int(selected["k"]),
            replicas=int(selected["g"]),
            output_dir=output_dir,
            probe=False,
            phase_timing_callback=phase_timing_callback,
            progress_callback=progress_callback,
            shard_kwargs=shard_kwargs,
        )
        return final, selected
    finally:
        pool.stop()


def _build_inference(policy, request, manifest, benchmark_spec: BenchmarkSpec) -> PolicyProcessAdapter:
    return PolicyProcessAdapter(
        command=policy.command,
        checkpoint_path=request.checkpoint_path,
        benchmark=request.benchmark,
        suite=request.suite,
        benchmark_spec=benchmark_spec,
        cwd=policy.cwd,
        load_timeout_s=float(manifest.defaults.get("ready_timeout_s", 900)),
    )


def _measure_shared(
    factory: BenchmarkFactory,
    request: BenchmarkRunRequest,
    manifest,
    pool,
    gpu_devices: tuple[str, ...],
    k: int,
    g: int,
    probe_limits: ProbeLimits,
    probe_progress: ProbeProgressFn,
    probe_root: Path,
):
    output = _run_with_pool(
        factory=factory,
        request=request,
        manifest=manifest,
        pool=pool,
        gpu_devices=gpu_devices,
        concurrency=k,
        replicas=g,
        output_dir=probe_root / "autotune" / f"candidate-k{k}-g{g}",
        probe=True,
        probe_limits=probe_limits,
        progress_callback=lambda progress, _results: probe_progress(progress),
    )
    return output, pool.adapter.vram_bytes_per_replica()


def _shard_kwargs(args: argparse.Namespace, manifest: BenchmarkManifest) -> dict[str, Any]:
    if args.shard_index is None:
        return {}
    return {
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "shard_episodes": args.shard_episodes,
        # The merge refuses shards that ran a different kit version or manifest.
        "shard_build": shard_build(manifest),
    }


def _run_with_pool(
    *,
    factory: BenchmarkFactory,
    request: BenchmarkRunRequest,
    manifest,
    pool,
    gpu_devices: tuple[str, ...],
    concurrency: int,
    replicas: int,
    output_dir: Path,
    probe: bool,
    probe_limits: ProbeLimits | None = None,
    phase_timing_callback: PhaseTimingCallback | None = None,
    progress_callback=None,
    shard_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    adapter = (
        factory.create_probe(request, manifest, concurrency) if probe else factory.create(request, manifest)
    )
    runtime = ParallelBenchmarkRuntime(
        config=RuntimeConfig(
            output_dir=output_dir,
            gpu_devices=gpu_devices,
            replicas=replicas,
            concurrency=concurrency,
            base_port=int(manifest.defaults.get("base_port", 16694)),
            ready_timeout_s=float(manifest.defaults.get("ready_timeout_s", 900)),
            probe_limits=probe_limits,
            **(shard_kwargs or {}),
        ),
        inference=pool.adapter,
        benchmark=adapter,
        server_pool=pool,
    )
    return runtime.run_workers(
        progress_callback=progress_callback,
        phase_timing_callback=phase_timing_callback,
    )


def run_candidate(
    *,
    factory: BenchmarkFactory,
    request: BenchmarkRunRequest,
    manifest,
    policy,
    benchmark_spec: BenchmarkSpec,
    gpu_devices: tuple[str, ...],
    concurrency: int,
    replicas: int,
    output_dir: Path,
    probe: bool,
    phase_timing_callback: PhaseTimingCallback | None = None,
    progress_callback=None,
    shard_kwargs: dict[str, Any] | None = None,
    cuda_mps: CudaMpsSession | None = None,
) -> dict[str, Any]:
    adapter = (
        factory.create_probe(request, manifest, concurrency) if probe else factory.create(request, manifest)
    )
    inference = _build_inference(policy, request, manifest, benchmark_spec)
    return ParallelBenchmarkRuntime(
        config=RuntimeConfig(
            output_dir=output_dir,
            gpu_devices=gpu_devices,
            replicas=replicas,
            concurrency=concurrency,
            base_port=int(manifest.defaults.get("base_port", 16694)),
            ready_timeout_s=float(manifest.defaults.get("ready_timeout_s", 900)),
            **(shard_kwargs or {}),
        ),
        inference=inference,
        benchmark=adapter,
        cuda_mps=cuda_mps,
    ).run(
        progress_callback=progress_callback,
        phase_timing_callback=phase_timing_callback,
    )


def detect_gpu_devices() -> tuple[str, ...]:
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        visible = os.environ["CUDA_VISIBLE_DEVICES"].strip()
        if not visible or visible == "-1":
            return ()
        if visible.lower() != "all":
            devices = tuple(item.strip() for item in visible.split(",") if item.strip())
            if len(devices) != len(set(devices)):
                raise ValueError("CUDA_VISIBLE_DEVICES must contain unique device identifiers")
            return devices
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
        capture_output=True,
        text=True,
        check=True,
        timeout=10,
    )
    return tuple(line.strip() for line in result.stdout.splitlines() if line.strip())


def detect_gpu_memory_bytes(gpu_devices: tuple[str, ...]) -> int | None:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,uuid,memory.total",
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
    memory_by_device: dict[str, int] = {}
    for line in result.stdout.splitlines():
        index, uuid, memory_mib = [item.strip() for item in line.split(",")]
        memory_bytes = int(float(memory_mib) * 1024**2)
        memory_by_device[index] = memory_bytes
        memory_by_device[uuid] = memory_bytes
    selected = [memory_by_device.get(device) for device in gpu_devices]
    if not selected or any(memory is None for memory in selected):
        return None
    return min(int(memory) for memory in selected if memory is not None)


__all__ = ["detect_gpu_devices", "detect_gpu_memory_bytes", "main", "parse_args"]

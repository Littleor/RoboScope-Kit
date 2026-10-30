# Copyright 2026 RoboScope Team
# SPDX-License-Identifier: Apache-2.0

"""Tune official LIBERO concurrency against a StarVLA policy-server pool."""

# ruff: noqa: E402, I001

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPOSITORY_ROOT))

from roboscope_kit.runtime.autotune import (
    ClientMetrics,
    GTrial,
    KTrial,
    ProbeMetrics,
    ServerMetrics,
    build_adaptive_k_candidates,
    calculate_g_limit,
    calculate_k_limits,
    select_g,
    select_k,
)
from roboscope_kit.runtime.benchmark.resources import (
    GpuSampler,
    cpu_quota,
    cpu_window,
    gpu_window,
    read_cpu_stat,
)


@dataclass
class ServerPoolProcess:
    process: subprocess.Popen
    log_file: Any
    output_dir: Path
    active_replicas: int


PROFILE_STAGE_FIELDS = (
    "observation_build_s",
    "client_step_s",
    "action_decode_s",
    "env_step_s",
    "reset_s",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--starvla-dir", type=Path, default=Path("/root/starVLA"))
    parser.add_argument("--libero-home", type=Path, default=Path("/root/LIBERO"))
    parser.add_argument(
        "--starvla-python",
        type=Path,
        default=Path("/root/miniconda3/envs/starVLA/bin/python"),
    )
    parser.add_argument(
        "--libero-python",
        type=Path,
        default=Path("/root/miniconda3/envs/libero/bin/python"),
    )
    parser.add_argument("--suite", default="libero_10")
    parser.add_argument("--gpu-count", type=int, default=4)
    parser.add_argument("--replicas", type=int, default=4)
    parser.add_argument("--base-port", type=int, default=16694)
    parser.add_argument("--server-cpu-threads", type=int, default=2)
    parser.add_argument("--client-cpu-threads", type=int, default=1)
    parser.add_argument("--replica-scale-settle-s", type=float, default=2.0)
    parser.add_argument("--probe-duration", type=float, default=10)
    parser.add_argument("--duration", type=float, default=60)
    parser.add_argument("--warmup-action-steps", type=int, default=8)
    parser.add_argument(
        "--k-candidates",
        default="",
        help="Optional comma-separated override; defaults to a ramp derived from the measured K limits",
    )
    parser.add_argument(
        "--g-candidates",
        default="",
        help="Optional comma-separated override; defaults to a geometric ramp from 1 to --replicas",
    )
    parser.add_argument("--chunk-count", type=int, default=500)
    parser.add_argument("--reserved-vram-per-gpu-gib", type=float, default=2.0)
    parser.add_argument(
        "--reserved-cpu-cores",
        type=float,
        default=1.0,
        help="CPU cores reserved for the OS and unrelated processes",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ready-timeout", type=float, default=900)
    return parser.parse_args()


def aggregate_client_profile(
    results: list[dict[str, Any]],
    *,
    elapsed_s: float,
    cgroup_cpu_cores: float,
) -> dict[str, Any]:
    """Aggregate worker CPU and phase timings without mixing in server CPU."""

    total_client_wall_s = sum(float(result["elapsed_s"]) for result in results)
    process_user_cpu_s = sum(float(result.get("process_user_cpu_s", 0.0)) for result in results)
    process_system_cpu_s = sum(float(result.get("process_system_cpu_s", 0.0)) for result in results)
    process_cpu_s = process_user_cpu_s + process_system_cpu_s
    stage_time_s = {
        field: sum(float(result.get(field, 0.0)) for result in results) for field in PROFILE_STAGE_FIELDS
    }
    accounted_stage_s = sum(stage_time_s.values())
    client_process_cpu_cores = process_cpu_s / max(elapsed_s, 1e-9)
    return {
        "worker_count": len(results),
        "client_wall_time_s": total_client_wall_s,
        "cgroup_cpu_cores": cgroup_cpu_cores,
        "process_user_cpu_s": process_user_cpu_s,
        "process_system_cpu_s": process_system_cpu_s,
        "process_cpu_s": process_cpu_s,
        "process_cpu_cores": client_process_cpu_cores,
        "process_cpu_cores_per_worker": client_process_cpu_cores / max(len(results), 1),
        "non_client_cgroup_cpu_cores": max(cgroup_cpu_cores - client_process_cpu_cores, 0.0),
        "stage_time_s": stage_time_s,
        "stage_fraction_of_client_wall": {
            field: value / max(total_client_wall_s, 1e-9) for field, value in stage_time_s.items()
        },
        "unaccounted_client_wall_s": max(total_client_wall_s - accounted_stage_s, 0.0),
    }


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(round((len(ordered) - 1) * fraction), len(ordered) - 1)
    return ordered[index]


def build_local_k_candidates(center: int, hard_maximum: int, gpu_count: int) -> tuple[int, ...]:
    step = max(1, math.ceil(gpu_count / 2))
    start = max(1, center - 2 * step)
    candidates = list(range(start, hard_maximum + 1, step))
    if candidates[-1] != hard_maximum:
        candidates.append(hard_maximum)
    return tuple(candidates)


def gpu_memory_capacity_bytes() -> int:
    command = [
        "nvidia-smi",
        "--query-gpu=memory.total",
        "--format=csv,noheader,nounits",
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=True, timeout=10)
    capacities = [int(line.strip()) * 1024**2 for line in result.stdout.splitlines() if line.strip()]
    if not capacities:
        raise RuntimeError("nvidia-smi returned no GPU memory capacities")
    return min(capacities)


def sample_server_vram(replicas: int, duration_s: float = 1.0) -> int:
    sampler = GpuSampler(interval_s=0.2)
    sampler.start()
    time.sleep(duration_s)
    server_vram_bytes = int(gpu_window(sampler.stop(), replicas)["peak_memory_bytes"])
    if server_vram_bytes <= 0:
        raise RuntimeError("failed to measure policy-server VRAM")
    return server_vram_bytes


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def append_event(path: Path, payload: dict) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload) + "\n")
        stream.flush()


def wait_for_files(paths: list[Path], processes: list[subprocess.Popen], timeout_s: float) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if all(path.exists() for path in paths):
            return
        for process in processes:
            if process.poll() is not None:
                raise RuntimeError(f"client exited before ready with code {process.returncode}")
        time.sleep(0.2)
    missing = [str(path) for path in paths if not path.exists()]
    raise TimeoutError(f"clients did not become ready: {missing}")


def stop_process(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=10)


def client_command(
    args: argparse.Namespace,
    *,
    worker_id: int,
    port: int,
    barrier_dir: Path,
    result_path: Path,
) -> list[str]:
    client_script = Path(__file__).with_name("libero_client.py")
    command = [
        str(args.libero_python),
        str(client_script),
        "--worker-id",
        str(worker_id),
        "--port",
        str(port),
        "--suite",
        args.suite,
        "--task-id",
        str(worker_id),
        "--episode-index",
        str(worker_id),
        "--warmup-action-steps",
        str(args.warmup_action_steps),
        "--barrier-dir",
        str(barrier_dir),
        "--result",
        str(result_path),
    ]
    return command


def run_trial(
    args: argparse.Namespace,
    *,
    k: int,
    trial_dir: Path,
    duration_s: float | None = None,
    replicas: int | None = None,
) -> tuple[KTrial, dict[str, Any]]:
    measurement_duration_s = args.duration if duration_s is None else duration_s
    active_replicas = args.replicas if replicas is None else replicas
    if active_replicas < 1 or active_replicas > args.replicas:
        raise ValueError(f"replicas must be between 1 and {args.replicas}")
    trial_dir.mkdir(parents=True, exist_ok=True)
    barrier_dir = trial_dir / "barrier"
    barrier_dir.mkdir()
    processes: list[subprocess.Popen] = []
    log_files: list[object] = []
    result_paths: list[Path] = []
    sampler: GpuSampler | None = None
    base_env = os.environ.copy()
    base_env["PYTHONPATH"] = f"{args.libero_home}:{args.starvla_dir}:{base_env.get('PYTHONPATH', '')}"
    base_env["LIBERO_CONFIG_PATH"] = "/root/.libero"
    base_env["MUJOCO_GL"] = "egl"
    base_env["PYOPENGL_PLATFORM"] = "egl"
    client_thread_count = str(args.client_cpu_threads)
    base_env["OMP_NUM_THREADS"] = client_thread_count
    base_env["MKL_NUM_THREADS"] = client_thread_count
    base_env["OPENBLAS_NUM_THREADS"] = client_thread_count
    base_env["NUMEXPR_NUM_THREADS"] = client_thread_count
    try:
        for worker_id in range(k):
            env = base_env.copy()
            env["MUJOCO_EGL_DEVICE_ID"] = str(worker_id % args.gpu_count)
            result_path = trial_dir / f"client-{worker_id}.json"
            log_file = (trial_dir / f"client-{worker_id}.log").open("w", encoding="utf-8")
            command = client_command(
                args,
                worker_id=worker_id,
                port=args.base_port + worker_id % active_replicas,
                barrier_dir=barrier_dir,
                result_path=result_path,
            )
            process = subprocess.Popen(
                command,
                env=env,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                text=True,
            )
            processes.append(process)
            log_files.append(log_file)
            result_paths.append(result_path)

        ready_paths = [barrier_dir / f"ready-{worker_id}.json" for worker_id in range(k)]
        wait_for_files(ready_paths, processes, args.ready_timeout)
        before = read_cpu_stat()
        sampler = GpuSampler()
        sampler.start()
        atomic_json(
            barrier_dir / "start.json",
            {"start_at_unix_ns": time.time_ns() + 200_000_000, "duration_s": measurement_duration_s},
        )
        deadline = time.monotonic() + measurement_duration_s + 180
        for process in processes:
            timeout = max(deadline - time.monotonic(), 1)
            process.wait(timeout=timeout)
            if process.returncode != 0:
                raise RuntimeError(f"client exited with code {process.returncode}")
        samples = sampler.stop()
        sampler = None
        after = read_cpu_stat()

        results = [json.loads(path.read_text(encoding="utf-8")) for path in result_paths]
        started_at = min(result["started_at_unix_ns"] for result in results)
        ended_at = max(result["ended_at_unix_ns"] for result in results)
        elapsed_s = max((ended_at - started_at) / 1e9, 1e-9)
        total_steps = sum(result["steps"] for result in results)
        inference_requests = sum(result["inference_requests"] for result in results)
        total_client_time = sum(result["elapsed_s"] for result in results)
        inference_wait_s = sum(result["inference_wait_s"] for result in results)
        inference_latencies = [latency for result in results for latency in result["inference_latencies_ms"]]
        cpu = cpu_window(before, after, elapsed_s, quota_cores=cpu_quota())
        gpu = gpu_window(samples, args.gpu_count)
        profiling = aggregate_client_profile(
            results,
            elapsed_s=elapsed_s,
            cgroup_cpu_cores=cpu["cpu_cores"],
        )
        clients = ClientMetrics(
            active_clients=k,
            throughput=total_steps / elapsed_s,
            cpu_cores=cpu["cpu_cores"],
            throttle_time_ratio=cpu["throttle_time_ratio"],
            throttle_period_ratio=cpu["throttle_period_ratio"],
            inference_wait_ratio=inference_wait_s / max(total_client_time, 1e-9),
            error_count=0,
        )
        servers = ServerMetrics(
            replicas=active_replicas,
            throughput=inference_requests / elapsed_s,
            queue_wait_p95_ms=percentile(inference_latencies, 0.95),
            gpu_utilization=gpu["mean_utilization"],
            vram_bytes=int(gpu["peak_memory_bytes"]),
            error_count=0,
        )
        raw = {
            "elapsed_s": elapsed_s,
            "client_results": results,
            "inference_latency_p50_ms": percentile(inference_latencies, 0.50),
            "inference_latency_p95_ms": percentile(inference_latencies, 0.95),
            "gpu_samples": len(samples),
            "profiling": profiling,
        }
        return KTrial(k=k, clients=clients, servers=servers), raw
    finally:
        if sampler is not None:
            sampler.stop()
        for process in processes:
            stop_process(process)
        for log_file in log_files:
            log_file.close()


def start_servers(
    args: argparse.Namespace,
    *,
    initial_replicas: int | None = None,
) -> ServerPoolProcess:
    server_dir = args.output_dir / "servers"
    server_dir.mkdir(parents=True, exist_ok=True)
    server_log = (server_dir / "manager.log").open("w", encoding="utf-8")
    manifest = server_dir / "server-manifest.json"
    control = server_dir / "server-control.json"
    manifest.unlink(missing_ok=True)
    control.unlink(missing_ok=True)
    initial = args.replicas if initial_replicas is None else initial_replicas
    command = [
        sys.executable,
        str(Path(__file__).with_name("libero_server.py")),
        "--checkpoint",
        str(args.checkpoint),
        "--starvla-dir",
        str(args.starvla_dir),
        "--starvla-python",
        str(args.starvla_python),
        "--replicas",
        str(args.replicas),
        "--initial-replicas",
        str(initial),
        "--gpu-count",
        str(args.gpu_count),
        "--base-port",
        str(args.base_port),
        "--server-cpu-threads",
        str(args.server_cpu_threads),
        "--output-dir",
        str(server_dir),
        "--ready-timeout",
        str(args.ready_timeout),
    ]
    process = subprocess.Popen(command, stdout=server_log, stderr=subprocess.STDOUT, text=True)
    try:
        wait_for_files([manifest], [process], args.ready_timeout)
        wait_for_server_replicas(process, manifest, initial, args.ready_timeout)
        return ServerPoolProcess(
            process=process,
            log_file=server_log,
            output_dir=server_dir,
            active_replicas=initial,
        )
    except Exception:
        stop_process(process)
        server_log.close()
        raise


def wait_for_server_replicas(
    process: subprocess.Popen,
    manifest_path: Path,
    target: int,
    timeout_s: float,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"policy server manager exited with {process.returncode}")
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            time.sleep(0.2)
            continue
        if manifest.get("state") == "ready" and manifest.get("active_replicas") == target:
            return manifest
        time.sleep(0.2)
    raise TimeoutError(f"policy server pool did not reach {target} replicas")


def scale_servers(pool: ServerPoolProcess, args: argparse.Namespace, target: int) -> dict[str, Any]:
    if pool.active_replicas == target:
        return json.loads((pool.output_dir / "server-manifest.json").read_text(encoding="utf-8"))
    atomic_json(pool.output_dir / "server-control.json", {"target_replicas": target})
    manifest = wait_for_server_replicas(
        pool.process,
        pool.output_dir / "server-manifest.json",
        target,
        args.ready_timeout,
    )
    pool.active_replicas = target
    if args.replica_scale_settle_s:
        time.sleep(args.replica_scale_settle_s)
    return manifest


def main() -> None:
    args = parse_args()
    if args.chunk_count < 1:
        raise ValueError("chunk-count must be positive")
    if args.reserved_vram_per_gpu_gib < 0:
        raise ValueError("reserved-vram-per-gpu-gib cannot be negative")
    if args.reserved_cpu_cores < 0:
        raise ValueError("reserved-cpu-cores cannot be negative")
    if args.server_cpu_threads < 1:
        raise ValueError("server-cpu-threads must be positive")
    if args.client_cpu_threads < 1:
        raise ValueError("client-cpu-threads must be positive")
    if args.replica_scale_settle_s < 0:
        raise ValueError("replica-scale-settle-s cannot be negative")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    events_path = args.output_dir / "events.jsonl"
    events_path.write_text("", encoding="utf-8")
    append_event(events_path, {"type": "phase", "phase": "server-starting"})

    initial_replicas = 1
    server_pool = start_servers(args, initial_replicas=initial_replicas)
    try:
        server_vram_bytes = sample_server_vram(args.gpu_count)
        vram_per_gpu_bytes = gpu_memory_capacity_bytes()
        args.replicas = min(
            args.replicas,
            calculate_g_limit(
                gpu_count=args.gpu_count,
                vram_per_gpu_bytes=vram_per_gpu_bytes,
                reserved_vram_per_gpu_bytes=int(args.reserved_vram_per_gpu_gib * 1024**3),
                server_vram_bytes=server_vram_bytes,
            ),
            args.chunk_count,
        )
        append_event(
            events_path,
            {"type": "resource_limit", "axis": "g", "maximum": args.replicas},
        )
        append_event(events_path, {"type": "phase", "phase": "probe"})
        probe_trial, probe_raw = run_trial(
            args,
            k=1,
            replicas=initial_replicas,
            trial_dir=args.output_dir / "probe-k1",
            duration_s=args.probe_duration,
        )
        client_vram_bytes = probe_trial.servers.vram_bytes - server_vram_bytes
        if client_vram_bytes <= 0:
            raise RuntimeError("failed to measure positive incremental client VRAM")
        probe = ProbeMetrics(
            throughput=probe_trial.clients.throughput,
            cpu_cores=probe_trial.clients.cpu_cores,
            vram_bytes=client_vram_bytes,
            inference_wait_ratio=probe_trial.clients.inference_wait_ratio,
        )
        append_event(
            events_path,
            {"type": "probe", "metrics": asdict(probe), "profiling": probe_raw["profiling"]},
        )

        quota = cpu_quota()
        limits = calculate_k_limits(
            gpu_count=args.gpu_count,
            vram_per_gpu_bytes=vram_per_gpu_bytes,
            reserved_vram_per_gpu_bytes=int(args.reserved_vram_per_gpu_gib * 1024**3),
            server_replicas=initial_replicas,
            server_vram_bytes=math.ceil(server_vram_bytes / initial_replicas),
            cpu_quota=quota,
            reserved_cpu_cores=args.reserved_cpu_cores,
            probe=probe,
            chunk_count=args.chunk_count,
        )
        fixed_k_candidates = bool(args.k_candidates.strip())
        candidates = [int(value) for value in args.k_candidates.split(",") if value.strip()]
        hard_k_limit = min(limits.vram, limits.queue)
        if not candidates:
            candidates = list(
                build_adaptive_k_candidates(
                    args.gpu_count,
                    hard_k_limit,
                )
            )
        if any(candidate < 1 or candidate > hard_k_limit for candidate in candidates):
            raise ValueError(f"K candidates must be between 1 and the hard limit {hard_k_limit}")
        if candidates != sorted(set(candidates)):
            raise ValueError("K candidates must be unique and increasing")
        k_trials = [probe_trial]
        raw_trials: dict[str, Any] = {"probe": probe_raw}
        append_event(events_path, {"type": "phase", "phase": "k-ramp"})
        for k in candidates:
            if k == 1:
                continue
            try:
                trial, raw = run_trial(
                    args,
                    k=k,
                    replicas=initial_replicas,
                    trial_dir=args.output_dir / f"trial-k{k}",
                )
            except (RuntimeError, TimeoutError, subprocess.TimeoutExpired) as exc:
                if fixed_k_candidates:
                    raise RuntimeError(f"fixed K candidate {k} was rejected: {exc}") from exc
                append_event(
                    events_path,
                    {"type": "candidate_rejected", "axis": "k", "candidate": k, "error": str(exc)},
                )
                break
            k_trials.append(trial)
            raw_trials[f"k{k}"] = raw
            append_event(
                events_path,
                {
                    "type": "metric",
                    "axis": "k",
                    "candidate": k,
                    "clients": asdict(trial.clients),
                    "servers": asdict(trial.servers),
                    "profiling": raw["profiling"],
                },
            )
            if (
                not fixed_k_candidates
                and len(k_trials) > 1
                and select_k(
                    k_trials,
                    cpu_quota=quota,
                    reserved_cpu_cores=args.reserved_cpu_cores,
                ).k
                != k
            ):
                break

        if fixed_k_candidates:
            selected_k_trial = k_trials[-1]
        else:
            selected_k_trial = select_k(
                k_trials,
                cpu_quota=quota,
                reserved_cpu_cores=args.reserved_cpu_cores,
            )
        append_event(events_path, {"type": "selection", "axis": "k", "value": selected_k_trial.k})

        fixed_g_candidates = bool(args.g_candidates.strip())
        g_candidates = [int(value) for value in args.g_candidates.split(",") if value.strip()]
        if not g_candidates:
            first = min(args.gpu_count * 2, args.replicas)
            g_candidates = [1, *range(first, args.replicas + 1, args.gpu_count)]
            if g_candidates[-1] != args.replicas:
                g_candidates.append(args.replicas)
            g_candidates = sorted(set(g_candidates))
        if any(candidate < 1 or candidate > args.replicas for candidate in g_candidates):
            raise ValueError(f"G candidates must be between 1 and {args.replicas}")
        if g_candidates != sorted(set(g_candidates)):
            raise ValueError("G candidates must be unique and increasing")

        g_trials = []
        append_event(events_path, {"type": "phase", "phase": "g-ramp"})
        for g in g_candidates:
            try:
                manifest = scale_servers(server_pool, args, g)
                trial, raw = run_trial(
                    args,
                    k=selected_k_trial.k,
                    replicas=g,
                    trial_dir=args.output_dir / f"trial-g{g}",
                )
            except (RuntimeError, TimeoutError, subprocess.TimeoutExpired) as exc:
                append_event(
                    events_path,
                    {"type": "candidate_rejected", "axis": "g", "candidate": g, "error": str(exc)},
                )
                break
            append_event(
                events_path,
                {
                    "type": "replica_scaled",
                    "target": g,
                    "replicas": manifest["replicas"],
                },
            )
            g_trial = GTrial(g=g, clients=trial.clients, servers=trial.servers)
            g_trials.append(g_trial)
            raw_trials[f"g{g}"] = raw
            append_event(
                events_path,
                {
                    "type": "metric",
                    "axis": "g",
                    "candidate": g,
                    "clients": asdict(g_trial.clients),
                    "servers": asdict(g_trial.servers),
                    "profiling": raw["profiling"],
                },
            )
            if not fixed_g_candidates and len(g_trials) > 1 and select_g(g_trials).g != g:
                break

        if not g_trials:
            raise RuntimeError("all policy-replica candidates were rejected")
        selected_g_trial = select_g(g_trials)
        append_event(
            events_path,
            {
                "type": "selection",
                "axis": "g",
                "value": selected_g_trial.g,
                "reason": "last replica count before throughput gain falls below 1%",
            },
        )

        retuned_k_trials = []
        final_k_trial = selected_k_trial
        if not fixed_k_candidates:
            scale_servers(server_pool, args, selected_g_trial.g)
            append_event(events_path, {"type": "phase", "phase": "k-retune"})
            for k in build_local_k_candidates(selected_k_trial.k, hard_k_limit, args.gpu_count):
                try:
                    trial, raw = run_trial(
                        args,
                        k=k,
                        replicas=selected_g_trial.g,
                        trial_dir=args.output_dir / f"retune-k{k}-g{selected_g_trial.g}",
                    )
                except (RuntimeError, TimeoutError, subprocess.TimeoutExpired) as exc:
                    append_event(
                        events_path,
                        {
                            "type": "candidate_rejected",
                            "axis": "k-retune",
                            "candidate": k,
                            "g": selected_g_trial.g,
                            "error": str(exc),
                        },
                    )
                    break
                retuned_k_trials.append(trial)
                raw_trials[f"retune-k{k}-g{selected_g_trial.g}"] = raw
                append_event(
                    events_path,
                    {
                        "type": "metric",
                        "axis": "k-retune",
                        "candidate": k,
                        "g": selected_g_trial.g,
                        "clients": asdict(trial.clients),
                        "servers": asdict(trial.servers),
                        "profiling": raw["profiling"],
                    },
                )
                if (
                    len(retuned_k_trials) > 1
                    and select_k(
                        retuned_k_trials,
                        cpu_quota=quota,
                        reserved_cpu_cores=args.reserved_cpu_cores,
                    ).k
                    != k
                ):
                    break
            if retuned_k_trials:
                final_k_trial = select_k(
                    retuned_k_trials,
                    cpu_quota=quota,
                    reserved_cpu_cores=args.reserved_cpu_cores,
                )
            append_event(
                events_path,
                {"type": "selection", "axis": "k-retune", "value": final_k_trial.k},
            )

        output = {
            "workload": {
                "benchmark": "LIBERO",
                "suite": args.suite,
                "checkpoint": str(args.checkpoint),
                "policy": "StarVLA Qwen2.5-VL-3B OFT LIBERO 4-in-1",
                "gpu_count": args.gpu_count,
                "cpu_quota": quota,
                "reserved_cpu_cores": args.reserved_cpu_cores,
                "server_cpu_threads": args.server_cpu_threads,
                "client_cpu_threads": args.client_cpu_threads,
                "probe_duration_s": args.probe_duration,
                "measurement_duration_s": args.duration,
            },
            "probe": asdict(probe),
            "k_limits": {**asdict(limits), "maximum": limits.maximum},
            "resource_probe": {
                "server_vram_bytes": server_vram_bytes,
                "client_vram_bytes": client_vram_bytes,
            },
            "selected": {"k": final_k_trial.k, "g": selected_g_trial.g},
            "k_trials": [asdict(trial) for trial in k_trials],
            "g_trials": [asdict(trial) for trial in g_trials],
            "retuned_k_trials": [asdict(trial) for trial in retuned_k_trials],
            "raw": raw_trials,
        }
        atomic_json(args.output_dir / "metrics.json", output)
        append_event(
            events_path,
            {
                "type": "complete",
                "selected_k": final_k_trial.k,
                "selected_g": selected_g_trial.g,
                "throughput": final_k_trial.clients.throughput,
            },
        )
        print(json.dumps(output, indent=2))
    finally:
        stop_process(server_pool.process)
        server_pool.log_file.close()


if __name__ == "__main__":
    main()

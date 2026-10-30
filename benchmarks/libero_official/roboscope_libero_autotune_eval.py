"""Run the official LIBERO runtime while publishing the product status contract."""

# ruff: noqa: E402, I001

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Callable, Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

KIT_ROOT = Path(os.environ.get("ROBOSCOPE_KIT_ROOT", "/root/roboscope-kit"))
SDK_SOURCE_ROOT = KIT_ROOT / "packages" / "roboscope-kit" / "src"
sys.path.insert(0, str(KIT_ROOT))
sys.path.insert(0, str(SDK_SOURCE_ROOT))

from roboscope_kit import (
    ActionChunk,
    BenchmarkName,
    EpisodeContext,
    Observation,
    Policy,
    PolicyCapabilities,
    PolicyClient,
    PolicyRuntimeConfig,
    SessionContext,
    get_benchmark_spec,
    run,
)
from roboscope_kit.runtime.benchmark import (
    EvaluationRecord,
    LegacyProgressAdapter,
    ParallelBenchmarkRuntime,
    PhaseTimingCallback,
    ProcessSpec,
    RuntimeConfig,
    WorkerContext,
    WorkerLoop,
    WorkItem,
    WorkResult,
    atomic_json,
    autotune_phase,
)
from roboscope_kit.runtime.benchmark.policy_process import (
    PolicyProcessAdapter,
)
from roboscope_kit.runtime.benchmark.resources import (
    cpu_quota,
)
from roboscope_kit.runtime.benchmarks.libero import (
    LiberoBenchmarkAdapter,
    LiberoRuntimeConfig,
)

from benchmarks.libero_official.starvla_runtime.libero_sdk import (
    LiberoBenchmarkAdapter as StarVLALiberoBenchmarkAdapter,
)
from benchmarks.libero_official.starvla_runtime.libero_sdk import (
    StarVLAInferenceAdapter,
)


class SelfTestPolicy(Policy):
    def load(self, _context: SessionContext) -> PolicyCapabilities:
        return PolicyCapabilities(
            supported_benchmarks=(BenchmarkName.LIBERO,),
            required_features=(
                "images.primary",
                "images.wrist",
                "state.proprioception",
                "language.instruction",
            ),
            supported_suites={BenchmarkName.LIBERO: ("libero_10",)},
            action_representations={BenchmarkName.LIBERO: "libero.delta-eef-v1"},
        )

    def act(self, _observation: Observation) -> ActionChunk:
        return ActionChunk(np.zeros(7, dtype=np.float32))


class SelfTestBenchmarkAdapter:
    def work_items(self) -> Sequence[WorkItem]:
        return tuple(
            WorkItem(
                ordinal=index,
                item_id=f"self-test-{index}",
                group_id=f"group-{index % 2}",
                payload={"success": index % 2 == 0, "steps": index + 1},
            )
            for index in range(4)
        )

    def worker_process(self, context: WorkerContext) -> ProcessSpec:
        return ProcessSpec(
            command=(
                sys.executable,
                str(Path(__file__).resolve()),
                "self-test-worker",
                "--worker-id",
                str(context.worker_id),
                "--queue-dir",
                str(context.queue_dir),
                "--state-dir",
                str(context.state_dir),
                "--policy-socket",
                str(context.replica_socket),
            ),
            env={"PYTHONPATH": (f"{SDK_SOURCE_ROOT}:{KIT_ROOT}:{os.environ.get('PYTHONPATH', '')}")},
        )

    def aggregate(self, results: Sequence[WorkResult]) -> dict[str, Any]:
        records = [record for result in results for record in result.records]
        return {"episodes": len(records), "successes": sum(record.success for record in records)}

    def metadata(self) -> dict[str, Any]:
        return {"adapter": type(self).__name__, "benchmark": "sdk-self-test"}


def run_self_test_worker(
    worker_id: int,
    queue_dir: Path,
    state_dir: Path,
    policy_socket: Path,
) -> None:
    def evaluate(item: WorkItem) -> WorkResult:
        episode_id = item.item_id
        with PolicyClient(policy_socket) as client:
            client.reset(EpisodeContext(episode_id=episode_id, task_id=item.group_id))
        with PolicyClient(policy_socket) as client:
            action = client.act(
                Observation(
                    episode_id=episode_id,
                    step=0,
                    features={
                        "images.primary": np.zeros((256, 256, 3), dtype=np.uint8),
                        "images.wrist": np.zeros((256, 256, 3), dtype=np.uint8),
                        "state.proprioception": np.zeros(16, dtype=np.float32),
                        "language.instruction": "self test",
                    },
                )
            )
            client.end_episode(episode_id)
        if action.actions.shape != (7,):
            raise RuntimeError(f"unexpected self-test action shape: {action.actions.shape}")
        return WorkResult(
            item_id=item.item_id,
            group_id=item.group_id,
            worker_id=worker_id,
            records=(
                EvaluationRecord(
                    record_id=item.item_id,
                    success=bool(item.payload["success"]),
                    steps=int(item.payload["steps"]),
                    elapsed_s=0.0,
                ),
            ),
        )

    WorkerLoop(worker_id=worker_id, queue_dir=queue_dir, state_dir=state_dir).run(evaluate)


def run_sdk_self_test() -> Path:
    output_dir = Path(tempfile.mkdtemp(prefix="roboscope-benchmark-self-test-"))
    inference = PolicyProcessAdapter(
        command=(sys.executable, str(Path(__file__).resolve()), "self-test-policy"),
        checkpoint_path=output_dir / "checkpoint",
        benchmark=BenchmarkName.LIBERO,
        suite="libero_10",
        benchmark_spec=get_benchmark_spec(BenchmarkName.LIBERO, "libero_10"),
        env={"PYTHONPATH": f"{SDK_SOURCE_ROOT}:{KIT_ROOT}:{os.environ.get('PYTHONPATH', '')}"},
    )
    runtime = ParallelBenchmarkRuntime(
        config=RuntimeConfig(
            output_dir=output_dir,
            gpu_devices=("0",),
            replicas=2,
            concurrency=2,
            ready_timeout_s=10,
            poll_interval_s=0.01,
        ),
        inference=inference,
        benchmark=SelfTestBenchmarkAdapter(),
    )
    output = runtime.run()
    result = output["result"]
    if result["episodes"] != 4 or result["successes"] != 2:
        raise RuntimeError(f"unexpected SDK self-test result: {result}")
    for name in ("manifest.json", "progress.json", "result.json", "events.jsonl"):
        if not (output_dir / name).is_file():
            raise RuntimeError(f"SDK self-test did not create {name}")
    persisted = json.loads((output_dir / "result.json").read_text(encoding="utf-8"))
    if persisted["result"]["state"] != "complete":
        raise RuntimeError("SDK self-test did not persist a complete result")
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    expected_policy_io = {
        "benchmark": "libero",
        "action_representation": "libero.delta-eef-v1",
        "action_representation_source": "policy",
        "observation_stride": 1,
        "observation_stride_source": "safe_default",
    }
    if manifest["policy_io"] != expected_policy_io:
        raise RuntimeError(f"SDK self-test manifest has unexpected policy I/O: {manifest}")
    if persisted["policy_io"] != expected_policy_io:
        raise RuntimeError(f"SDK self-test result has unexpected policy I/O: {persisted}")
    return output_dir


def detect_gpu_count() -> int:
    explicit = os.environ.get("BENCHMARK_GPU_COUNT")
    if explicit:
        return max(1, int(explicit))
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if visible and visible != "all":
        devices = [part for part in visible.split(",") if part.strip()]
        if devices:
            return len(devices)
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
    except (FileNotFoundError, PermissionError, OSError, subprocess.TimeoutExpired):
        return 0
    if result.returncode != 0:
        return 0
    return len([line for line in result.stdout.splitlines() if line.strip()])


def env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    return default if raw in (None, "") else int(raw)


def env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw in (None, ""):
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def env_task_ids(name: str) -> tuple[int, ...] | None:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return None
    return tuple(int(value.strip()) for value in raw.split(","))


def build_autotune_args(
    *,
    checkpoint: Path,
    suite: str,
    num_tasks: int,
    num_trials: int,
    output_dir: Path,
) -> argparse.Namespace:
    gpu_count = detect_gpu_count()
    if gpu_count < 1:
        raise RuntimeError("no GPU detected; set BENCHMARK_GPU_COUNT or attach GPUs")
    total_episodes = num_tasks * num_trials
    replicas = env_int("BENCHMARK_REPLICAS", total_episodes)
    concurrency = env_int("BENCHMARK_CONCURRENCY", total_episodes)
    allow_sharding = env_bool("BENCHMARK_ALLOW_EPISODE_SHARDING", True)
    concurrency = max(1, min(concurrency, total_episodes))
    replicas = max(1, min(replicas, concurrency))
    if concurrency > num_tasks and not allow_sharding:
        concurrency = num_tasks
    render_backend = os.environ.get("BENCHMARK_RENDER_BACKEND", "egl")
    if render_backend not in {"egl", "osmesa"}:
        raise ValueError(f"unsupported BENCHMARK_RENDER_BACKEND: {render_backend}")
    return SimpleNamespace(
        checkpoint=checkpoint,
        starvla_dir=Path(os.environ.get("STARVLA_DIR", "/root/starVLA")),
        libero_home=Path(os.environ.get("LIBERO_HOME", "/root/LIBERO")),
        libero_config_path=Path(os.environ.get("LIBERO_CONFIG_PATH", "/root/.libero/config.yaml")),
        starvla_python=Path(os.environ.get("STARVLA_PYTHON", "/root/miniconda3/envs/starVLA/bin/python")),
        libero_python=Path(os.environ.get("LIBERO_PYTHON", "/root/miniconda3/envs/libero/bin/python")),
        suite=suite,
        benchmark_name="LIBERO",
        num_tasks=num_tasks,
        task_ids=env_task_ids("BENCHMARK_TASK_IDS"),
        sample_size=None,
        sample_seed=env_int("BENCHMARK_SEED", 7),
        gpu_count=gpu_count,
        replicas=replicas,
        concurrency=concurrency,
        num_trials=num_trials,
        chunk_size=env_int("BENCHMARK_CHUNK_SIZE", 1),
        base_port=env_int("BENCHMARK_POLICY_PORT", 16694),
        server_cpu_threads=env_int("BENCHMARK_SERVER_CPU_THREADS", 2),
        client_cpu_threads=env_int("BENCHMARK_CLIENT_CPU_THREADS", 1),
        seed=env_int("BENCHMARK_SEED", 7),
        render_backend=render_backend,
        save_videos=env_bool("BENCHMARK_SAVE_VIDEOS", True),
        allow_episode_sharding=allow_sharding,
        output_dir=output_dir,
        ready_timeout=float(os.environ.get("BENCHMARK_READY_TIMEOUT", "900")),
    )


def probe_runtime_parameters(
    args: argparse.Namespace,
    total_episodes: int,
    phase_timing_callback: PhaseTimingCallback | None = None,
) -> dict[str, Any]:
    with autotune_phase(phase_timing_callback):
        return _probe_runtime_parameters(args, total_episodes)


def _probe_runtime_parameters(args: argparse.Namespace, total_episodes: int) -> dict[str, Any]:
    if env_bool("BENCHMARK_QUICK_TEST", False):
        return {"selected": {"k": args.concurrency, "g": args.replicas}, "mode": "quick-test"}
    probe_dir = args.output_dir / "probe"
    command = [
        str(args.libero_python),
        str(KIT_ROOT / "benchmarks/libero_official/starvla_runtime/libero_controller.py"),
        "--checkpoint",
        str(args.checkpoint),
        "--starvla-dir",
        str(args.starvla_dir),
        "--libero-home",
        str(args.libero_home),
        "--starvla-python",
        str(args.starvla_python),
        "--libero-python",
        str(args.libero_python),
        "--suite",
        args.suite,
        "--gpu-count",
        str(args.gpu_count),
        "--replicas",
        str(args.replicas),
        "--base-port",
        str(args.base_port),
        "--server-cpu-threads",
        str(args.server_cpu_threads),
        "--client-cpu-threads",
        str(args.client_cpu_threads),
        "--probe-duration",
        os.environ.get("BENCHMARK_AUTOTUNE_PROBE_DURATION", "10"),
        "--duration",
        os.environ.get("BENCHMARK_AUTOTUNE_TRIAL_DURATION", "20"),
        "--chunk-count",
        str(total_episodes),
        "--output-dir",
        str(probe_dir),
        "--ready-timeout",
        str(args.ready_timeout),
    ]
    explicit_k = os.environ.get("BENCHMARK_CONCURRENCY", "").strip()
    explicit_g = os.environ.get("BENCHMARK_REPLICAS", "").strip()
    if explicit_k:
        command.extend(("--k-candidates", explicit_k))
    if explicit_g:
        command.extend(("--g-candidates", explicit_g))
    subprocess.run(command, check=True)
    metrics = json.loads((probe_dir / "metrics.json").read_text(encoding="utf-8"))
    selected = metrics["selected"]
    args.concurrency = min(int(selected["k"]), total_episodes)
    args.replicas = min(int(selected["g"]), args.concurrency)
    atomic_json(
        args.output_dir / "autotune-selection.json",
        {
            "selected": {"k": args.concurrency, "g": args.replicas},
            "metrics": str(probe_dir / "metrics.json"),
        },
    )
    return metrics


def prepare_video_link(output_dir: Path, video_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    video_dir.mkdir(parents=True, exist_ok=True)
    video_link = output_dir / "video"
    if video_link.is_symlink() or video_link.exists():
        if video_link.is_dir() and not video_link.is_symlink():
            shutil.rmtree(video_link)
        else:
            video_link.unlink()
    os.symlink(video_dir, video_link)


def append_event(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, sort_keys=True) + "\n")


def build_sdk_policy_runtime(
    *,
    policy_config: PolicyRuntimeConfig,
    args: argparse.Namespace,
    output_dir: Path,
    replicas: int,
    concurrency: int,
    num_trials: int,
    task_ids: tuple[int, ...] | None,
    save_videos: bool,
) -> ParallelBenchmarkRuntime:
    benchmark = LiberoBenchmarkAdapter(
        LiberoRuntimeConfig(
            python=args.libero_python,
            libero_home=args.libero_home,
            libero_config_path=args.libero_config_path,
            suite=args.suite,
            num_tasks=args.num_tasks,
            num_trials=num_trials,
            task_ids=task_ids,
            sample_size=None,
            sample_seed=args.sample_seed,
            chunk_size=args.chunk_size,
            seed=args.seed,
            render_backend=args.render_backend,
            save_videos=save_videos,
            cpu_threads=args.client_cpu_threads,
        )
    )
    inference = PolicyProcessAdapter(
        command=policy_config.command,
        checkpoint_path=args.checkpoint,
        benchmark=BenchmarkName.LIBERO,
        suite=args.suite,
        benchmark_spec=get_benchmark_spec(BenchmarkName.LIBERO, args.suite),
        cwd=policy_config.cwd,
        load_timeout_s=args.ready_timeout,
    )
    return ParallelBenchmarkRuntime(
        config=RuntimeConfig(
            output_dir=output_dir,
            gpu_devices=tuple(str(index) for index in range(args.gpu_count)),
            replicas=replicas,
            concurrency=concurrency,
            base_port=args.base_port,
            ready_timeout_s=args.ready_timeout,
        ),
        inference=inference,
        benchmark=benchmark,
    )


def gpu_memory_capacities() -> dict[str, int]:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.total",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=True,
        timeout=10,
    )
    capacities = {}
    for line in result.stdout.splitlines():
        index, memory_mib = [item.strip() for item in line.split(",")]
        capacities[index] = int(memory_mib) * 1024**2
    if not capacities:
        raise RuntimeError("nvidia-smi returned no GPU memory capacities")
    return capacities


def policy_probe_shape(args: argparse.Namespace, concurrency: int) -> tuple[tuple[int, ...], int]:
    episodes_per_worker = env_int("BENCHMARK_AUTOTUNE_EPISODES_PER_WORKER", 2)
    if episodes_per_worker < 1:
        raise ValueError("BENCHMARK_AUTOTUNE_EPISODES_PER_WORKER must be positive")
    target_episodes = concurrency * episodes_per_worker
    if target_episodes > args.num_trials:
        raise ValueError(
            f"candidate K={concurrency} needs {target_episodes} comparable probe episodes, "
            f"but the evaluation provides {args.num_trials} initial states per task"
        )
    task_id = args.task_ids[0] if args.task_ids else 0
    return (task_id,), target_episodes


def run_sdk_policy_candidate(
    *,
    policy_config: PolicyRuntimeConfig,
    args: argparse.Namespace,
    probe_dir: Path,
    replicas: int,
    concurrency: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    task_ids, num_trials = policy_probe_shape(args, concurrency)
    candidate_dir = probe_dir / f"candidate-k{concurrency}-g{replicas}"
    output = build_sdk_policy_runtime(
        policy_config=policy_config,
        args=args,
        output_dir=candidate_dir,
        replicas=replicas,
        concurrency=concurrency,
        num_trials=num_trials,
        task_ids=task_ids,
        save_videos=False,
    ).run()
    resources = output["resources"]
    inference_metrics = output.get("inference_metrics", {})
    libero_runtime_metrics = output["official_result"]["runtime_metrics"]
    metrics = {
        "k": concurrency,
        "g": replicas,
        "episodes": output["result"]["episodes"],
        "elapsed_s": output["result"]["elapsed_s"],
        "episodes_per_s": output["result"]["episodes_per_s"],
        "steps_per_s": output["result"]["steps_per_s"],
        "cpu": resources["cpu"],
        "gpu": resources["gpu"],
        "inference": {
            "wait_ratio": libero_runtime_metrics["inference_wait_ratio"],
            "wait_s": libero_runtime_metrics["inference_wait_s"],
            "requests_per_s": inference_metrics.get("requests_per_s"),
            "queue_wait_p95_ms": (inference_metrics.get("queue_wait_ms") or {}).get("p95"),
            "request_total_p95_ms": (inference_metrics.get("request_total_ms") or {}).get("p95"),
        },
        "result": str(candidate_dir / "result.json"),
    }
    return output, metrics


def geometric_candidates(maximum: int) -> tuple[int, ...]:
    candidates = [1]
    while candidates[-1] < maximum:
        candidates.append(min(candidates[-1] * 2, maximum))
    return tuple(candidates)


def policy_candidate_pairs(
    *,
    total: int,
    maximum_replicas: int,
) -> tuple[tuple[int, int], ...]:
    replicas = geometric_candidates(maximum_replicas)
    pairs = [(1, 1)]
    pairs.extend((min(total, g), g) for g in replicas)
    return tuple(dict.fromkeys(pairs))


def policy_candidate_rejection(
    previous: dict[str, Any],
    current: dict[str, Any],
    *,
    quota: float,
    memory_capacities: dict[str, int],
) -> str | None:
    gain = current["episodes_per_s"] / max(previous["episodes_per_s"], 1e-9) - 1
    if gain < 0.01:
        return f"episode throughput gain {gain:.1%} is below 1%"
    if current["cpu"]["cpu_cores"] > max(quota - 1.0, 0.0):
        return "CPU usage reached the reserved-core limit"
    if current["cpu"]["throttle_time_ratio"] > 0.40:
        return "CPU throttle time exceeded 40%"
    if current["gpu"]["mean_utilization"] > 0.95:
        return "mean GPU utilization exceeded 95%"
    for gpu_id, memory_mib in current["gpu"]["gpu_mem_mib_max"].items():
        capacity = memory_capacities[gpu_id]
        if memory_mib * 1024**2 > capacity * 0.90:
            return f"GPU {gpu_id} memory usage exceeded 90%"
    return None


def probe_sdk_policy_runtime(
    *,
    policy_config: PolicyRuntimeConfig,
    args: argparse.Namespace,
    total: int,
    phase_timing_callback: PhaseTimingCallback | None = None,
) -> dict[str, Any]:
    with autotune_phase(phase_timing_callback):
        return _probe_sdk_policy_runtime(policy_config=policy_config, args=args, total=total)


def _probe_sdk_policy_runtime(
    *,
    policy_config: PolicyRuntimeConfig,
    args: argparse.Namespace,
    total: int,
) -> dict[str, Any]:
    if env_bool("BENCHMARK_QUICK_TEST", False):
        output = {
            "selected": {"k": 1, "g": 1},
            "mode": "quick-test-fixed",
            "selection_reason": "quick test always uses one Policy replica and one worker",
        }
        atomic_json(args.output_dir / "autotune-selection.json", output)
        return output
    probe_dir = args.output_dir / "probe"
    probe_dir.mkdir(parents=True, exist_ok=True)
    events_path = probe_dir / "events.jsonl"
    events_path.write_text("", encoding="utf-8")
    memory_capacities = gpu_memory_capacities()
    quota = cpu_quota()
    maximum_replicas = min(total, max(args.gpu_count, math.floor(quota)))

    trials: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    _first_output, first_metrics = run_sdk_policy_candidate(
        policy_config=policy_config,
        args=args,
        probe_dir=probe_dir,
        replicas=1,
        concurrency=1,
    )
    trials.append(first_metrics)
    append_event(events_path, {"type": "candidate", "status": "measured", **first_metrics})
    pairs = policy_candidate_pairs(
        total=total,
        maximum_replicas=maximum_replicas,
    )
    selected = first_metrics
    selection_reason = "only the K=1/g=1 candidate was available"
    for concurrency, replicas in pairs:
        if concurrency == 1 and replicas == 1:
            continue
        try:
            _output, metrics = run_sdk_policy_candidate(
                policy_config=policy_config,
                args=args,
                probe_dir=probe_dir,
                replicas=replicas,
                concurrency=concurrency,
            )
        except (OSError, RuntimeError, TimeoutError, ValueError, subprocess.SubprocessError) as exc:
            rejection = {"k": concurrency, "g": replicas, "reason": str(exc)}
            rejected.append(rejection)
            append_event(events_path, {"type": "candidate", "status": "rejected", **rejection})
            selection_reason = f"K={concurrency}/g={replicas} failed: {exc}"
            break
        trials.append(metrics)
        append_event(events_path, {"type": "candidate", "status": "measured", **metrics})
        reason = policy_candidate_rejection(
            selected,
            metrics,
            quota=quota,
            memory_capacities=memory_capacities,
        )
        if reason is not None:
            rejection = {"k": concurrency, "g": replicas, "reason": reason}
            rejected.append(rejection)
            append_event(events_path, {"type": "candidate", "status": "rejected", **rejection})
            selection_reason = reason
            break
        selected = metrics
        selection_reason = "highest measured safe candidate with at least 1% episode-throughput gain"

    output = {
        "selected": {"k": selected["k"], "g": selected["g"]},
        "mode": "dynamic-probe",
        "selection_reason": selection_reason,
        "cpu_quota": quota,
        "gpu_memory_capacity_bytes": memory_capacities,
        "trials": trials,
        "rejected": rejected,
    }
    append_event(
        events_path,
        {
            "type": "selection",
            "mode": output["mode"],
            "selected": output["selected"],
            "reason": selection_reason,
        },
    )
    atomic_json(probe_dir / "metrics.json", output)
    atomic_json(args.output_dir / "autotune-selection.json", output)
    return output


def run_sdk_policy_evaluation(
    *,
    policy_config: PolicyRuntimeConfig,
    args: argparse.Namespace,
    total: int,
    output_dir: Path,
    write_log: Callable[..., None],
    phase_timing_callback: PhaseTimingCallback | None = None,
) -> dict[str, Any]:
    autotune = probe_sdk_policy_runtime(
        policy_config=policy_config,
        args=args,
        total=total,
        phase_timing_callback=phase_timing_callback,
    )
    replicas = int(autotune["selected"]["g"])
    concurrency = int(autotune["selected"]["k"])
    runtime = build_sdk_policy_runtime(
        policy_config=policy_config,
        args=args,
        output_dir=output_dir,
        replicas=replicas,
        concurrency=concurrency,
        num_trials=args.num_trials,
        task_ids=args.task_ids,
        save_videos=args.save_videos,
    )
    benchmark = runtime.benchmark
    expected_total = sum(item.expected_results for item in benchmark.work_items())
    if expected_total != total:
        raise RuntimeError(f"episode count mismatch: SDK={expected_total} legacy={total}")
    legacy = LegacyProgressAdapter(
        suite=args.suite,
        total=total,
        write_log=write_log,
    )
    output = runtime.run(
        progress_callback=legacy.update,
        phase_timing_callback=phase_timing_callback,
    )
    output["autotune"] = {
        "selected": autotune["selected"],
        "mode": autotune["mode"],
        "selection_reason": autotune["selection_reason"],
    }
    if autotune["mode"] == "dynamic-probe":
        output["autotune"]["metrics"] = str(output_dir / "probe" / "metrics.json")
    atomic_json(output_dir / "result.json", output)
    legacy.finish()
    return output


def run_autotune_evaluation(
    *,
    suite: str,
    eval_num: int,
    ckpt_path: Path,
    total: int,
    num_tasks: int,
    output_dir: Path,
    video_dir: Path,
    write_log: Callable[..., None],
    policy_config: PolicyRuntimeConfig | None = None,
    phase_timing_callback: PhaseTimingCallback | None = None,
) -> dict[str, Any]:
    prepare_video_link(output_dir, video_dir)
    args = build_autotune_args(
        checkpoint=ckpt_path,
        suite=suite,
        num_tasks=num_tasks,
        num_trials=eval_num,
        output_dir=output_dir,
    )
    if policy_config is not None:
        return run_sdk_policy_evaluation(
            policy_config=policy_config,
            args=args,
            total=total,
            output_dir=output_dir,
            write_log=write_log,
            phase_timing_callback=phase_timing_callback,
        )
    benchmark = StarVLALiberoBenchmarkAdapter(args)
    expected_total = sum(item.expected_results for item in benchmark.work_items())
    if expected_total != total:
        raise RuntimeError(f"episode count mismatch: SDK={expected_total} legacy={total}")
    inference = StarVLAInferenceAdapter(
        checkpoint=ckpt_path,
        starvla_dir=args.starvla_dir,
        python=args.starvla_python,
        cpu_threads=args.server_cpu_threads,
    )
    concurrency_ceiling = total if args.allow_episode_sharding else num_tasks
    autotune = probe_runtime_parameters(
        args,
        concurrency_ceiling,
        phase_timing_callback=phase_timing_callback,
    )
    runtime = ParallelBenchmarkRuntime(
        config=RuntimeConfig(
            output_dir=output_dir,
            gpu_devices=tuple(str(index) for index in range(args.gpu_count)),
            replicas=args.replicas,
            concurrency=args.concurrency,
            base_port=args.base_port,
            ready_timeout_s=args.ready_timeout,
        ),
        inference=inference,
        benchmark=benchmark,
    )
    legacy = LegacyProgressAdapter(
        suite=suite,
        total=total,
        write_log=write_log,
    )
    output = runtime.run(
        progress_callback=legacy.update,
        phase_timing_callback=phase_timing_callback,
    )
    output["autotune"] = {
        "selected": autotune["selected"],
        "mode": autotune.get("mode", "dynamic-probe"),
    }
    if "mode" not in autotune:
        output["autotune"]["metrics"] = str(output_dir / "probe" / "metrics.json")
    atomic_json(output_dir / "result.json", output)
    legacy.finish()
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    worker = sub.add_parser("self-test-worker")
    worker.add_argument("--worker-id", type=int, required=True)
    worker.add_argument("--queue-dir", type=Path, required=True)
    worker.add_argument("--state-dir", type=Path, required=True)
    worker.add_argument("--policy-socket", type=Path, required=True)
    sub.add_parser("self-test-policy")
    args = parser.parse_args()
    if args.command == "self-test-worker":
        run_self_test_worker(args.worker_id, args.queue_dir, args.state_dir, args.policy_socket)
        return 0
    if args.command == "self-test-policy":
        run(SelfTestPolicy())
        return 0
    raise ValueError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())

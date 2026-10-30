#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None

if TYPE_CHECKING:
    from roboscope_kit.runtime.benchmark import BenchmarkTimingRecorder

ROOT = Path("/root")
STARVLA_DIR = ROOT / "starVLA"
LIBERO_HOME = ROOT / "LIBERO"
BENCHMARK_DIR = ROOT / "benchmark"
LOCAL_CHECKPOINT_DIR = BENCHMARK_DIR / "checkpoint"
POLICY_CHECKPOINT_SOURCE_FILE = LOCAL_CHECKPOINT_DIR / ".roboscope-source.json"
LOCAL_BASE_MODEL_DIR = BENCHMARK_DIR / "basemodel"
TESTVIDEO_DIR = ROOT / "testvideo"
TESTRESULT_DIR = ROOT / "testresult"
POLICY_CONFIG_RESULT = TESTRESULT_DIR / "policy-config.json"
TIMING_METRICS_FILE = TESTRESULT_DIR / "metrics.json"
TESTLOG_FILE = ROOT / "testlog.txt"
ASSET_MAP = Path(os.environ.get("ROBOSCOPE_BASE_ASSET_MAP", ROOT / "roboscope_assets" / "base_assets.json"))
LIBERO_PYTHON = ROOT / "miniconda3" / "envs" / "libero" / "bin" / "python"
VALID_SUITES = {"libero_spatial", "libero_object", "libero_goal", "libero_10", "libero_90"}
CKPT_ENV = [
    "BENCHMARK_CKPT_ENDPOINT",
    "BENCHMARK_CKPT_ACCESS_KEY_ID",
    "BENCHMARK_CKPT_ACCESS_KEY_SECRET",
]
BASE_ENV = [
    "BENCHMARK_BASE_ENDPOINT",
    "BENCHMARK_BASE_ACCESS_KEY_ID",
    "BENCHMARK_BASE_ACCESS_KEY_SECRET",
    "BENCHMARK_BASE_OSS_URI",
]
LOG_LOCK = threading.Lock()
LAST_LOG_ARGS = (0, 0, "none", 0, 0, 0, "idle")
PROGRESS_RE = re.compile(r"Progress:\s*([0-9.]+)%")
LOGGER = logging.getLogger(__name__)
MODEL_WEIGHT_PATTERNS = (
    "*.safetensors",
    "*.safetensors.index.json",
    "*.bin",
    "*.bin.index.json",
    "*.pt",
    "*.pth",
    "*.ckpt",
)


class BenchmarkError(Exception):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


def ensure_dirs() -> None:
    for path in [
        LOCAL_CHECKPOINT_DIR,
        LOCAL_BASE_MODEL_DIR,
        TESTVIDEO_DIR,
        TESTRESULT_DIR,
        STARVLA_DIR / "playground" / "Checkpoints",
        STARVLA_DIR / "playground" / "Pretrained_models",
    ]:
        path.mkdir(parents=True, exist_ok=True)


def write_log(
    user_pct: int,
    suite_pct: int,
    suite: str,
    total: int,
    success: int,
    failed: int,
    status: str,
) -> None:
    global LAST_LOG_ARGS
    with LOG_LOCK:
        LAST_LOG_ARGS = (user_pct, suite_pct, suite, total, success, failed, status)
        tmp = TESTLOG_FILE.with_name(f"{TESTLOG_FILE.name}.{os.getpid()}.{threading.get_ident()}.tmp")
        tmp.write_text(
            f"用户模型权重下载进度 {user_pct}%\n"
            f"测试套件下载进度 {suite_pct}%\n"
            f"LIBERO {suite} {total} {success} {failed} {status}\n",
            encoding="utf-8",
        )
        tmp.replace(TESTLOG_FILE)


def refresh_log() -> None:
    with LOG_LOCK:
        args = LAST_LOG_ARGS
    write_log(*args)


def update_download_progress(env_kind: str, pct: float) -> None:
    if env_kind != "ckpt":
        return
    pct_int = max(0, min(100, int(round(pct))))
    with LOG_LOCK:
        user_pct, suite_pct, suite, total, success, failed, status = LAST_LOG_ARGS
    user_pct = max(user_pct, pct_int)
    write_log(user_pct, suite_pct, suite, total, success, failed, status)


def start_log_heartbeat(interval: int = 5) -> tuple[threading.Event, threading.Thread]:
    stop = threading.Event()

    def loop() -> None:
        while not stop.wait(interval):
            refresh_log()

    thread = threading.Thread(target=loop, name="testlog-refresh", daemon=True)
    thread.start()
    return stop, thread


def stop_log_heartbeat(heartbeat: tuple[threading.Event, threading.Thread]) -> None:
    stop, thread = heartbeat
    stop.set()
    thread.join(timeout=2)


def write_error_log(code: str, requested_suite: str) -> None:
    with LOG_LOCK:
        user_pct, suite_pct, suite, total, success, failed, _ = LAST_LOG_ARGS
    if suite == "none":
        suite = requested_suite
    if suite in VALID_SUITES and suite_pct == 0:
        suite_pct = 100
    write_log(user_pct, suite_pct, suite, total, success, failed, f"error:{code}")


def ossutil_bin() -> str:
    explicit = os.environ.get("OSSUTIL_BIN")
    candidates = [explicit] if explicit else []
    candidates.extend(["ossutil", "ossutil64"])
    for name in candidates:
        if not name:
            continue
        resolved = shutil.which(name)
        if resolved:
            return resolved
    raise BenchmarkError("missing_ossutil", "ossutil/ossutil64 not found in PATH")


def require_env(names: Iterable[str], code: str) -> None:
    missing = [name for name in names if not os.environ.get(name)]
    if missing:
        raise BenchmarkError(code, "missing environment variables: " + ",".join(missing))


def oss_auth_args(env_kind: str) -> list[str]:
    if env_kind == "ckpt":
        require_env(CKPT_ENV, "missing_oss_env")
        endpoint = os.environ["BENCHMARK_CKPT_ENDPOINT"]
        ak = os.environ["BENCHMARK_CKPT_ACCESS_KEY_ID"]
        sk = os.environ["BENCHMARK_CKPT_ACCESS_KEY_SECRET"]
        token = os.environ.get("BENCHMARK_CKPT_SECURITY_TOKEN", "")
    else:
        require_env(BASE_ENV, "missing_oss_env")
        endpoint = os.environ["BENCHMARK_BASE_ENDPOINT"]
        ak = os.environ["BENCHMARK_BASE_ACCESS_KEY_ID"]
        sk = os.environ["BENCHMARK_BASE_ACCESS_KEY_SECRET"]
        token = os.environ.get("BENCHMARK_BASE_SECURITY_TOKEN", "")
    args = ["-e", endpoint, "--region", oss_region(endpoint), "-i", ak, "-k", sk]
    if token and token != "无":
        args.extend(["-t", token])
    return args


def oss_region(endpoint: str) -> str:
    """Resolve the OSS signing region for ossutil sign-v4 requests.

    Benchmark instances are provisioned without ``OSS_REGION``, so derive it
    from the endpoint host (``oss-<region>.aliyuncs.com``) when it is unset.
    """
    explicit = os.environ.get("OSS_REGION", "").strip()
    if explicit:
        return explicit
    host = endpoint.rstrip("/").split("//")[-1].split("/")[0]
    label = host.split(".")[0]
    if label.startswith("oss-"):
        region = label[len("oss-"):]
        if region:
            return region
    raise BenchmarkError("missing_oss_region", f"unable to resolve OSS region from endpoint: {endpoint}")


def oss_cp(
    src: str,
    dst: Path,
    env_kind: str,
    *,
    exclude_patterns: tuple[str, ...] = (),
) -> None:
    bin_path = ossutil_bin()
    dst.mkdir(parents=True, exist_ok=True)
    cmd = [bin_path, "cp", src, str(dst) + "/", "-r", *oss_auth_args(env_kind)]
    for pattern in exclude_patterns:
        cmd.extend(["--exclude", pattern])
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=0)
    output: list[str] = []
    assert proc.stdout is not None
    chunk = ""
    last_pct = -1
    while True:
        char = proc.stdout.read(1)
        if char == "" and proc.poll() is not None:
            break
        if char == "":
            time.sleep(0.1)
            continue
        chunk += char
        if char in ("\r", "\n"):
            output.append(chunk)
            match = PROGRESS_RE.search(chunk)
            if match:
                pct = float(match.group(1))
                pct_int = int(round(pct))
                if pct_int != last_pct:
                    update_download_progress(env_kind, pct)
                    last_pct = pct_int
            chunk = ""
            if len(output) > 200:
                output = output[-200:]
    if proc.wait() != 0:
        if chunk:
            output.append(chunk)
        raise BenchmarkError(f"{env_kind}_download_failed", "".join(output)[-2000:])
    update_download_progress(env_kind, 100)


def verify_oss_prefix(src: str, env_kind: str) -> None:
    result = subprocess.run(
        [ossutil_bin(), "ls", src, *oss_auth_args(env_kind)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise BenchmarkError(f"{env_kind}_access_failed", (result.stdout + result.stderr)[-2000:])


def load_asset_map() -> dict:
    if not ASSET_MAP.exists():
        raise BenchmarkError("missing_asset_map", f"missing {ASSET_MAP}")
    return json.loads(ASSET_MAP.read_text(encoding="utf-8"))


def find_checkpoint_layout(checkpoint_root: Path = LOCAL_CHECKPOINT_DIR) -> tuple[Path, Path]:
    candidates: list[tuple[float, Path, Path]] = []
    for cfg in checkpoint_root.rglob("config.yaml"):
        run_dir = cfg.parent
        stats = run_dir / "dataset_statistics.json"
        ckpt_dir = run_dir / "checkpoints"
        if not stats.exists() or not ckpt_dir.is_dir():
            continue
        ckpts = sorted(
            list(ckpt_dir.glob("*.pt")) + list(ckpt_dir.glob("*.safetensors")),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if ckpts:
            candidates.append((ckpts[0].stat().st_mtime, run_dir, ckpts[0]))
    if not candidates:
        raise BenchmarkError(
            "invalid_checkpoint_layout",
            "no run dir with config.yaml, dataset_statistics.json and checkpoints/*.pt found",
        )
    candidates.sort(reverse=True)
    return candidates[0][1], candidates[0][2]


def cached_checkpoint_layout() -> tuple[Path, Path] | None:
    try:
        return find_checkpoint_layout()
    except BenchmarkError as exc:
        if exc.code != "invalid_checkpoint_layout":
            raise
        return None


def stage_oss_checkpoint(source: str) -> tuple[Path, Path]:
    checkpoint_layout = None
    if os.environ.get("BENCHMARK_QUICK_TEST") == "1":
        verify_oss_prefix(source, "ckpt")
        checkpoint_layout = cached_checkpoint_layout()
    if checkpoint_layout is None:
        oss_cp(source, LOCAL_CHECKPOINT_DIR, "ckpt")
        return find_checkpoint_layout()
    update_download_progress("ckpt", 100)
    return checkpoint_layout


def stage_local_checkpoint(source: str) -> tuple[Path, Path]:
    source_path = Path(source)
    if not source_path.is_absolute():
        raise BenchmarkError("invalid_checkpoint_source", source)
    if not source_path.is_dir():
        raise BenchmarkError("local_checkpoint_missing", source)
    update_download_progress("ckpt", 100)
    return find_checkpoint_layout(source_path)


def stage_checkpoint(source: str) -> tuple[Path, Path]:
    if source.startswith("oss://"):
        return stage_oss_checkpoint(source)
    if os.environ.get("BENCHMARK_QUICK_TEST") == "1":
        raise BenchmarkError("invalid_checkpoint_source", "quick test requires an OSS checkpoint")
    return stage_local_checkpoint(source)


def cached_policy_checkpoint(source: str) -> Path | None:
    if not POLICY_CHECKPOINT_SOURCE_FILE.is_file():
        return None
    try:
        cache_source = json.loads(POLICY_CHECKPOINT_SOURCE_FILE.read_text(encoding="utf-8"))["source"]
    except (json.JSONDecodeError, KeyError, TypeError):
        return None
    has_model_weights = any(
        path.is_file()
        for pattern in MODEL_WEIGHT_PATTERNS
        if not pattern.endswith(".index.json")
        for path in LOCAL_CHECKPOINT_DIR.rglob(pattern)
    )
    if cache_source != source or not has_model_weights:
        return None
    update_download_progress("ckpt", 100)
    return LOCAL_CHECKPOINT_DIR


def record_policy_checkpoint_source(source: str) -> None:
    tmp = POLICY_CHECKPOINT_SOURCE_FILE.with_suffix(f".json.{os.getpid()}.tmp")
    tmp.write_text(json.dumps({"source": source}), encoding="utf-8")
    tmp.replace(POLICY_CHECKPOINT_SOURCE_FILE)


def stage_policy_checkpoint(source: str) -> Path:
    if source.startswith("oss://"):
        if os.environ.get("BENCHMARK_QUICK_TEST") == "1":
            verify_oss_prefix(source, "ckpt")
            cached = cached_policy_checkpoint(source)
            if cached is not None:
                return cached
            shutil.rmtree(LOCAL_CHECKPOINT_DIR, ignore_errors=True)
        oss_cp(source, LOCAL_CHECKPOINT_DIR, "ckpt")
        if os.environ.get("BENCHMARK_QUICK_TEST") == "1":
            record_policy_checkpoint_source(source)
        return LOCAL_CHECKPOINT_DIR
    if os.environ.get("BENCHMARK_QUICK_TEST") == "1":
        raise BenchmarkError("invalid_checkpoint_source", "quick test requires an OSS checkpoint")
    source_path = Path(source)
    if not source_path.is_absolute():
        raise BenchmarkError("invalid_checkpoint_source", source)
    if not source_path.exists():
        raise BenchmarkError("local_checkpoint_missing", source)
    update_download_progress("ckpt", 100)
    return source_path


def read_yaml(path: Path) -> dict:
    if yaml is None:
        raise BenchmarkError("missing_yaml", "PyYAML is not available in libero env")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise BenchmarkError("invalid_checkpoint_config", "config.yaml is not a mapping")
    return data


def infer_required_asset_ids(run_dir: Path) -> tuple[str, ...]:
    cfg = read_yaml(run_dir / "config.yaml")
    framework = cfg.get("framework") or {}
    qwenvl = framework.get("qwenvl") or {}
    base_vlm = qwenvl.get("base_vlm")
    if not base_vlm:
        raise BenchmarkError("missing_base_vlm", "framework.qwenvl.base_vlm not found in config.yaml")
    asset_ids = [Path(str(base_vlm).rstrip("/")).name]
    action_model = framework.get("action_model") or {}
    action_model_type = str(action_model.get("action_model_type") or "").lower()
    if "fast" in action_model_type:
        asset_ids.append("fast")
    return tuple(asset_ids)


def base_root_uri() -> str:
    root = os.environ.get("BENCHMARK_BASE_OSS_URI", "").strip()
    if not root:
        raise RuntimeError("BENCHMARK_BASE_OSS_URI is required to download base-model assets")
    return root.rstrip("/")


def link_or_replace(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.is_symlink() or dst.exists():
        if dst.is_dir() and not dst.is_symlink():
            shutil.rmtree(dst)
        else:
            dst.unlink()
    os.symlink(src, dst)


def ensure_base_asset(asset_id: str, asset_cfg: dict) -> None:
    src_uri = f"{base_root_uri()}/{asset_cfg.get('oss_subdir', asset_id).strip('/')}/"
    local_dir = LOCAL_BASE_MODEL_DIR / asset_id
    target = Path(asset_cfg["target_path"])
    if target.exists():
        return
    if not local_dir.exists() or not any(local_dir.iterdir()):
        oss_cp(src_uri, local_dir, "base", exclude_patterns=MODEL_WEIGHT_PATTERNS)
    link_or_replace(local_dir, target)


def ensure_required_assets(asset_ids: tuple[str, ...]) -> None:
    amap = load_asset_map()
    base_models = amap.get("base_models", {})
    assets = amap.get("assets", {})
    for asset_id in asset_ids:
        asset = base_models.get(asset_id) or assets.get(asset_id)
        if asset is None:
            raise BenchmarkError("unknown_base_model", f"{asset_id} not in base_assets.json")
        ensure_base_asset(asset_id, asset)


def count_tasks(suite: str) -> int:
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{LIBERO_HOME}:{env.get('PYTHONPATH', '')}"
    code = (
        "from libero.libero import benchmark; "
        f'task_suite=benchmark.get_benchmark_dict()["{suite}"](); '
        "print(task_suite.n_tasks)"
    )
    proc = subprocess.run([str(LIBERO_PYTHON), "-c", code], capture_output=True, text=True, env=env)
    if proc.returncode != 0:
        raise BenchmarkError("task_count_failed", proc.stderr[-1000:])
    nums = re.findall(r"\d+", proc.stdout)
    if not nums:
        raise BenchmarkError("task_count_failed", proc.stdout[-1000:] + proc.stderr[-1000:])
    return int(nums[-1])


def selected_task_count(num_tasks: int) -> int:
    raw = os.environ.get("BENCHMARK_TASK_IDS", "").strip()
    if not raw:
        return num_tasks
    try:
        task_ids = tuple(int(value.strip()) for value in raw.split(","))
    except ValueError:
        raise BenchmarkError("invalid_task_ids", raw) from None
    if not task_ids or len(task_ids) != len(set(task_ids)):
        raise BenchmarkError("invalid_task_ids", raw)
    if any(task_id < 0 or task_id >= num_tasks for task_id in task_ids):
        raise BenchmarkError("invalid_task_ids", raw)
    return len(task_ids)


def load_user_policy_config() -> object | None:
    if os.environ.get("ROBOSCOPE_BENCHMARK_RUNTIME", "starvla").strip() != "policy":
        return None
    try:
        from roboscope_kit import PolicyConfigError, load_policy_runtime_config
    except ImportError as exc:
        raise BenchmarkError("missing_policy_config_runtime", str(exc)) from exc
    try:
        config = load_policy_runtime_config()
    except PolicyConfigError as exc:
        raise BenchmarkError(exc.code, exc.message) from exc
    tmp = POLICY_CONFIG_RESULT.with_name(f"{POLICY_CONFIG_RESULT.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(config.summary(), ensure_ascii=False, sort_keys=True), encoding="utf-8")
    os.replace(tmp, POLICY_CONFIG_RESULT)
    return config


def run_evaluation(
    suite: str,
    eval_num: int,
    ckpt_path: Path,
    total: int,
    num_tasks: int,
    policy_config: object | None = None,
    timing_recorder: BenchmarkTimingRecorder | None = None,
) -> None:
    """Run high-throughput autotune evaluation while publishing product status."""
    runner_dir = Path(__file__).resolve().parent
    if str(runner_dir) not in sys.path:
        sys.path.insert(0, str(runner_dir))
    try:
        from roboscope_libero_autotune_eval import run_autotune_evaluation
    except ImportError as exc:
        raise BenchmarkError("missing_autotune_runtime", str(exc)) from exc

    output_dir = TESTRESULT_DIR / "autotune"

    def record_phase(field: str, event: str) -> None:
        if timing_recorder is None:
            return
        try:
            timing_recorder.handle_phase(field, event)
        except (RuntimeError, ValueError) as exc:
            LOGGER.warning(
                "benchmark timing phase update failed: field=%s event=%s error=%s",
                field,
                event,
                exc,
            )

    try:
        run_autotune_evaluation(
            suite=suite,
            eval_num=eval_num,
            ckpt_path=ckpt_path,
            total=total,
            num_tasks=num_tasks,
            output_dir=output_dir,
            video_dir=TESTVIDEO_DIR,
            write_log=write_log,
            policy_config=policy_config,
            phase_timing_callback=record_phase,
        )
    except BenchmarkError:
        raise
    except Exception as exc:
        raise BenchmarkError("evaluation_failed", str(exc)) from exc


def self_test() -> int:
    runner_dir = Path(__file__).resolve().parent
    if str(runner_dir) not in sys.path:
        sys.path.insert(0, str(runner_dir))
    from roboscope_libero_autotune_eval import run_sdk_self_test

    output = run_sdk_self_test()
    print(f"LIBERO benchmark self-test passed: {output}")
    return 0


def start(args: argparse.Namespace) -> int:
    suite = args.suite
    if suite not in VALID_SUITES:
        raise BenchmarkError("invalid_suite", f"unsupported suite: {suite}")
    try:
        eval_num = int(args.evaluation_num)
    except ValueError:
        raise BenchmarkError("invalid_evaluation_num", args.evaluation_num) from None
    if eval_num <= 0:
        raise BenchmarkError("invalid_evaluation_num", args.evaluation_num)
    if os.environ.get("ROBOSCOPE_BENCHMARK_SELF_TEST") == "1":
        return self_test()

    from roboscope_kit.runtime.benchmark import BenchmarkTimingRecorder

    ensure_dirs()
    TIMING_METRICS_FILE.unlink(missing_ok=True)
    num_tasks = count_tasks(suite)
    total = selected_task_count(num_tasks) * eval_num
    write_log(0, 100, suite, total, 0, 0, "downloading")
    timing_recorder = BenchmarkTimingRecorder(TIMING_METRICS_FILE)
    heartbeat = start_log_heartbeat()
    try:
        policy_config = load_user_policy_config()
        timing_recorder.start("checkpoint_prepare_s")
        if policy_config is not None:
            ckpt_path = stage_policy_checkpoint(args.checkpoint_source)
            timing_recorder.finish("checkpoint_prepare_s")
            write_log(100, 100, suite, total, 0, 0, "running")
            run_evaluation(
                suite,
                eval_num,
                ckpt_path,
                total,
                num_tasks,
                policy_config,
                timing_recorder,
            )
            return 0
        checkpoint_layout = stage_checkpoint(args.checkpoint_source)
        write_log(100, 100, suite, total, 0, 0, "downloading")

        run_dir, ckpt_path = checkpoint_layout
        asset_ids = infer_required_asset_ids(run_dir)
        if os.environ.get("BENCHMARK_QUICK_TEST") == "1":
            amap = load_asset_map()
            base_models = amap.get("base_models", {})
            assets = amap.get("assets", {})
            for asset_id in asset_ids:
                asset = base_models.get(asset_id) or assets.get(asset_id)
                if asset is None:
                    raise BenchmarkError("unknown_base_model", f"{asset_id} not in base_assets.json")
                asset_uri = f"{base_root_uri()}/{asset.get('oss_subdir', asset_id).strip('/')}/"
                verify_oss_prefix(asset_uri, "base")
        ensure_required_assets(asset_ids)
        timing_recorder.finish("checkpoint_prepare_s")
        write_log(100, 100, suite, total, 0, 0, "running")

        run_evaluation(suite, eval_num, ckpt_path, total, num_tasks, timing_recorder=timing_recorder)
    finally:
        stop_log_heartbeat(heartbeat)
        timing_recorder.close()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("start")
    p.add_argument("--suite", required=True)
    p.add_argument("--checkpoint-source", required=True)
    p.add_argument("--evaluation-num", required=True)
    sub.add_parser("self-test")
    args = parser.parse_args()

    if args.cmd == "self-test":
        return self_test()
    if os.environ.get("ROBOSCOPE_BENCHMARK_SELF_TEST") == "1":
        return start(args)
    ensure_dirs()
    try:
        if args.cmd == "start":
            return start(args)
        raise BenchmarkError("unknown_command", args.cmd)
    except BenchmarkError as exc:
        suite = getattr(args, "suite", "none") or "none"
        write_error_log(exc.code, suite)
        print(f"ERROR {exc.code}: {exc.message}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

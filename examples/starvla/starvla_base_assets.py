"""Prepare StarVLA base-model metadata required for framework construction."""

from __future__ import annotations

import fcntl
import os
import shutil
import subprocess
import tempfile
from collections.abc import Mapping
from pathlib import Path

import yaml

BASE_MODEL_SUBDIRECTORIES = {
    "Qwen2.5-VL-3B-Instruct": "Qwen2.5-VL-3B-Instruct",
    "Qwen2.5-VL-3B-Instruct-Action": "Qwen2.5-VL-3B-Instruct-Action",
    "Qwen3-VL-4B-Instruct": "Qwen3-VL-4B-Instruct",
}
BASE_ENVIRONMENT = (
    "BENCHMARK_BASE_ENDPOINT",
    "BENCHMARK_BASE_ACCESS_KEY_ID",
    "BENCHMARK_BASE_ACCESS_KEY_SECRET",
    "BENCHMARK_BASE_OSS_URI",
)
MODEL_WEIGHT_PATTERNS = (
    "*.safetensors",
    "*.safetensors.index.json",
    "*.bin",
    "*.bin.index.json",
    "*.pt",
    "*.pth",
    "*.ckpt",
)
DEFAULT_CACHE_ROOT = Path("/root/benchmark/basemodel")


def ensure_starvla_base_model(run_dir: str | Path) -> Path:
    """Ensure the configured StarVLA base-model metadata exists locally."""

    run_dir = Path(run_dir).resolve(strict=True)
    base_vlm = _configured_base_vlm(run_dir / "config.yaml")
    asset_id = Path(base_vlm.rstrip("/")).name
    oss_subdirectory = BASE_MODEL_SUBDIRECTORIES.get(asset_id)
    if oss_subdirectory is None:
        raise ValueError(f"unsupported StarVLA base model: {asset_id}")
    target = _resolve_snapshot_target(run_dir, base_vlm)
    if _base_model_ready(target):
        return target

    cache_root = Path(os.environ.get("ROBOSCOPE_STARVLA_BASE_CACHE", str(DEFAULT_CACHE_ROOT)))
    cache_root.mkdir(parents=True, exist_ok=True)
    cache = cache_root / asset_id
    lock_path = cache_root / f".{asset_id}.lock"
    with lock_path.open("a", encoding="utf-8") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not _base_model_ready(cache):
            _download_base_model(cache_root, cache, asset_id, oss_subdirectory)
        _link_cache(cache, target)
    return target


def _configured_base_vlm(config_path: Path) -> str:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, Mapping):
        raise ValueError("StarVLA config.yaml must contain a mapping")
    framework = config.get("framework")
    if not isinstance(framework, Mapping):
        raise ValueError("StarVLA config is missing framework")
    qwenvl = framework.get("qwenvl")
    if not isinstance(qwenvl, Mapping):
        raise ValueError("StarVLA config is missing framework.qwenvl")
    base_vlm = qwenvl.get("base_vlm")
    if not isinstance(base_vlm, str) or not base_vlm.strip():
        raise ValueError("StarVLA config is missing framework.qwenvl.base_vlm")
    return base_vlm.strip()


def _resolve_snapshot_target(run_dir: Path, configured_path: str) -> Path:
    path = Path(configured_path).expanduser()
    if path.is_absolute():
        raise ValueError("StarVLA base_vlm must be relative to the run snapshot")
    target_parent = (run_dir / path).parent.resolve()
    try:
        target_parent.relative_to(run_dir)
    except ValueError as exc:
        raise ValueError("StarVLA base_vlm must stay inside the run snapshot") from exc
    return target_parent / path.name


def _base_model_ready(path: Path) -> bool:
    return path.is_dir() and (path / "config.json").is_file()


def _oss_region(endpoint: str) -> str:
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
    raise RuntimeError(f"unable to resolve OSS region from endpoint: {endpoint}")


def _download_base_model(
    cache_root: Path,
    cache: Path,
    asset_id: str,
    oss_subdirectory: str,
) -> None:
    missing = [name for name in BASE_ENVIRONMENT if not os.environ.get(name)]
    if missing:
        raise RuntimeError("missing StarVLA base-model environment: " + ",".join(missing))
    ossutil = _ossutil_bin()
    endpoint = os.environ["BENCHMARK_BASE_ENDPOINT"]
    source = f"{os.environ['BENCHMARK_BASE_OSS_URI'].strip().rstrip('/')}/{oss_subdirectory.strip('/')}/"
    staging = Path(tempfile.mkdtemp(prefix=f".{asset_id}-", dir=cache_root))
    try:
        command = [
            ossutil,
            "cp",
            source,
            f"{staging}/",
            "-r",
            "-e",
            endpoint,
            "--region",
            _oss_region(endpoint),
            "-i",
            os.environ["BENCHMARK_BASE_ACCESS_KEY_ID"],
            "-k",
            os.environ["BENCHMARK_BASE_ACCESS_KEY_SECRET"],
        ]
        token = os.environ.get("BENCHMARK_BASE_SECURITY_TOKEN", "")
        if token and token != "无":
            command.extend(["-t", token])
        for pattern in MODEL_WEIGHT_PATTERNS:
            command.extend(["--exclude", pattern])
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode != 0:
            output = (result.stdout + result.stderr)[-2000:]
            raise RuntimeError(f"StarVLA base-model download failed: {output}")
        if not _base_model_ready(staging):
            raise RuntimeError(f"StarVLA base-model metadata is incomplete: {source}")
        if cache.is_symlink() or cache.is_file():
            cache.unlink()
        elif cache.exists():
            shutil.rmtree(cache)
        staging.rename(cache)
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def _ossutil_bin() -> str:
    explicit = os.environ.get("OSSUTIL_BIN")
    candidates = [explicit] if explicit else []
    candidates.extend(("ossutil", "ossutil64"))
    for candidate in candidates:
        if not candidate:
            continue
        resolved = shutil.which(candidate)
        if resolved:
            return resolved
    raise RuntimeError("ossutil/ossutil64 not found in PATH")


def _link_cache(cache: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_symlink() or target.is_file():
        target.unlink()
    elif target.exists():
        shutil.rmtree(target)
    target.symlink_to(cache, target_is_directory=True)

"""Export one offline-loadable StarVLA run snapshot for evaluation."""

from __future__ import annotations

import argparse
import gc
import json
import os
import shutil
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from starvla_full_checkpoint import CHECKPOINT_SUFFIXES, load_full_checkpoint
from starvla_policy import parse_inference_config, resolve_starvla_run_snapshot

OFFLINE_ENVIRONMENT = ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE")


@dataclass
class ResourceCopier:
    output: Path
    resource_root: Path
    copied_sources: dict[Path, Path] = field(default_factory=dict)
    destination_sources: dict[Path, Path] = field(default_factory=dict)

    def rewrite(self, value: Any, key_path: tuple[str, ...]) -> Any:
        if isinstance(value, dict):
            return {key: self.rewrite(child, (*key_path, str(key))) for key, child in value.items()}
        if isinstance(value, list):
            return [self.rewrite(child, (*key_path, str(index))) for index, child in enumerate(value)]
        if not isinstance(value, str):
            return value
        source = self._local_source(value, key_path)
        if source is None:
            return value
        relative = self._copy(source, value, key_path)
        return f"./{relative.as_posix()}"

    def _local_source(self, value: str, key_path: tuple[str, ...]) -> Path | None:
        path = Path(value).expanduser()
        candidate = path if path.is_absolute() else self.resource_root / path
        if candidate.exists():
            return candidate.resolve()
        if value.startswith(("/", "./", "../", "~/")):
            dotted = ".".join(key_path)
            raise FileNotFoundError(f"config resource does not exist for {dotted}: {candidate}")
        return None

    def _copy(self, source: Path, original: str, key_path: tuple[str, ...]) -> Path:
        if source in self.copied_sources:
            return self.copied_sources[source]
        relative = _portable_relative_path(original, source, self.resource_root, key_path)
        other_source = self.destination_sources.get(relative)
        if other_source is not None and other_source != source:
            raise ValueError(f"config resources collide at {relative}: {other_source} and {source}")
        destination = self.output / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if source.is_dir():
            shutil.copytree(source, destination)
        elif source.is_file():
            shutil.copy2(source, destination)
        else:
            raise ValueError(f"config resource must be a file or directory: {source}")
        self.copied_sources[source] = relative
        self.destination_sources[relative] = source
        return relative


def export_eval_run(
    *,
    run_dir: Path,
    checkpoint: Path,
    output: Path,
    resource_root: Path,
) -> Path:
    run_dir = run_dir.resolve(strict=True)
    checkpoint = checkpoint.resolve(strict=True)
    resource_root = resource_root.resolve(strict=True)
    _validate_source_run(run_dir, checkpoint)
    config = _read_config(run_dir / "config.yaml")
    parse_inference_config(config)

    output = output.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
    try:
        _write_snapshot(
            run_dir=run_dir,
            checkpoint=checkpoint,
            output=staging,
            resource_root=resource_root,
            config=config,
        )
        exported_checkpoint, exported_run = resolve_starvla_run_snapshot(staging)
        if exported_run != staging or exported_checkpoint.name != checkpoint.name:
            raise RuntimeError("exported StarVLA snapshot layout is invalid")
        _strict_load_preflight(staging, exported_checkpoint)
        staging.rename(output)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return output


def _validate_source_run(run_dir: Path, checkpoint: Path) -> None:
    config_path = run_dir / "config.yaml"
    statistics_path = run_dir / "dataset_statistics.json"
    if not config_path.is_file() or not statistics_path.is_file():
        raise ValueError("StarVLA run must contain config.yaml and dataset_statistics.json")
    expected_checkpoint_dir = (run_dir / "checkpoints").resolve()
    if checkpoint.parent != expected_checkpoint_dir or checkpoint.suffix.lower() not in CHECKPOINT_SUFFIXES:
        raise ValueError("selected checkpoint must be a .pt or .safetensors file in run/checkpoints")
    statistics = json.loads(statistics_path.read_text(encoding="utf-8"))
    if not isinstance(statistics, Mapping) or not statistics:
        raise ValueError("dataset_statistics.json must contain a non-empty mapping")


def _read_config(path: Path) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("StarVLA config.yaml must contain a mapping")
    return config


def _write_snapshot(
    *,
    run_dir: Path,
    checkpoint: Path,
    output: Path,
    resource_root: Path,
    config: dict[str, Any],
) -> None:
    shutil.copy2(run_dir / "dataset_statistics.json", output / "dataset_statistics.json")
    checkpoint_output = output / "checkpoints" / checkpoint.name
    checkpoint_output.parent.mkdir()
    shutil.copy2(checkpoint, checkpoint_output)
    copier = ResourceCopier(output=output, resource_root=resource_root)
    for section_name in ("framework", "inference"):
        section = config.get(section_name)
        if isinstance(section, dict):
            config[section_name] = copier.rewrite(section, (section_name,))
    parse_inference_config(config)
    (output / "config.yaml").write_text(
        yaml.safe_dump(config, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )


def _portable_relative_path(
    original: str,
    source: Path,
    resource_root: Path,
    key_path: tuple[str, ...],
) -> Path:
    path = Path(original).expanduser()
    if not path.is_absolute() and ".." not in path.parts:
        parts = tuple(part for part in path.parts if part not in (".", ""))
        if parts:
            return Path(*parts)
    try:
        return source.relative_to(resource_root)
    except ValueError:
        safe_key = "-".join(part.replace("_", "-") for part in key_path)
        return Path("resources") / safe_key / source.name


def _strict_load_preflight(run_dir: Path, checkpoint: Path) -> None:
    original_cwd = Path.cwd()
    original_environment = {name: os.environ.get(name) for name in OFFLINE_ENVIRONMENT}
    try:
        os.chdir(run_dir)
        for name in OFFLINE_ENVIRONMENT:
            os.environ[name] = "1"
        loaded = load_full_checkpoint(checkpoint)
        del loaded
        gc.collect()
    finally:
        os.chdir(original_cwd)
        for name, value in original_environment.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resource-root", type=Path, default=Path.cwd())
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = export_eval_run(
        run_dir=args.run_dir,
        checkpoint=args.checkpoint,
        output=args.output,
        resource_root=args.resource_root,
    )
    print(output)


if __name__ == "__main__":
    main()

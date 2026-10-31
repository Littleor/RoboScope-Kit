"""Download one demo's StarVLA checkpoint and merge the demo configuration into its config.yaml."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

DEMO_ROOT = Path(__file__).resolve().parent


def load_demo(benchmark: str) -> dict[str, Any]:
    path = DEMO_ROOT / benchmark / "demo.yaml"
    if not path.is_file():
        raise SystemExit(f"no StarVLA demo for benchmark {benchmark!r}")
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def merge(base: dict[str, Any], overlay: Mapping[str, Any]) -> dict[str, Any]:
    for key, value in overlay.items():
        if isinstance(value, Mapping) and isinstance(base.get(key), Mapping):
            base[key] = merge(dict(base[key]), value)
        else:
            base[key] = value
    return base


def prepare(benchmark: str, output: Path) -> Path:
    from huggingface_hub import snapshot_download

    demo = load_demo(benchmark)
    checkpoint = demo["checkpoint"]
    snapshot_download(
        repo_id=checkpoint["repo"],
        revision=checkpoint["revision"],
        local_dir=output,
        allow_patterns=["config.yaml", "dataset_statistics.json", "checkpoints/*"],
    )
    config_path = output / "config.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config = merge(config, demo["config"])
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("benchmark", help="demo directory name, for example libero or robotwin")
    parser.add_argument("output", type=Path, help="run snapshot directory to create")
    args = parser.parse_args()
    print(prepare(args.benchmark, args.output.resolve()))


if __name__ == "__main__":
    main()

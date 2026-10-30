# Copyright 2026 RoboScope Team
# SPDX-License-Identifier: Apache-2.0

"""Factory for the official CALVIN benchmark adapter."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from roboscope_kit.runtime.benchmark.factories import BenchmarkRunRequest
from roboscope_kit.runtime.benchmark.image_manifest import BenchmarkImageManifest

from .adapter import CALVIN_OFFICIAL_SUITES, CalvinBenchmarkAdapter, CalvinRuntimeConfig


class CalvinBenchmarkFactory:
    @staticmethod
    def validate_suite(suite: str) -> None:
        if suite not in CALVIN_OFFICIAL_SUITES:
            supported = ", ".join(sorted(CALVIN_OFFICIAL_SUITES))
            raise ValueError(f"calvin does not support suite {suite!r}; supported suites: {supported}")

    def create(
        self,
        request: BenchmarkRunRequest,
        image: BenchmarkImageManifest,
    ) -> CalvinBenchmarkAdapter:
        return self._create(request, image, self._episodes(request, image))

    def create_probe(
        self,
        request: BenchmarkRunRequest,
        image: BenchmarkImageManifest,
        concurrency: int,
    ) -> CalvinBenchmarkAdapter:
        del concurrency
        sequences = int(image.defaults.get("probe_sequence_count", 10))
        return self._create(request, image, sequences)

    def self_test(self, image: BenchmarkImageManifest, output_dir: Path) -> dict[str, Any]:
        runtime = self._runtime(image, next(iter(CALVIN_OFFICIAL_SUITES)))
        dataset_paths = self._dataset_paths(image)
        missing = [str(path) for path in (*runtime.values(), *dataset_paths.values()) if not path.exists()]
        if missing:
            raise RuntimeError(f"CALVIN image paths are missing: {', '.join(missing)}")
        output_dir.mkdir(parents=True, exist_ok=True)
        return {"benchmark": "calvin", "paths": [str(path) for path in runtime.values()]}

    def _create(
        self,
        request: BenchmarkRunRequest,
        image: BenchmarkImageManifest,
        episodes: int,
    ) -> CalvinBenchmarkAdapter:
        self.validate_suite(request.suite)
        runtime = self._runtime(image, request.suite)
        return CalvinBenchmarkAdapter(
            CalvinRuntimeConfig(
                python=runtime["python"],
                calvin_root=runtime["benchmark_root"],
                dataset_path=runtime["dataset_path"],
                config_path=runtime["config_path"],
                eval_sequences_path=runtime["eval_sequences_path"],
                suite=request.suite,
                num_sequences=episodes,
                chunk_size=int(image.defaults.get("chunk_size", 1)),
                cpu_threads=int(image.defaults.get("client_cpu_threads", 1)),
                render_backend=str(image.defaults.get("render_backend", "egl")),
            )
        )

    @staticmethod
    def _runtime(image: BenchmarkImageManifest, suite: str) -> dict[str, Path]:
        runtime = {
            name: _required_path(image.runtime, name)
            for name in (
                "python",
                "benchmark_root",
                "config_path",
                "eval_sequences_path",
            )
        }
        runtime["asset_root"] = runtime["benchmark_root"] / "calvin_env" / "data"
        runtime["dataset_path"] = CalvinBenchmarkFactory._dataset_paths(image)[suite]
        return runtime

    @staticmethod
    def _dataset_paths(image: BenchmarkImageManifest) -> dict[str, Path]:
        raw = image.runtime.get("dataset_paths")
        if not isinstance(raw, dict):
            raise ValueError("CALVIN runtime dataset_paths must be an object")
        if set(raw) != set(CALVIN_OFFICIAL_SUITES):
            raise ValueError("CALVIN runtime dataset_paths must define every supported suite")
        return {suite: _required_path(raw, suite) for suite in CALVIN_OFFICIAL_SUITES}

    @staticmethod
    def _episodes(request: BenchmarkRunRequest, image: BenchmarkImageManifest) -> int:
        if request.quick_test:
            return 1
        value = request.episodes or int(image.defaults.get("episodes", 1000))
        if value < 1:
            raise ValueError("CALVIN sequence count must be positive")
        return value


def _required_path(runtime: dict[str, Any], name: str) -> Path:
    value = Path(str(runtime.get(name) or ""))
    if not value.is_absolute():
        raise ValueError(f"CALVIN runtime {name} must be an absolute path")
    return value


__all__ = ["CalvinBenchmarkFactory"]

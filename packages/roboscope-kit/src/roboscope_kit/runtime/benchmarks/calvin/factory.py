# Copyright 2026 RoboScope Team
# SPDX-License-Identifier: Apache-2.0

"""Factory for the official CALVIN benchmark adapter."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from roboscope_kit.runtime.benchmark.autotune import probe_items_per_worker
from roboscope_kit.runtime.benchmark.factories import BenchmarkRunRequest
from roboscope_kit.runtime.benchmark.manifest import BenchmarkManifest

from .adapter import (
    CALVIN_OFFICIAL_SUITES,
    OFFICIAL_EVAL_SEQUENCE_COUNT,
    CalvinBenchmarkAdapter,
    CalvinRuntimeConfig,
)


class CalvinBenchmarkFactory:
    @staticmethod
    def validate_suite(suite: str) -> None:
        if suite not in CALVIN_OFFICIAL_SUITES:
            supported = ", ".join(sorted(CALVIN_OFFICIAL_SUITES))
            raise ValueError(f"calvin does not support suite {suite!r}; supported suites: {supported}")

    def create(
        self,
        request: BenchmarkRunRequest,
        manifest: BenchmarkManifest,
    ) -> CalvinBenchmarkAdapter:
        return self._create(request, manifest, self._episodes(request, manifest))

    def create_probe(
        self,
        request: BenchmarkRunRequest,
        manifest: BenchmarkManifest,
        concurrency: int,
    ) -> CalvinBenchmarkAdapter:
        sequences = int(manifest.defaults.get("probe_sequence_count", 10))
        # Give every worker items_per_worker items. The probe takes a prefix of the
        # evaluation sequences, so each probe repeats the sequences of every smaller one.
        items_per_worker = probe_items_per_worker(manifest.defaults)
        if items_per_worker > 0:
            chunk_size = int(manifest.defaults.get("chunk_size", 1))
            sequences = max(sequences, items_per_worker * concurrency * chunk_size)
            sequences = min(sequences, OFFICIAL_EVAL_SEQUENCE_COUNT)
        return self._create(request, manifest, sequences)

    def self_test(self, manifest: BenchmarkManifest, output_dir: Path) -> dict[str, Any]:
        runtime = self._runtime(manifest, next(iter(CALVIN_OFFICIAL_SUITES)))
        dataset_paths = self._dataset_paths(manifest)
        missing = [str(path) for path in (*runtime.values(), *dataset_paths.values()) if not path.exists()]
        if missing:
            raise RuntimeError(f"CALVIN runtime paths are missing: {', '.join(missing)}")
        output_dir.mkdir(parents=True, exist_ok=True)
        return {"benchmark": "calvin", "paths": [str(path) for path in runtime.values()]}

    def _create(
        self,
        request: BenchmarkRunRequest,
        manifest: BenchmarkManifest,
        episodes: int,
    ) -> CalvinBenchmarkAdapter:
        self.validate_suite(request.suite)
        runtime = self._runtime(manifest, request.suite)
        return CalvinBenchmarkAdapter(
            CalvinRuntimeConfig(
                python=runtime["python"],
                calvin_root=runtime["benchmark_root"],
                dataset_path=runtime["dataset_path"],
                config_path=runtime["config_path"],
                eval_sequences_path=runtime["eval_sequences_path"],
                suite=request.suite,
                num_sequences=episodes,
                chunk_size=int(manifest.defaults.get("chunk_size", 1)),
                cpu_threads=int(manifest.defaults.get("client_cpu_threads", 1)),
                render_backend=str(manifest.defaults.get("render_backend", "egl")),
            )
        )

    @staticmethod
    def _runtime(manifest: BenchmarkManifest, suite: str) -> dict[str, Path]:
        runtime = {
            name: manifest.runtime_path(name)
            for name in (
                "python",
                "benchmark_root",
                "config_path",
                "eval_sequences_path",
            )
        }
        runtime["asset_root"] = runtime["benchmark_root"] / "calvin_env" / "data"
        runtime["dataset_path"] = CalvinBenchmarkFactory._dataset_paths(manifest)[suite]
        return runtime

    @staticmethod
    def _dataset_paths(manifest: BenchmarkManifest) -> dict[str, Path]:
        raw = manifest.runtime.get("dataset_paths")
        if not isinstance(raw, dict):
            raise ValueError("CALVIN runtime dataset_paths must be an object")
        if set(raw) != set(CALVIN_OFFICIAL_SUITES):
            raise ValueError("CALVIN runtime dataset_paths must define every supported suite")
        return {
            suite: manifest.resolve(raw[suite], f"dataset_paths.{suite}") for suite in CALVIN_OFFICIAL_SUITES
        }

    @staticmethod
    def _episodes(request: BenchmarkRunRequest, manifest: BenchmarkManifest) -> int:
        if request.quick_test:
            return 1
        value = request.episodes or int(manifest.defaults.get("episodes", 1000))
        if value < 1:
            raise ValueError("CALVIN sequence count must be positive")
        return value



__all__ = ["CalvinBenchmarkFactory"]

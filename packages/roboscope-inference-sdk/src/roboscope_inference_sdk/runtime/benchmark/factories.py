# Copyright 2026 RoboScope Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Factory contracts for constructing platform-owned benchmark adapters."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from roboscope_inference_sdk.benchmark_specs import BenchmarkName

from .adapters import BenchmarkAdapter
from .image_manifest import BenchmarkImageManifest


@dataclass(frozen=True)
class BenchmarkRunRequest:
    benchmark: BenchmarkName
    suite: str
    checkpoint_path: Path
    output_dir: Path
    episodes: int | None
    quick_test: bool
    evaluation_config_path: Path | None = None


class BenchmarkFactory(Protocol):
    def validate_suite(self, suite: str) -> None: ...

    def create(
        self,
        request: BenchmarkRunRequest,
        image: BenchmarkImageManifest,
    ) -> BenchmarkAdapter: ...

    def create_probe(
        self,
        request: BenchmarkRunRequest,
        image: BenchmarkImageManifest,
        concurrency: int,
    ) -> BenchmarkAdapter: ...

    def self_test(self, image: BenchmarkImageManifest, output_dir: Path) -> dict: ...


def representative_indices(total: int, count: int) -> tuple[int, ...]:
    """Select deterministic indices spanning a benchmark workload."""

    if total < 1 or count < 1:
        raise ValueError("probe workload size must be positive")
    selected_count = min(total, count)
    return tuple(index * total // selected_count for index in range(selected_count))


__all__ = ["BenchmarkFactory", "BenchmarkRunRequest", "representative_indices"]

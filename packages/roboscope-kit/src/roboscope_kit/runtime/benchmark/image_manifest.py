# Copyright 2026 RoboScope Team
# SPDX-License-Identifier: Apache-2.0

"""Immutable benchmark-image metadata used by the unified runtime entrypoint."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from roboscope_kit.benchmark_specs import BenchmarkName

BENCHMARK_IMAGE_MANIFEST_PATH = Path("/opt/roboscope/benchmark-image.json")
BENCHMARK_IMAGE_SCHEMA_VERSION = 2


@dataclass(frozen=True)
class BenchmarkImageManifest:
    path: Path
    image_name: str
    evaluation_kit_commit: str
    sdk_version: str
    supported_benchmarks: tuple[BenchmarkName, ...]
    protocol_versions: dict[BenchmarkName, str]
    supported_suites: dict[BenchmarkName, tuple[str, ...]]
    runtime: dict[str, Any]
    defaults: dict[str, Any]
    benchmark_protocols: dict[BenchmarkName, dict[str, Any]] = field(default_factory=dict)

    def validate_request(self, benchmark: BenchmarkName, suite: str | None) -> None:
        if benchmark not in self.supported_benchmarks:
            raise ValueError(
                f"benchmark image {self.image_name!r} does not support {benchmark.value!r}"
            )
        suites = self.supported_suites.get(benchmark, ())
        if suite is not None and suite not in suites:
            raise ValueError(
                f"benchmark image {self.image_name!r} does not support suite {suite!r} "
                f"for {benchmark.value!r}"
            )


def load_benchmark_image_manifest(
    path: str | Path = BENCHMARK_IMAGE_MANIFEST_PATH,
) -> BenchmarkImageManifest:
    manifest_path = Path(path)
    value = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("benchmark image manifest root must be an object")
    expected = {
        "schema_version",
        "image_name",
        "evaluation_kit_commit",
        "sdk_version",
        "supported_benchmarks",
        "protocol_versions",
        "supported_suites",
        "runtime",
        "defaults",
    }
    schema_version = value["schema_version"]
    if schema_version != BENCHMARK_IMAGE_SCHEMA_VERSION:
        raise ValueError(
            f"benchmark image schema_version must be {BENCHMARK_IMAGE_SCHEMA_VERSION}"
        )
    expected.add("benchmark_protocols")
    if set(value) != expected:
        raise ValueError(f"benchmark image manifest fields do not match schema version {schema_version}")
    benchmarks = tuple(BenchmarkName(item) for item in value["supported_benchmarks"])
    if not benchmarks or len(benchmarks) != len(set(benchmarks)):
        raise ValueError("supported_benchmarks must contain unique benchmark names")
    protocol_versions = {
        BenchmarkName(name): str(version) for name, version in value["protocol_versions"].items()
    }
    supported_suites = {
        BenchmarkName(name): tuple(str(suite) for suite in suites)
        for name, suites in value["supported_suites"].items()
    }
    if set(protocol_versions) != set(benchmarks):
        raise ValueError("protocol_versions must cover every supported benchmark exactly")
    if set(supported_suites) != set(benchmarks):
        raise ValueError("supported_suites must cover every supported benchmark exactly")
    if any(not suites or len(suites) != len(set(suites)) for suites in supported_suites.values()):
        raise ValueError("every supported benchmark must declare unique suites")
    benchmark_protocols = {
        BenchmarkName(name): dict(protocol)
        for name, protocol in value["benchmark_protocols"].items()
    }
    if set(benchmark_protocols) - set(benchmarks):
        raise ValueError("benchmark_protocols contains an unsupported benchmark")
    runtime = value["runtime"]
    defaults = value["defaults"]
    if not isinstance(runtime, dict) or not isinstance(defaults, dict):
        raise ValueError("runtime and defaults must be objects")
    return BenchmarkImageManifest(
        path=manifest_path,
        image_name=_required_text(value, "image_name"),
        evaluation_kit_commit=_required_text(value, "evaluation_kit_commit"),
        sdk_version=_required_text(value, "sdk_version"),
        supported_benchmarks=benchmarks,
        protocol_versions=protocol_versions,
        supported_suites=supported_suites,
        runtime=dict(runtime),
        defaults=dict(defaults),
        benchmark_protocols=benchmark_protocols,
    )


def _required_text(value: dict[str, Any], name: str) -> str:
    result = str(value[name]).strip()
    if not result:
        raise ValueError(f"benchmark image manifest {name} cannot be empty")
    return result


__all__ = [
    "BENCHMARK_IMAGE_MANIFEST_PATH",
    "BenchmarkImageManifest",
    "load_benchmark_image_manifest",
]

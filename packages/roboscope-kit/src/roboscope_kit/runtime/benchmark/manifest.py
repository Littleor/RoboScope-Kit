# Copyright 2026 RoboScope Team
# SPDX-License-Identifier: Apache-2.0

"""The benchmark manifest: suites, protocol, runtime paths, and evaluation defaults."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from roboscope_kit.benchmark_specs import BenchmarkName

BENCHMARK_MANIFEST_SCHEMA_VERSION = 3


@dataclass(frozen=True)
class BenchmarkManifest:
    path: Path
    supported_benchmarks: tuple[BenchmarkName, ...]
    protocol_versions: dict[BenchmarkName, str]
    supported_suites: dict[BenchmarkName, tuple[str, ...]]
    runtime: dict[str, Any]
    defaults: dict[str, Any]
    benchmark_protocols: dict[BenchmarkName, dict[str, Any]] = field(default_factory=dict)

    def validate_request(self, benchmark: BenchmarkName, suite: str | None) -> None:
        if benchmark not in self.supported_benchmarks:
            raise ValueError(f"benchmark manifest {self.path} does not support {benchmark.value!r}")
        suites = self.supported_suites.get(benchmark, ())
        if suite is not None and suite not in suites:
            raise ValueError(
                f"benchmark manifest {self.path} does not support suite {suite!r} for {benchmark.value!r}"
            )


    def runtime_path(self, name: str, *, required: bool = True) -> Path | None:
        """Return ``runtime[name]`` as a path; a relative path is relative to the manifest's directory."""

        value = self.runtime.get(name)
        if value is None and not required:
            return None
        return self.resolve(value, name)

    def resolve(self, value: Any, name: str) -> Path:
        """Resolve one runtime path value against the manifest's directory."""

        text = str(value or "").strip()
        if not text:
            raise ValueError(f"benchmark manifest runtime {name} is required")
        path = Path(text)
        return path if path.is_absolute() else self.path.parent / path


def load_benchmark_manifest(path: str | Path) -> BenchmarkManifest:
    manifest_path = Path(path).resolve()
    value = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("benchmark manifest root must be an object")
    expected = {
        "schema_version",
        "supported_benchmarks",
        "protocol_versions",
        "supported_suites",
        "runtime",
        "defaults",
        "benchmark_protocols",
    }
    if value.get("schema_version") != BENCHMARK_MANIFEST_SCHEMA_VERSION:
        raise ValueError(f"benchmark manifest schema_version must be {BENCHMARK_MANIFEST_SCHEMA_VERSION}")
    if set(value) != expected:
        raise ValueError(f"benchmark manifest fields must be {sorted(expected)}")
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
    return BenchmarkManifest(
        path=manifest_path,
        supported_benchmarks=benchmarks,
        protocol_versions=protocol_versions,
        supported_suites=supported_suites,
        runtime=dict(runtime),
        defaults=dict(defaults),
        benchmark_protocols=benchmark_protocols,
    )


__all__ = [
    "BENCHMARK_MANIFEST_SCHEMA_VERSION",
    "BenchmarkManifest",
    "load_benchmark_manifest",
]

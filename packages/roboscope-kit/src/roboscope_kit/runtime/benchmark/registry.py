# Copyright 2026 RoboScope Team
# SPDX-License-Identifier: Apache-2.0

"""Explicit platform-owned benchmark factory registry."""

from __future__ import annotations

from roboscope_kit.benchmark_specs import BenchmarkName

from .factories import BenchmarkFactory


def benchmark_factory(name: BenchmarkName) -> BenchmarkFactory:
    if name is BenchmarkName.LIBERO:
        from roboscope_kit.runtime.benchmarks.libero import LiberoBenchmarkFactory

        return LiberoBenchmarkFactory()
    if name is BenchmarkName.LIBERO_PLUS:
        from roboscope_kit.runtime.benchmarks.libero_plus import LiberoPlusBenchmarkFactory

        return LiberoPlusBenchmarkFactory()
    if name is BenchmarkName.ROBOTWIN:
        from roboscope_kit.runtime.benchmarks.robotwin import RoboTwinBenchmarkFactory

        return RoboTwinBenchmarkFactory()
    if name is BenchmarkName.CALVIN:
        from roboscope_kit.runtime.benchmarks.calvin import CalvinBenchmarkFactory

        return CalvinBenchmarkFactory()
    if name is BenchmarkName.ROBOCASA:
        from roboscope_kit.runtime.benchmarks.robocasa import RoboCasaBenchmarkFactory

        return RoboCasaBenchmarkFactory()
    if name is BenchmarkName.METAWORLD:
        from roboscope_kit.runtime.benchmarks.metaworld import MetaworldBenchmarkFactory

        return MetaworldBenchmarkFactory()
    raise ValueError(f"unsupported benchmark: {name.value}")


__all__ = ["benchmark_factory"]

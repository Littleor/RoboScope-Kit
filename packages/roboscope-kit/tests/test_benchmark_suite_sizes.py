"""The suite-size table the platform shards by must agree with what the runtime actually runs."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from roboscope_kit.runtime.benchmark.suite_sizes import SUITE_SIZES, suite_total_records
from roboscope_kit.runtime.benchmarks.libero.factory import LIBERO_TASK_COUNTS
from roboscope_kit.runtime.benchmarks.metaworld.instructions import MT50_TASKS
from roboscope_kit.runtime.benchmarks.robocasa.adapter import ROBOCASA_TABLETOP_TASKS
from roboscope_kit.runtime.benchmarks.robotwin.adapter import ROBOTWIN_ALL_TASKS

MANIFESTS = Path(__file__).resolve().parents[3] / "benchmarks" / "starvla_platform_runtime"
# No fixed size: LIBERO+ samples its tasks per run.
UNSIZED_BENCHMARKS = {"libero_plus"}


def _manifest(benchmark: str) -> dict:
    return json.loads((MANIFESTS / benchmark / "benchmark-image.json").read_text(encoding="utf-8"))


@pytest.mark.parametrize("benchmark", sorted({benchmark for benchmark, _ in SUITE_SIZES}))
def test_episodes_per_task_and_suites_match_the_image_manifest(benchmark: str) -> None:
    manifest = _manifest(benchmark)
    listed = {suite for name, suite in SUITE_SIZES if name == benchmark}
    assert listed == set(manifest["supported_suites"][benchmark])
    for suite in listed:
        assert SUITE_SIZES[(benchmark, suite)].episodes_per_task == manifest["defaults"]["episodes"], suite


def test_every_image_manifest_is_sized_or_declared_unsized() -> None:
    manifests = {path.parent.name for path in MANIFESTS.glob("*/benchmark-image.json")}
    sized = {benchmark for benchmark, _ in SUITE_SIZES}
    assert manifests == sized | UNSIZED_BENCHMARKS


def test_task_counts_match_the_runtimes_task_lists() -> None:
    for suite, count in LIBERO_TASK_COUNTS.items():
        assert SUITE_SIZES[("libero", suite)].tasks == count
    assert SUITE_SIZES[("robocasa", "tabletop")].tasks == len(ROBOCASA_TABLETOP_TASKS)
    assert {SUITE_SIZES[("robotwin", suite)].tasks for suite in ("demo_clean", "demo_randomized")} == {
        len(ROBOTWIN_ALL_TASKS)
    }
    assert SUITE_SIZES[("metaworld", "mt50")].tasks == len(MT50_TASKS)


def test_suite_total_records() -> None:
    assert suite_total_records("libero", "libero_90") == 4500
    assert suite_total_records("robotwin", "demo_clean") == 5000
    assert suite_total_records("calvin", "task_ABC_D") == 1000
    assert suite_total_records("robocasa", "tabletop") == 1200
    assert suite_total_records("metaworld", "mt50") == 2500
    # No fixed size: never sharded.
    assert suite_total_records("libero_plus", "libero_spatial") is None
    assert suite_total_records("nonexistent", "suite") is None

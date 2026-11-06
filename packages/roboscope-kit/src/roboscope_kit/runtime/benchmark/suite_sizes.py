# Copyright 2026 RoboScope Team
# SPDX-License-Identifier: Apache-2.0

"""How many records one formal run of a suite produces, for callers that must decide how to
split it before any instance exists, for example to size a sharded run.

A record is one episode (CALVIN: one sequence). Most sizes are ``tasks * episodes_per_task``.
RoboDojo has different upstream episode limits by task, so its manifest sets ``defaults.episodes``
to null and its total is stated explicitly. The manifests are not part of the installed package,
so the totals are stated here. Suites whose size is not a fixed number (LIBERO+ samples its tasks per run) are
deliberately absent: they are never sharded.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SuiteSize:
    tasks: int
    episodes_per_task: int | None
    fixed_total_records: int | None = None

    @property
    def total_records(self) -> int:
        if self.fixed_total_records is not None:
            return self.fixed_total_records
        if self.episodes_per_task is None:
            raise ValueError("a variable-size suite requires a fixed total")
        return self.tasks * self.episodes_per_task


SUITE_SIZES: dict[tuple[str, str], SuiteSize] = {
    ("libero", "libero_spatial"): SuiteSize(tasks=10, episodes_per_task=50),
    ("libero", "libero_object"): SuiteSize(tasks=10, episodes_per_task=50),
    ("libero", "libero_goal"): SuiteSize(tasks=10, episodes_per_task=50),
    ("libero", "libero_10"): SuiteSize(tasks=10, episodes_per_task=50),
    ("libero", "libero_90"): SuiteSize(tasks=90, episodes_per_task=50),
    ("robotwin", "demo_clean"): SuiteSize(tasks=50, episodes_per_task=100),
    ("robotwin", "demo_randomized"): SuiteSize(tasks=50, episodes_per_task=100),
    ("calvin", "task_D_D"): SuiteSize(tasks=1, episodes_per_task=1000),
    ("calvin", "task_ABC_D"): SuiteSize(tasks=1, episodes_per_task=1000),
    ("calvin", "task_ABCD_D"): SuiteSize(tasks=1, episodes_per_task=1000),
    ("robocasa", "tabletop"): SuiteSize(tasks=24, episodes_per_task=50),
    ("metaworld", "mt50"): SuiteSize(tasks=50, episodes_per_task=50),
    ("robomimic", "ph_low_dim"): SuiteSize(tasks=5, episodes_per_task=50),
    ("robodojo", "arx_x5"): SuiteSize(tasks=42, episodes_per_task=None, fixed_total_records=1800),
    ("robodojo", "arx_x5_randomized"): SuiteSize(tasks=12, episodes_per_task=None, fixed_total_records=300),
}


def suite_total_records(benchmark: str, suite: str) -> int | None:
    """Records in one formal run of the suite, or ``None`` when it has no fixed size."""

    size = SUITE_SIZES.get((benchmark, suite))
    return None if size is None else size.total_records


__all__ = ["SUITE_SIZES", "SuiteSize", "suite_total_records"]

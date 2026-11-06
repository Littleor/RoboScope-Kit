# Copyright 2026 RoboScope Team
# SPDX-License-Identifier: Apache-2.0

"""Official LIBERO-Pro work manifest and SDK-native evaluator process adapter.

LIBERO-Pro is a fork of LIBERO on the same simulator stack, so evaluator processes run the shared
``runtime.benchmarks.libero.worker``. This adapter owns the LIBERO-Pro protocol: each suite is one base
LIBERO suite under one released single-perturbation set, runs at that base suite's horizon, and feeds
the Policy the instruction of the perturbed BDDL file.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any

from roboscope_kit.runtime.benchmark import ProcessSpec, WorkerContext
from roboscope_kit.runtime.benchmarks.libero.adapter import LiberoBenchmarkAdapter, LiberoRuntimeConfig


@dataclass(frozen=True)
class LiberoProSuite:
    base_suite: str
    perturbation: str
    dimension: str
    max_steps: int


# Policy action steps per episode of each base suite: the LIBERO horizons, which LIBERO-Pro keeps.
_BASE_MAX_STEPS = {
    "libero_goal": 300,
    "libero_spatial": 220,
    "libero_10": 520,
    "libero_object": 280,
}
# Suffix of each released single-perturbation suite and the generalization dimension it tests
# (upstream evaluation_config.yaml perturbation_mapping).
_PERTURBATIONS = {
    "object": "Object",
    "swap": "Position",
    "lan": "Semantic",
    "task": "Task",
}

# The 16 suites whose BDDL and initial-state files LIBERO-Pro released; Environment (``*_env``) and
# combined (``*_temp``) suites have no released files.
LIBERO_PRO_SUITES = {
    f"{base}_{perturbation}": LiberoProSuite(base, perturbation, dimension, max_steps)
    for base, max_steps in _BASE_MAX_STEPS.items()
    for perturbation, dimension in _PERTURBATIONS.items()
}


@dataclass(frozen=True)
class LiberoProRuntimeConfig(LiberoRuntimeConfig):
    def __post_init__(self) -> None:
        super().__post_init__()
        if self.suite not in LIBERO_PRO_SUITES:
            raise ValueError(f"unsupported LIBERO-Pro suite: {self.suite}")


class LiberoProBenchmarkAdapter(LiberoBenchmarkAdapter):
    """Run the LIBERO-Pro protocol against an SDK Policy process."""

    config: LiberoProRuntimeConfig

    def __init__(self, config: LiberoProRuntimeConfig) -> None:
        super().__init__(config)
        self.suite = LIBERO_PRO_SUITES[config.suite]

    def worker_process(self, context: WorkerContext) -> ProcessSpec:
        spec = super().worker_process(context)
        return dataclasses.replace(
            spec,
            command=(
                *spec.command,
                "--max-steps",
                str(self.suite.max_steps),
                "--instruction-source",
                "bddl",
            ),
        )

    def metadata(self) -> dict[str, Any]:
        return {
            **super().metadata(),
            "benchmark": "libero_pro",
            "base_suite": self.suite.base_suite,
            "perturbation": self.suite.perturbation,
            "dimension": self.suite.dimension,
            "max_steps": self.suite.max_steps,
            "instruction_source": "bddl",
        }


__all__ = ["LIBERO_PRO_SUITES", "LiberoProBenchmarkAdapter", "LiberoProRuntimeConfig", "LiberoProSuite"]

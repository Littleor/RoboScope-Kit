# Copyright 2026 RoboScope Team
# SPDX-License-Identifier: Apache-2.0

"""Run a benchmark Worker under Nsight Systems for ``--profile``."""

from __future__ import annotations

import shutil
from dataclasses import replace
from pathlib import Path

from roboscope_kit.profiling import DOMAIN, EPISODE_RANGE, PROFILE_ENV

from .adapters import ProcessSpec


def require_nsys() -> None:
    if shutil.which("nsys") is None:
        raise ValueError("--profile requires Nsight Systems: nsys is not on PATH")


def profiled_process(spec: ProcessSpec, report: Path) -> ProcessSpec:
    """Capture the Worker's first episode, including its child processes, into ``report``."""
    report.parent.mkdir(parents=True, exist_ok=True)
    return replace(
        spec,
        command=(
            "nsys", "profile",
            "--output", str(report),
            "--force-overwrite=true",
            "--trace=cuda,nvtx",
            "--sample=none",
            "--cpuctxsw=none",
            "--capture-range=nvtx",
            f"--nvtx-capture={EPISODE_RANGE}@{DOMAIN}",
            "--capture-range-end=stop",
            "--kill=none",
            *spec.command,
        ),
        env={**spec.env, PROFILE_ENV: "1", "NSYS_NVTX_PROFILER_REGISTER_ONLY": "0"},
    )

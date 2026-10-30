# Copyright 2026 RoboScope Team
# SPDX-License-Identifier: Apache-2.0

"""Official RoboCasa GR1 Tabletop adapter for SDK-native policies."""

from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from roboscope_kit.runtime.benchmark import ProcessSpec, WorkerContext, WorkItem, WorkResult

SDK_IMPORT_ROOT = Path(__file__).resolve().parents[4]
ROBOCASA_TABLETOP_SUITE = "tabletop"
ROBOCASA_TABLETOP_TASKS = (
    ("PnPBottleToCabinetClose", "gr1_unified/PnPBottleToCabinetClose_GR1ArmsAndWaistFourierHands_Env"),
    ("PnPCanToDrawerClose", "gr1_unified/PnPCanToDrawerClose_GR1ArmsAndWaistFourierHands_Env"),
    ("PnPCupToDrawerClose", "gr1_unified/PnPCupToDrawerClose_GR1ArmsAndWaistFourierHands_Env"),
    ("PnPMilkToMicrowaveClose", "gr1_unified/PnPMilkToMicrowaveClose_GR1ArmsAndWaistFourierHands_Env"),
    ("PnPPotatoToMicrowaveClose", "gr1_unified/PnPPotatoToMicrowaveClose_GR1ArmsAndWaistFourierHands_Env"),
    ("PnPWineToCabinetClose", "gr1_unified/PnPWineToCabinetClose_GR1ArmsAndWaistFourierHands_Env"),
    (
        "PosttrainCuttingboardToBasket",
        "gr1_unified/PosttrainPnPNovelFromCuttingboardToBasketSplitA_GR1ArmsAndWaistFourierHands_Env",
    ),
    (
        "PosttrainCuttingboardToCardboardbox",
        "gr1_unified/PosttrainPnPNovelFromCuttingboardToCardboardboxSplitA_GR1ArmsAndWaistFourierHands_Env",
    ),
    (
        "PosttrainCuttingboardToPan",
        "gr1_unified/PosttrainPnPNovelFromCuttingboardToPanSplitA_GR1ArmsAndWaistFourierHands_Env",
    ),
    (
        "PosttrainCuttingboardToPot",
        "gr1_unified/PosttrainPnPNovelFromCuttingboardToPotSplitA_GR1ArmsAndWaistFourierHands_Env",
    ),
    (
        "PosttrainCuttingboardToTieredbasket",
        "gr1_unified/PosttrainPnPNovelFromCuttingboardToTieredbasketSplitA_GR1ArmsAndWaistFourierHands_Env",
    ),
    (
        "PosttrainPlacematToBasket",
        "gr1_unified/PosttrainPnPNovelFromPlacematToBasketSplitA_GR1ArmsAndWaistFourierHands_Env",
    ),
    (
        "PosttrainPlacematToBowl",
        "gr1_unified/PosttrainPnPNovelFromPlacematToBowlSplitA_GR1ArmsAndWaistFourierHands_Env",
    ),
    (
        "PosttrainPlacematToPlate",
        "gr1_unified/PosttrainPnPNovelFromPlacematToPlateSplitA_GR1ArmsAndWaistFourierHands_Env",
    ),
    (
        "PosttrainPlacematToTieredshelf",
        "gr1_unified/PosttrainPnPNovelFromPlacematToTieredshelfSplitA_GR1ArmsAndWaistFourierHands_Env",
    ),
    (
        "PosttrainPlateToBowl",
        "gr1_unified/PosttrainPnPNovelFromPlateToBowlSplitA_GR1ArmsAndWaistFourierHands_Env",
    ),
    (
        "PosttrainPlateToCardboardbox",
        "gr1_unified/PosttrainPnPNovelFromPlateToCardboardboxSplitA_GR1ArmsAndWaistFourierHands_Env",
    ),
    (
        "PosttrainPlateToPan",
        "gr1_unified/PosttrainPnPNovelFromPlateToPanSplitA_GR1ArmsAndWaistFourierHands_Env",
    ),
    (
        "PosttrainPlateToPlate",
        "gr1_unified/PosttrainPnPNovelFromPlateToPlateSplitA_GR1ArmsAndWaistFourierHands_Env",
    ),
    (
        "PosttrainTrayToCardboardbox",
        "gr1_unified/PosttrainPnPNovelFromTrayToCardboardboxSplitA_GR1ArmsAndWaistFourierHands_Env",
    ),
    (
        "PosttrainTrayToPlate",
        "gr1_unified/PosttrainPnPNovelFromTrayToPlateSplitA_GR1ArmsAndWaistFourierHands_Env",
    ),
    (
        "PosttrainTrayToPot",
        "gr1_unified/PosttrainPnPNovelFromTrayToPotSplitA_GR1ArmsAndWaistFourierHands_Env",
    ),
    (
        "PosttrainTrayToTieredbasket",
        "gr1_unified/PosttrainPnPNovelFromTrayToTieredbasketSplitA_GR1ArmsAndWaistFourierHands_Env",
    ),
    (
        "PosttrainTrayToTieredshelf",
        "gr1_unified/PosttrainPnPNovelFromTrayToTieredshelfSplitA_GR1ArmsAndWaistFourierHands_Env",
    ),
)


@dataclass(frozen=True)
class RoboCasaRuntimeConfig:
    python: Path
    benchmark_root: Path
    suite: str
    tasks: tuple[tuple[str, str], ...]
    episodes_per_task: int
    action_steps: int = 12
    max_episode_steps: int = 720
    save_videos: bool = False
    cpu_threads: int = 1
    render_backend: str = "egl"

    def __post_init__(self) -> None:
        if self.suite != ROBOCASA_TABLETOP_SUITE:
            raise ValueError(f"unsupported RoboCasa suite: {self.suite}")
        if not self.tasks or self.episodes_per_task < 1:
            raise ValueError("RoboCasa tasks and episodes must be positive")
        if self.action_steps < 1 or self.max_episode_steps < 1 or self.cpu_threads < 1:
            raise ValueError("RoboCasa action steps, episode steps, and CPU threads must be positive")
        if self.render_backend != "egl":
            raise ValueError("RoboCasa SDK evaluation currently requires EGL rendering")
        names = [name for name, _ in self.tasks]
        if len(names) != len(set(names)):
            raise ValueError("RoboCasa task names must be unique")


class RoboCasaBenchmarkAdapter:
    def __init__(self, config: RoboCasaRuntimeConfig) -> None:
        self.config = config

    def work_items(self) -> Sequence[WorkItem]:
        return tuple(
            WorkItem(
                ordinal=index,
                item_id=f"{self.config.suite}-{task_name}",
                group_id=task_name,
                payload={
                    "task_name": task_name,
                    "env_name": env_name,
                    "episodes": self.config.episodes_per_task,
                },
                expected_results=self.config.episodes_per_task,
            )
            for index, (task_name, env_name) in enumerate(self.config.tasks)
        )

    def worker_process(self, context: WorkerContext) -> ProcessSpec:
        threads = str(self.config.cpu_threads)
        command = (
            str(self.config.python),
            "-m",
            "roboscope_kit.runtime.benchmarks.robocasa.worker",
            "--worker-id",
            str(context.worker_id),
            "--policy-socket",
            str(context.replica_socket),
            "--benchmark-root",
            str(self.config.benchmark_root),
            "--queue-dir",
            str(context.queue_dir),
            "--worker-state-dir",
            str(context.state_dir),
            "--output-dir",
            str(context.output_dir),
            "--action-steps",
            str(self.config.action_steps),
            "--max-episode-steps",
            str(self.config.max_episode_steps),
            *(("--disable-video",) if not self.config.save_videos else ()),
        )
        return ProcessSpec(
            command=command,
            cwd=self.config.benchmark_root,
            env={
                "PYTHONPATH": ":".join(
                    (
                        str(self.config.benchmark_root),
                        str(SDK_IMPORT_ROOT),
                        os.environ.get("PYTHONPATH", ""),
                    )
                ),
                "CUDA_VISIBLE_DEVICES": context.gpu_device,
                "MUJOCO_GL": self.config.render_backend,
                "PYOPENGL_PLATFORM": self.config.render_backend,
                "MUJOCO_EGL_DEVICE_ID": context.gpu_device,
                "OMP_NUM_THREADS": threads,
                "MKL_NUM_THREADS": threads,
                "OPENBLAS_NUM_THREADS": threads,
                "NUMEXPR_NUM_THREADS": threads,
                "TOKENIZERS_PARALLELISM": "false",
                "PYTHONDONTWRITEBYTECODE": "1",
            },
        )

    def aggregate(self, results: Sequence[WorkResult]) -> dict[str, Any]:
        by_task: dict[str, dict[str, Any]] = {}
        for result in results:
            task_name = str(result.metadata["task_name"])
            row = by_task.setdefault(task_name, {"task": task_name, "episodes": 0, "successes": 0})
            row["episodes"] += len(result.records)
            row["successes"] += sum(record.success for record in result.records)
        rows = []
        for task_name, _ in self.config.tasks:
            row = by_task.get(task_name, {"task": task_name, "episodes": 0, "successes": 0})
            row["success_rate"] = row["successes"] / row["episodes"] if row["episodes"] else 0.0
            rows.append(row)
        return {"tasks": rows}

    def metadata(self) -> dict[str, Any]:
        return {
            "adapter": type(self).__name__,
            "benchmark": "robocasa",
            "suite": self.config.suite,
            "protocol": "robocasa-gr1-tabletop-v1",
            "bound_hint": "gpu_infer",
            "task_count": len(self.config.tasks),
            "episodes_per_task": self.config.episodes_per_task,
            "action_steps": self.config.action_steps,
            "max_episode_steps": self.config.max_episode_steps,
            "video_logging": self.config.save_videos,
        }


__all__ = [
    "ROBOCASA_TABLETOP_SUITE",
    "ROBOCASA_TABLETOP_TASKS",
    "RoboCasaBenchmarkAdapter",
    "RoboCasaRuntimeConfig",
]

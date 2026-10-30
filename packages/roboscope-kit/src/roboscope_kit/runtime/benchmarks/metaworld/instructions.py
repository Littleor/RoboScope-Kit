# Copyright 2026 RoboScope Team
# SPDX-License-Identifier: Apache-2.0

"""Frozen MetaWorld MT50 task, language, and difficulty-mapping constants.

``INSTRUCTIONS`` values are the verbatim action-instruction sentences from the
MetaWorld v3.1.1 ``docs/benchmark/task_descriptions.md`` (original casing,
wording, punctuation, and typos preserved; the "Randomize ..." environment
notes are not part of the instruction). ``MT50_TASKS`` follows the Farama
``ALL_V3_ENVIRONMENTS`` order; the worker asserts both facts against the
installed upstream package at startup.

The four-difficulty mapping is the frozen LeRobot ``DIFFICULTY_TO_TASKS``
snapshot read on 2026-09-18. It is used only for score aggregation and display;
the language instructions stay the official Farama sentences. The module
validates at import that the four buckets are disjoint, that their union equals
the 50 MT50 slugs, and that the bucket sizes are 28/11/6/5.
"""

from __future__ import annotations

import hashlib
import json

MT50_TASKS: tuple[str, ...] = (
    "assembly-v3",
    "basketball-v3",
    "bin-picking-v3",
    "box-close-v3",
    "button-press-topdown-v3",
    "button-press-topdown-wall-v3",
    "button-press-v3",
    "button-press-wall-v3",
    "coffee-button-v3",
    "coffee-pull-v3",
    "coffee-push-v3",
    "dial-turn-v3",
    "disassemble-v3",
    "door-close-v3",
    "door-lock-v3",
    "door-open-v3",
    "door-unlock-v3",
    "hand-insert-v3",
    "drawer-close-v3",
    "drawer-open-v3",
    "faucet-open-v3",
    "faucet-close-v3",
    "hammer-v3",
    "handle-press-side-v3",
    "handle-press-v3",
    "handle-pull-side-v3",
    "handle-pull-v3",
    "lever-pull-v3",
    "pick-place-wall-v3",
    "pick-out-of-hole-v3",
    "pick-place-v3",
    "plate-slide-v3",
    "plate-slide-side-v3",
    "plate-slide-back-v3",
    "plate-slide-back-side-v3",
    "peg-insert-side-v3",
    "peg-unplug-side-v3",
    "soccer-v3",
    "stick-push-v3",
    "stick-pull-v3",
    "push-v3",
    "push-wall-v3",
    "push-back-v3",
    "reach-v3",
    "reach-wall-v3",
    "shelf-place-v3",
    "sweep-into-v3",
    "sweep-v3",
    "window-open-v3",
    "window-close-v3",
)

INSTRUCTIONS: dict[str, str] = {
    "assembly-v3": "Pick up a nut and place it onto a peg.",
    "basketball-v3": "Dunk the basketball into the basket.",
    "bin-picking-v3": "Grasp the puck from one bin and place it into another bin.",
    "box-close-v3": "Grasp the cover and close the box with it.",
    "button-press-topdown-v3": "Press a button from the top.",
    "button-press-topdown-wall-v3": "Bypass a wall and press a button from the top.",
    "button-press-v3": "Press a button.",
    "button-press-wall-v3": "Bypass a wall and press a button.",
    "coffee-button-v3": "Push a button on the coffee machine.",
    "coffee-pull-v3": "Pull a mug from a coffee machine.",
    "coffee-push-v3": "Push a mug under a coffee machine.",
    "dial-turn-v3": "Rotate a dial 180 degrees.",
    "disassemble-v3": "Pick a nut out of the a peg.",
    "door-close-v3": "Close a door with a revolving joint.",
    "door-lock-v3": "Lock the door by rotating the lock clockwise.",
    "door-open-v3": "Open a door with a revolving joint.",
    "door-unlock-v3": "Unlock the door by rotating the lock counter-clockwise.",
    "hand-insert-v3": "Insert the gripper into a hole.",
    "drawer-close-v3": "Push and close a drawer.",
    "drawer-open-v3": "Open a drawer.",
    "faucet-open-v3": "Rotate the faucet counter-clockwise.",
    "faucet-close-v3": "Rotate the faucet clockwise.",
    "hammer-v3": "Hammer a screw on the wall.",
    "handle-press-side-v3": "Press a handle down sideways.",
    "handle-press-v3": "Press a handle down.",
    "handle-pull-side-v3": "Pull a handle up sideways.",
    "handle-pull-v3": "Pull a handle up.",
    "lever-pull-v3": "Pull a lever down 90 degrees.",
    "pick-place-wall-v3": "Pick a puck, bypass a wall and place the puck.",
    "pick-out-of-hole-v3": "Pick up a puck from a hole.",
    "pick-place-v3": "Pick and place a puck to a goal.",
    "plate-slide-v3": "Slide a plate into a cabinet.",
    "plate-slide-side-v3": "Slide a plate into a cabinet sideways.",
    "plate-slide-back-v3": "Get a plate from the cabinet.",
    "plate-slide-back-side-v3": "Get a plate from the cabinet sideways.",
    "peg-insert-side-v3": "Insert a peg sideways.",
    "peg-unplug-side-v3": "Unplug a peg sideways.",
    "soccer-v3": "Kick a soccer into the goal.",
    "stick-push-v3": "Grasp a stick and push a box using the stick.",
    "stick-pull-v3": "Grasp a stick and pull a box with the stick.",
    "push-v3": "Push the puck to a goal.",
    "push-wall-v3": "Bypass a wall and push a puck to a goal.",
    "push-back-v3": "Pull a puck to a goal.",
    "reach-v3": "Reach a goal position.",
    "reach-wall-v3": "Bypass a wall and reach a goal.",
    "shelf-place-v3": "Pick and place a puck onto a shelf.",
    "sweep-into-v3": "Sweep a puck into a hole.",
    "sweep-v3": "Sweep a puck off the table.",
    "window-open-v3": "Push and open a window.",
    "window-close-v3": "Push and close a window.",
}

DIFFICULTY_MAPPING_VERSION = "lerobot-difficulty-to-tasks-2026-09-18"

DIFFICULTY_TO_TASKS: dict[str, tuple[str, ...]] = {
    "easy": (
        "button-press-v3",
        "button-press-topdown-v3",
        "button-press-topdown-wall-v3",
        "button-press-wall-v3",
        "coffee-button-v3",
        "dial-turn-v3",
        "door-close-v3",
        "door-lock-v3",
        "door-open-v3",
        "door-unlock-v3",
        "drawer-close-v3",
        "drawer-open-v3",
        "faucet-close-v3",
        "faucet-open-v3",
        "handle-press-v3",
        "handle-press-side-v3",
        "handle-pull-v3",
        "handle-pull-side-v3",
        "lever-pull-v3",
        "plate-slide-v3",
        "plate-slide-back-v3",
        "plate-slide-back-side-v3",
        "plate-slide-side-v3",
        "reach-v3",
        "reach-wall-v3",
        "window-close-v3",
        "window-open-v3",
        "peg-unplug-side-v3",
    ),
    "medium": (
        "basketball-v3",
        "bin-picking-v3",
        "box-close-v3",
        "coffee-pull-v3",
        "coffee-push-v3",
        "hammer-v3",
        "peg-insert-side-v3",
        "push-wall-v3",
        "soccer-v3",
        "sweep-v3",
        "sweep-into-v3",
    ),
    "hard": (
        "assembly-v3",
        "hand-insert-v3",
        "pick-out-of-hole-v3",
        "pick-place-v3",
        "push-v3",
        "push-back-v3",
    ),
    "very_hard": (
        "shelf-place-v3",
        "disassemble-v3",
        "stick-pull-v3",
        "stick-push-v3",
        "pick-place-wall-v3",
    ),
}

# SHA-256 of the frozen MT50(seed=42) train_tasks artifact: for each of the
# 2500 upstream Task objects in MT50.train_tasks order, feed
# ``env_name UTF-8 | NUL | task.data (pickled kwargs) | NUL`` into one SHA-256.
# Goal positions come from the simulator, so the digest is platform-specific:
# this value was generated by the pinned Linux x86_64 runtime
# (metaworld==3.1.1, numpy==2.1.3) that build-metaworld-env.sh produces, and
# every evaluator worker regenerates the artifact and refuses to run on a
# mismatch.
FROZEN_MT50_TASKS_SHA256 = "8653c0de9165302cb753912851983c090126d892d1eeef097c8fb98b816ad9aa"


def mt50_train_tasks_sha256(tasks) -> str:
    """Canonical digest over an ordered ``MT50.train_tasks`` sequence."""

    digest = hashlib.sha256()
    for task in tasks:
        digest.update(task.env_name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(task.data)
        digest.update(b"\0")
    return digest.hexdigest()


def _difficulty_mapping_artifact() -> dict:
    return {
        "version": DIFFICULTY_MAPPING_VERSION,
        "source": "https://github.com/huggingface/lerobot/blob/main/src/lerobot/envs/metaworld_config.json",
        "snapshot_date": "2026-09-18",
        "tasks": {name: list(tasks) for name, tasks in DIFFICULTY_TO_TASKS.items()},
    }


def _rfc8785_sha256(value: dict) -> str:
    """SHA-256 over the RFC 8785 canonical JSON of an ASCII-only structure."""

    canonical = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _validate() -> None:
    if len(MT50_TASKS) != 50 or len(set(MT50_TASKS)) != 50:
        raise RuntimeError("MT50_TASKS must contain 50 unique task slugs")
    if set(INSTRUCTIONS) != set(MT50_TASKS):
        raise RuntimeError("INSTRUCTIONS keys must equal the MT50 task slugs")
    expected_sizes = {"easy": 28, "medium": 11, "hard": 6, "very_hard": 5}
    if set(DIFFICULTY_TO_TASKS) != set(expected_sizes):
        raise RuntimeError("difficulty mapping must define easy/medium/hard/very_hard")
    union: list[str] = []
    for name, expected in expected_sizes.items():
        tasks = DIFFICULTY_TO_TASKS[name]
        if len(tasks) != expected or len(set(tasks)) != expected:
            raise RuntimeError(f"difficulty {name} must contain {expected} unique tasks")
        union.extend(tasks)
    if len(set(union)) != 50 or set(union) != set(MT50_TASKS):
        raise RuntimeError("difficulty buckets must be disjoint and cover exactly the 50 MT50 tasks")


_validate()

DIFFICULTY_BY_TASK: dict[str, str] = {
    task: difficulty for difficulty, tasks in DIFFICULTY_TO_TASKS.items() for task in tasks
}
DIFFICULTY_MAPPING_SHA256 = _rfc8785_sha256(_difficulty_mapping_artifact())


__all__ = [
    "DIFFICULTY_BY_TASK",
    "DIFFICULTY_MAPPING_SHA256",
    "DIFFICULTY_MAPPING_VERSION",
    "DIFFICULTY_TO_TASKS",
    "FROZEN_MT50_TASKS_SHA256",
    "INSTRUCTIONS",
    "MT50_TASKS",
    "mt50_train_tasks_sha256",
]

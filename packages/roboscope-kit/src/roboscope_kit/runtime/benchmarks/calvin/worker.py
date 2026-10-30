# Copyright 2026 RoboScope Team
# SPDX-License-Identifier: Apache-2.0

"""Run persistent official CALVIN sequences against an SDK-native policy."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from roboscope_kit import EpisodeContext, Observation, PolicyClient
from roboscope_kit.runtime.benchmark import EvaluationRecord, WorkerLoop, WorkItem, WorkResult

CALVIN_RELATIVE_ACTION = "calvin.relative-v1"
CALVIN_ABSOLUTE_ACTION = "calvin.absolute-compact-quat-v1"


def _absolute_action_tuple(action: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    """Rebuild the recipe's absolute (pos, quat_xyzw, grip) tuple from the
    compact 7-dim chunk, restoring the unit quaternion's non-negative w."""
    pos = action[:3]
    qxyz = action[3:6].astype(np.float64)
    if not np.all(np.isfinite(qxyz)):
        raise ValueError("CALVIN compact quaternion xyz must be finite")
    squared_norm = float(np.dot(qxyz, qxyz))
    if squared_norm > 1.0 + 1e-5:
        raise ValueError(f"CALVIN compact quaternion xyz squared norm {squared_norm:.8f} exceeds 1")
    if squared_norm > 1.0:
        qxyz = qxyz / np.sqrt(squared_norm)
        squared_norm = 1.0
    w = float(np.sqrt(1.0 - squared_norm))
    return (pos, np.concatenate([qxyz, [w]]), int(action[6]))


class PolicyActionQueue:
    """Adapt the SDK action-chunk protocol to CALVIN's reset/step interface."""

    def __init__(self, socket_path: Path, action_representation: str) -> None:
        self.socket_path = socket_path
        self.action_representation = action_representation
        self.sequence_index = -1
        self.eval_sequence: tuple[str, ...] = ()
        self.subtask_index = 0
        self.episode_id = ""
        self.step_count = 0
        self.actions = np.empty((0, 7), dtype=np.float32)

    def begin_sequence(self, sequence_index: int, eval_sequence: list[str]) -> None:
        self.end_episode()
        self.sequence_index = sequence_index
        self.eval_sequence = tuple(eval_sequence)
        self.subtask_index = 0
        self.episode_id = ""
        self.actions = np.empty((0, 7), dtype=np.float32)

    def reset(self) -> None:
        if self.sequence_index < 0 or self.subtask_index >= len(self.eval_sequence):
            raise RuntimeError("CALVIN policy reset occurred outside the active sequence")
        self.end_episode()
        task_id = self.eval_sequence[self.subtask_index]
        self.episode_id = f"sequence-{self.sequence_index:04d}-subtask-{self.subtask_index:02d}-{task_id}"
        with PolicyClient(self.socket_path) as client:
            client.reset(
                EpisodeContext(
                    episode_id=self.episode_id,
                    task_id=task_id,
                    metadata={
                        "sequence_index": self.sequence_index,
                        "subtask_index": self.subtask_index,
                    },
                )
            )
        self.subtask_index += 1
        self.step_count = 0
        self.actions = np.empty((0, 7), dtype=np.float32)

    def step(
        self, observation: dict, language_instruction: str
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray, int]:
        if not self.episode_id:
            raise RuntimeError("CALVIN policy step occurred before reset")
        if not self.actions.size:
            with PolicyClient(self.socket_path) as client:
                chunk = client.act(
                    build_observation(
                        self.episode_id,
                        self.step_count,
                        observation,
                        language_instruction,
                    )
                ).actions
            self.actions = chunk.reshape(1, -1) if chunk.ndim == 1 else chunk
        action = np.asarray(self.actions[0], dtype=np.float32).copy()
        self.actions = self.actions[1:]
        self.step_count += 1
        if self.action_representation == CALVIN_RELATIVE_ACTION:
            return action
        return _absolute_action_tuple(action)

    def end_episode(self) -> None:
        if not self.episode_id:
            return
        episode_id = self.episode_id
        self.episode_id = ""
        self.actions = np.empty((0, 7), dtype=np.float32)
        with PolicyClient(self.socket_path) as client:
            client.end_episode(episode_id)


def end_episode_after_failure(policy: PolicyActionQueue, primary_exc: BaseException) -> None:
    """Best-effort cleanup after a failed sequence without masking the primary error.

    end_episode clears its local episode state before contacting the policy
    server, so the action queue stays empty even when the cleanup call fails.
    A cleanup failure is chained to the original exception instead of replacing
    it, keeping the evaluate_sequence failure as the task's top-level cause.
    """
    try:
        policy.end_episode()
    except BaseException as cleanup_exc:
        raise primary_exc from cleanup_exc


def build_observation(
    episode_id: str,
    step: int,
    observation: dict,
    language_instruction: str,
) -> Observation:
    rgb = observation["rgb_obs"]
    return Observation(
        episode_id=episode_id,
        step=step,
        timestamp_s=time.time(),
        features={
            "images.static": np.ascontiguousarray(rgb["rgb_static"], dtype=np.uint8),
            "images.gripper": np.ascontiguousarray(rgb["rgb_gripper"], dtype=np.uint8),
            "state.proprioception": np.ascontiguousarray(observation["robot_obs"], dtype=np.float32),
            "language.instruction": language_instruction,
        },
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker-id", type=int, required=True)
    parser.add_argument("--policy-socket", type=Path, required=True)
    parser.add_argument("--dataset-path", type=Path, required=True)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--config-path", type=Path, required=True)
    parser.add_argument("--eval-sequences-path", type=Path, required=True)
    parser.add_argument("--eval-sequences-sha256", required=True)
    parser.add_argument(
        "--action-representation",
        choices=(CALVIN_RELATIVE_ACTION, CALVIN_ABSOLUTE_ACTION),
        required=True,
    )
    parser.add_argument("--queue-dir", type=Path, required=True)
    parser.add_argument("--worker-state-dir", type=Path, required=True)
    parser.add_argument("--poll-interval", type=float, default=0.05)
    return parser.parse_args()


def create_calvin_env(dataset_path: Path, asset_root: Path):
    import hydra
    from hydra.core.global_hydra import GlobalHydra
    from omegaconf import OmegaConf

    merged_config = dataset_path / "validation" / ".hydra" / "merged_config.yaml"
    if not merged_config.is_file():
        raise RuntimeError(f"CALVIN merged environment config is missing: {merged_config}")
    if not asset_root.is_dir():
        raise RuntimeError(f"CALVIN asset root is missing: {asset_root}")

    render_config = OmegaConf.load(merged_config)
    env_config = render_config.get("env")
    if env_config is None or env_config.get("scene_cfg") is None or env_config.get("robot_cfg") is None:
        raise RuntimeError(f"CALVIN merged environment config is invalid: {merged_config}")
    env_config.scene_cfg.data_path = str(asset_root)

    if not GlobalHydra.instance().is_initialized():
        hydra.initialize(".")
    return hydra.utils.instantiate(env_config, show_gui=False, use_vr=False, use_scene_info=True)


def main() -> None:
    args = parse_args()
    if args.poll_interval <= 0:
        raise ValueError("poll-interval must be positive")
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    # Shared AutoDL FS inode quota can be exhausted; never write pyc.
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

    import hydra
    from calvin_agent.evaluation.evaluate_policy import evaluate_sequence
    from omegaconf import OmegaConf

    task_oracle = hydra.utils.instantiate(
        OmegaConf.load(args.config_path / "callbacks/rollout/tasks/new_playtable_tasks.yaml")
    )
    val_annotations = OmegaConf.load(args.config_path / "annotations/new_playtable_validation.yaml")
    sequence_bytes = args.eval_sequences_path.read_bytes()
    if hashlib.sha256(sequence_bytes).hexdigest() != args.eval_sequences_sha256:
        raise RuntimeError("CALVIN evaluation-sequence manifest changed before worker start")
    eval_sequences = json.loads(sequence_bytes)

    env = create_calvin_env(args.dataset_path, args.asset_root)
    policy = PolicyActionQueue(args.policy_socket, args.action_representation)
    step_counter = 0
    original_step = env.step

    def counting_step(action):
        nonlocal step_counter
        step_counter += 1
        return original_step(action)

    env.step = counting_step

    def evaluate(item: WorkItem) -> WorkResult:
        nonlocal step_counter
        start_sequence = int(item.payload["start_sequence"])
        num_sequences = int(item.payload["num_sequences"])
        records: list[EvaluationRecord] = []
        started = time.perf_counter()
        for sequence_index in range(start_sequence, start_sequence + num_sequences):
            initial_state, eval_sequence = eval_sequences[sequence_index]
            policy.begin_sequence(sequence_index, eval_sequence)
            step_counter = 0
            sequence_started = time.perf_counter()
            try:
                success_length = evaluate_sequence(
                    env,
                    policy,
                    task_oracle,
                    initial_state,
                    eval_sequence,
                    val_annotations,
                    defaultdict(list),
                    False,
                )
            except BaseException as primary_exc:
                end_episode_after_failure(policy, primary_exc)
                raise
            else:
                policy.end_episode()
            records.append(
                EvaluationRecord(
                    record_id=f"sequence-{sequence_index:04d}",
                    success=int(success_length) >= 5,
                    steps=step_counter,
                    elapsed_s=time.perf_counter() - sequence_started,
                    metadata={
                        "sequence_index": sequence_index,
                        "chained_success_length": int(success_length),
                        "eval_sequence": list(eval_sequence),
                    },
                )
            )
        return WorkResult(
            item_id=item.item_id,
            group_id=item.group_id,
            worker_id=args.worker_id,
            records=tuple(records),
            metadata={"elapsed_s": time.perf_counter() - started},
        )

    WorkerLoop(
        worker_id=args.worker_id,
        queue_dir=args.queue_dir,
        state_dir=args.worker_state_dir,
        poll_interval_s=args.poll_interval,
    ).run(evaluate)


if __name__ == "__main__":
    main()

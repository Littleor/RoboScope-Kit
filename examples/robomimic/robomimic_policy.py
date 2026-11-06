# Copyright 2026 RoboScope Team
# SPDX-License-Identifier: Apache-2.0

"""robomimic model-zoo Policy example for RoboScope Kit.

Serves the five robomimic v0.1 proficient-human (PH) low-dimensional BC-RNN checkpoints
of the model zoo as one RoboScope Kit Policy for the ``robomimic`` benchmark's
``ph_low_dim`` suite. ``--checkpoint-source`` is a directory holding the five ``.pth``
files that ``download_model_zoo.py`` fetches; each is verified against its pinned
SHA-256 and loaded with robomimic's own ``FileUtils.policy_from_checkpoint``.

Everything model-specific stays here, as the upstream ``run_trained_agent.py`` does it:

- the observation dict goes through robomimic's ``RolloutPolicy`` (tensor, batch,
  float32 on the device, and observation normalization when the checkpoint has stats);
- BC-RNN keeps an LSTM hidden state that robomimic re-initializes every
  ``algo.rnn.horizon`` steps; one replica serves many concurrent episodes, so the
  hidden state and step counter are kept per episode and swapped in around each call;
- the GMM head samples its mixture component from the torch RNG. Upstream seeds that
  RNG once with ``torch.manual_seed(seed)`` before its rollout loop; here every episode
  owns a torch RNG state seeded from the episode's seed (``EpisodeContext.metadata``),
  so episode ``seed`` reproduces ``run_trained_agent.py --seed <seed> --n_rollouts 1``
  regardless of how episodes interleave on the replica.

Run it with the robomimic runtime's own Python, which already has the pinned robomimic
and torch, and put the kit source on ``PYTHONPATH`` (see ``roboscope-policy.json``).
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from model_zoo import CHECKPOINTS, ZooCheckpoint, file_sha256  # noqa: E402

from roboscope_kit.policy import (  # noqa: E402
    ActionChunk,
    EpisodeContext,
    Observation,
    Policy,
    PolicyCapabilities,
    SessionContext,
)

BENCHMARK = "robomimic"
SUITE = "ph_low_dim"
ACTION_REPRESENTATION = "robomimic.osc-pose-delta-v1"
REQUIRED_FEATURES = (
    "state.robot0_eef_pos",
    "state.robot0_eef_quat",
    "state.robot0_gripper_qpos",
    "state.object",
)


@dataclass
class TaskModel:
    checkpoint: ZooCheckpoint
    rollout: Any  # robomimic.algo.RolloutPolicy
    horizon: int


@dataclass
class EpisodeState:
    task: str
    seed: int
    rnn_hidden_state: Any
    rnn_counter: int
    cpu_rng: Any
    cuda_rng: Any


class RobomimicModelZooPolicy(Policy):
    def __init__(self) -> None:
        self.models: dict[str, TaskModel] = {}
        self.episodes: dict[str, EpisodeState] = {}
        self.device = None

    def load(self, context: SessionContext) -> PolicyCapabilities:
        if context.benchmark.name != BENCHMARK or context.benchmark.suite != SUITE:
            raise ValueError(
                f"this example serves {BENCHMARK}/{SUITE}; "
                f"got {context.benchmark.name}/{context.benchmark.suite}"
            )
        import robomimic.utils.file_utils as FileUtils  # noqa: PLC0415
        import robomimic.utils.torch_utils as TorchUtils  # noqa: PLC0415

        root = Path(context.checkpoint_path)
        if not root.is_dir():
            raise ValueError(f"--checkpoint-source must be the model-zoo directory; got {root}")
        # The upstream script's device choice.
        self.device = TorchUtils.get_torch_device(try_to_use_cuda=True)
        for checkpoint in CHECKPOINTS:
            path = root / checkpoint.filename
            if not path.is_file():
                raise FileNotFoundError(f"missing model-zoo checkpoint {path}; run download_model_zoo.py")
            actual = file_sha256(path)
            if actual != checkpoint.sha256:
                raise ValueError(f"{path} has SHA-256 {actual}, expected {checkpoint.sha256}")
            rollout, ckpt_dict = FileUtils.policy_from_checkpoint(ckpt_path=str(path), device=self.device)
            env_name = ckpt_dict["env_metadata"]["env_name"]
            if env_name != checkpoint.env_name:
                raise ValueError(f"{path} was trained on {env_name}, expected {checkpoint.env_name}")
            config, _ = FileUtils.config_from_checkpoint(ckpt_dict=ckpt_dict)
            rollout.start_episode()  # set_eval() once; per-episode state is swapped in by act()
            self.models[checkpoint.task] = TaskModel(
                checkpoint, rollout, int(config.experiment.rollout.horizon)
            )
        return PolicyCapabilities(
            supported_benchmarks=(BENCHMARK,),
            supported_suites={BENCHMARK: (SUITE,)},
            required_features=REQUIRED_FEATURES,
            max_batch_size=1,
            supported_action_chunk_sizes=(1,),
            action_representations={BENCHMARK: ACTION_REPRESENTATION},
        )

    def reset(self, context: EpisodeContext) -> None:
        import torch  # noqa: PLC0415

        model = self.models.get(context.task_id)
        if model is None:
            raise ValueError(f"no model-zoo checkpoint for robomimic task {context.task_id!r}")
        metadata = context.metadata
        if metadata.get("env_name") != model.checkpoint.env_name:
            raise ValueError(
                f"episode environment {metadata.get('env_name')!r} does not match the "
                f"{model.checkpoint.filename} environment {model.checkpoint.env_name!r}"
            )
        if int(metadata.get("horizon", -1)) != model.horizon:
            raise ValueError(
                f"episode horizon {metadata.get('horizon')} differs from the checkpoint's rollout horizon "
                f"{model.horizon}"
            )
        seed = int(metadata["seed"])
        with torch.random.fork_rng(devices=self._rng_devices()):
            torch.manual_seed(seed)
            cpu_rng, cuda_rng = self._capture_rng()
        self.episodes[context.episode_id] = EpisodeState(
            task=context.task_id,
            seed=seed,
            rnn_hidden_state=None,
            rnn_counter=0,
            cpu_rng=cpu_rng,
            cuda_rng=cuda_rng,
        )

    def act(self, observation: Observation) -> ActionChunk:
        import torch  # noqa: PLC0415

        state = self.episodes.get(observation.episode_id)
        if state is None:
            raise RuntimeError(f"robomimic action requested for unknown episode {observation.episode_id}")
        rollout = self.models[state.task].rollout
        algo = rollout.policy
        obs = {
            name.removeprefix("state."): np.asarray(value)
            for name, value in observation.features.items()
            if name.startswith("state.")
        }
        algo._rnn_hidden_state = state.rnn_hidden_state
        algo._rnn_counter = state.rnn_counter
        with torch.random.fork_rng(devices=self._rng_devices()):
            torch.set_rng_state(state.cpu_rng)
            if state.cuda_rng is not None:
                torch.cuda.set_rng_state(state.cuda_rng, self.device)
            action = rollout(ob=obs)
            state.cpu_rng, state.cuda_rng = self._capture_rng()
        state.rnn_hidden_state = algo._rnn_hidden_state
        state.rnn_counter = algo._rnn_counter
        algo._rnn_hidden_state = None
        return ActionChunk(actions=np.asarray(action, dtype=np.float32))

    def end_episode(self, episode_id: str) -> None:
        self.episodes.pop(episode_id, None)

    def unload(self) -> None:
        self.episodes.clear()
        self.models.clear()

    def _rng_devices(self) -> list[int]:
        return [self.device.index or 0] if self.device.type == "cuda" else []

    def _capture_rng(self) -> tuple[Any, Any]:
        import torch  # noqa: PLC0415

        cuda_rng = torch.cuda.get_rng_state(self.device) if self.device.type == "cuda" else None
        return torch.get_rng_state(), cuda_rng


def main() -> None:
    from roboscope_kit.server import run  # noqa: PLC0415

    run(RobomimicModelZooPolicy())


if __name__ == "__main__":
    main()

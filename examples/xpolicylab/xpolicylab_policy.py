"""XPolicyLab Policy example for RoboScope Kit on RoboDojo.

Serves any XPolicyLab policy (``XPolicyLab/policy/<name>/model.py``) as a
RoboScope Kit Policy. The process runs from the RoboDojo checkout built by
``benchmarks/robodojo/build-runtime.sh``, which contains the pinned XPolicyLab,
with the Python environment of the chosen policy::

    xpolicylab_policy.py --policy Simate_beta [--action-type joint|ee]

The policy's ``deploy.yml`` is loaded with the same overrides XPolicyLab's
server receives for RoboDojo; ``--checkpoint-source`` is passed as
``checkpoint_path``.

The episode loop is XPolicyLab's reference ``deploy.py``: ``update_obs`` on
every control step, ``get_action`` for an action chunk, and the observations
between the chunk's actions fed back through ``update_obs``. Here the first
observation of a chunk arrives through ``act()`` and the others through
``observe()`` (``step_observation_updates``).

XPolicyLab models keep one episode of state unless they scope it by evaluation
ID (``reset_evaluation``, as ``Simate_beta`` does). An unscoped model serves one
episode at a time, so run it with as many replicas as Workers
(``--concurrency K --replicas K``).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from roboscope_kit.policy import (
    ActionChunk,
    EpisodeContext,
    Observation,
    Policy,
    PolicyCapabilities,
    SessionContext,
)

BENCHMARK = "robodojo"
SUITES = ("arx_x5", "arx_x5_randomized")
ENV_CFG_TYPE = "arx_x5"
# RoboDojo seed passed to XPolicyLab's server as the `seed` override.
SEED = 1
MAX_CHUNK = 256
ACTION_REPRESENTATIONS = {
    "joint": "robodojo.x5-joint-v1",
    "ee": "robodojo.x5-ee-compact-quat-v1",
}
REQUIRED_FEATURES = (
    "images.head",
    "images.left_wrist",
    "images.right_wrist",
    "state.joints",
    "state.ee_pose",
    "language.instruction",
)


def _native_observation(observation: Observation, scope: dict) -> dict:
    """Rebuild the RoboDojo ``get_obs()`` fields an XPolicyLab model reads."""

    features = observation.features
    joints = np.asarray(features["state.joints"], dtype=np.float32)
    pose = np.asarray(features["state.ee_pose"], dtype=np.float32)
    return {
        "vision": {
            "cam_head": {"color": features["images.head"]},
            "cam_left_wrist": {"color": features["images.left_wrist"]},
            "cam_right_wrist": {"color": features["images.right_wrist"]},
        },
        "state": {
            "left_arm_joint_state": joints[0:6],
            "left_ee_joint_state": joints[6:7],
            "right_arm_joint_state": joints[7:13],
            "right_ee_joint_state": joints[13:14],
            "left_ee_pose": pose[0:7],
            "right_ee_pose": pose[7:14],
        },
        "instruction": features["language.instruction"],
        **scope,
    }


def _compact_pose(pose) -> np.ndarray:
    """``[x, y, z, qw, qx, qy, qz]`` to the kit's ``[x, y, z, qx, qy, qz]`` with ``qw >= 0``."""

    pose = np.asarray(pose, dtype=np.float64).reshape(7)
    quaternion = pose[3:] / np.linalg.norm(pose[3:])
    if quaternion[0] < 0:
        quaternion = -quaternion
    return np.concatenate((pose[:3], quaternion[1:]))


def _action_row(action: dict, action_type: str) -> np.ndarray:
    parts = []
    for side in ("left", "right"):
        if action_type == "joint":
            parts.append(np.asarray(action[f"{side}_arm_joint_state"], dtype=np.float64).reshape(6))
        else:
            parts.append(_compact_pose(action[f"{side}_ee_pose"]))
        parts.append(np.asarray(action[f"{side}_ee_joint_state"], dtype=np.float64).reshape(1))
    return np.concatenate(parts)


class XPolicyLabPolicy(Policy):
    def __init__(self, policy_name: str, action_type: str | None) -> None:
        self.policy_name = policy_name
        self.action_type = action_type
        self.model = None
        self.scoped = False
        self.active_episode = ""

    def load(self, context: SessionContext) -> PolicyCapabilities:
        if context.benchmark.name != BENCHMARK:
            raise ValueError(f"this example supports only {BENCHMARK!r}; got {context.benchmark.name!r}")
        root = Path.cwd()
        policy_dir = root / "XPolicyLab/policy" / self.policy_name
        if not (policy_dir / "model.py").is_file():
            raise RuntimeError(f"Run from the RoboDojo checkout; XPolicyLab policy not found: {policy_dir}")
        # The same import roots XPolicyLab's setup_eval_policy_server.sh puts on PYTHONPATH.
        for path in (root / "XPolicyLab", root):
            if str(path) not in sys.path:
                sys.path.insert(0, str(path))

        import importlib  # noqa: PLC0415

        import yaml  # noqa: PLC0415

        config = yaml.safe_load((policy_dir / "deploy.yml").read_text()) or {}
        action_type = self.action_type or config.get("action_type") or "joint"
        if action_type not in ACTION_REPRESENTATIONS:
            raise ValueError(f"unsupported XPolicyLab action type for RoboDojo X5: {action_type!r}")
        config.update(
            policy_name=self.policy_name,
            bench_name="RoboDojo",
            env_cfg_type=ENV_CFG_TYPE,
            seed=SEED,
            action_type=action_type,
            checkpoint_path=context.checkpoint_path,
        )
        model_class = importlib.import_module(f"XPolicyLab.policy.{self.policy_name}.model").Model
        self.model = model_class(config)
        self.action_type = action_type
        self.scoped = callable(getattr(self.model, "reset_evaluation", None))
        return PolicyCapabilities(
            supported_benchmarks=(BENCHMARK,),
            supported_suites={BENCHMARK: SUITES},
            required_features=REQUIRED_FEATURES,
            max_batch_size=1,
            # XPolicyLab policies choose their own chunk length; the kit checks each reply against this set.
            supported_action_chunk_sizes=(int(config["exec_chunk_size"]),) if config.get("exec_chunk_size")
            else tuple(range(1, MAX_CHUNK + 1)),
            action_representations={BENCHMARK: ACTION_REPRESENTATIONS[action_type]},
            step_observation_updates=(BENCHMARK,),
        )

    def _scope(self, episode_id: str) -> dict:
        return {"evaluation_id": episode_id, "env_idx": 0} if self.scoped else {}

    def reset(self, context: EpisodeContext) -> None:
        if self.scoped:
            self.model.reset_evaluation({"evaluation_id": context.episode_id})
            return
        if self.active_episode:
            raise RuntimeError(
                f"XPolicyLab policy {self.policy_name} keeps one episode of state; "
                "run it with as many replicas as Workers (--concurrency K --replicas K)"
            )
        self.model.reset()
        self.active_episode = context.episode_id

    def observe(self, observation: Observation) -> None:
        self.model.update_obs(_native_observation(observation, self._scope(observation.episode_id)))

    def act(self, observation: Observation) -> ActionChunk:
        scope = self._scope(observation.episode_id)
        self.model.update_obs(_native_observation(observation, scope))
        actions = self.model.get_action({"evaluation_id": observation.episode_id}) if self.scoped \
            else self.model.get_action()
        if not len(actions):
            raise RuntimeError(f"XPolicyLab policy {self.policy_name} returned an empty action chunk")
        rows = [_action_row(action, self.action_type) for action in actions]
        return ActionChunk(actions=np.stack(rows).astype(np.float32))

    def end_episode(self, episode_id: str) -> None:
        if self.scoped:
            self.model.reset_evaluation({"evaluation_id": episode_id})
        elif episode_id == self.active_episode:
            self.model.reset()
            self.active_episode = ""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--policy", required=True, help="XPolicyLab policy directory name, e.g. Simate_beta")
    parser.add_argument("--action-type", choices=tuple(ACTION_REPRESENTATIONS),
                        help="defaults to the policy's deploy.yml action_type, else joint")
    args = parser.parse_args()

    from roboscope_kit.server import run  # noqa: PLC0415

    run(XPolicyLabPolicy(args.policy, args.action_type))


if __name__ == "__main__":
    main()

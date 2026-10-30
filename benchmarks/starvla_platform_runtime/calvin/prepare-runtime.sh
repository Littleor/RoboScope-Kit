#!/usr/bin/env bash
set -Eeuo pipefail

LOCAL_ROOT="${1:?local runtime root is required}"
ARCHIVE=/root/autodl-fs/benchmarks/calvin/artifacts/calvin-env.tar
PYTHON="${LOCAL_ROOT}/calvin/bin/python"
if [[ ! -f "${ARCHIVE}" ]]; then
  echo "CALVIN environment archive is missing: ${ARCHIVE}" >&2
  exit 11
fi
if [[ ! -x "${PYTHON}" ]]; then
  tar -xf "${ARCHIVE}" -C "${LOCAL_ROOT}"
fi
if [[ ! -x "${PYTHON}" ]]; then
  echo "CALVIN archive did not produce ${PYTHON}" >&2
  exit 12
fi

# CALVIN's recorded validation environment uses the long-finger Panda. The
# minimal runtime rebuilt from calvin_env's default panda.yaml has different
# gripper joints and TCP geometry, so loading the checkpoint still succeeds but
# manipulation episodes run with the wrong robot. Apply the upstream robot
# override to the shared evaluation config before a worker instantiates it.
SHARED_ROOT="${ARCHIVE%/artifacts/calvin-env.tar}"
env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 "${PYTHON}" -I - \
  "${SHARED_ROOT}/benchmark/calvin/calvin_env/conf/robot/panda_longer_finger.yaml" \
  "${SHARED_ROOT}/smoke-assets/suite/minimal_task_D_D/validation/.hydra/merged_config.yaml" <<'PY'
from pathlib import Path
import sys

from omegaconf import OmegaConf

robot_override = OmegaConf.load(sys.argv[1])
del robot_override.defaults
path = Path(sys.argv[2])
config = OmegaConf.load(path)
robot = OmegaConf.merge(config.env.robot_cfg, robot_override)
if robot != config.env.robot_cfg:
    config.env.robot_cfg = robot
    temporary = path.with_name(path.name + ".tmp")
    OmegaConf.save(config, temporary)
    temporary.replace(path)
print(f"CALVIN validation robot: {robot.filename}; joints={list(robot.gripper_joint_ids)}; tcp={robot.tcp_link_id}")
PY

#!/usr/bin/env bash
set -Eeuo pipefail

# Build the CALVIN benchmark environment under <install-dir>:
#
#   <install-dir>/
#     benchmark-manifest.json       this directory's manifest; runtime paths are relative to it
#     env/                          conda prefix, CPython 3.8.20 with runtime-requirements.txt
#     calvin/                       mees/calvin with calvin_env and tacto at pinned commits
#     dataset/minimal_task_D_D/     the evaluation dataset inputs, assembled from the sources
#     eval_sequences.json           the official 1,000 challenge sequences, pinned by SHA-256
#
# Usage: build-runtime.sh <install-dir>
# Then:  python -m roboscope_kit.runtime.benchmark --benchmark calvin \
#          --benchmark-manifest <install-dir>/benchmark-manifest.json ...
#
# Sources. Every file matches upstream byte for byte:
#   calvin        mees/calvin at fa03f01, the main tip
#   calvin_env    mees/calvin_env at 797142c (CALVIN's own submodule pointer is older)
#   tacto         the calvin_env submodule
# runtime-requirements.txt is the complete, SHA-256-verified dependency set of
# the 2026-09-30 build.
#
# No CALVIN dataset download is needed. get_env_state_for_initial_condition
# hard-codes the initial states and reads no episode data, so the only dataset
# inputs are three configuration files: two ship in calvin_models/conf and this
# script composes validation/.hydra/merged_config.yaml from calvin_env's conf.
#
# Validation target. This environment reproduces the CALVIN numbers StarVLA
# reports in examples/calvin/README.md; the kit emits the same quantities,
# avg_seq_len plus task_1..task_5:
#
#   qwenpi  qwen2.5-vl-3B-instruct-action  avg 3.576  90.9 / 79.5 / 69.6 / 62.2 / 55.4
#   qwenpi  qwen3-vl-4B-instruct           avg 3.472  87.7 / 75.2 / 67.4 / 61.8 / 55.1
#   qwengr00t qwen2.5-vl-3B-instruct       avg 3.697  91.7 / 81.9 / 72.7 / 65.3 / 58.1
#   qwengr00t qwen2.5-vl-3B-instruct-action avg 3.786 92.5 / 83.9 / 74.4 / 67.9 / 59.9
#   qwengr00t qwen3-vl-4B-instruct         avg 3.645  89.8 / 79.9 / 71.8 / 64.9 / 58.1
#   qwengr00t qwen3-vl-4B-instruct-action  avg 3.757  91.1 / 81.8 / 74.1 / 67.6 / 61.1
#
# Score parity also requires the same checkpoint, Policy action schedule and the
# full 1000-sequence evaluation; the environment alone does not establish it.
#
# Re-running the script on an existing <install-dir> reuses the environment and
# checkouts, re-pins them, and regenerates the dataset inputs.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
[[ $# -eq 1 ]] || {
  echo "usage: $0 <install-dir>" >&2
  exit 2
}
install -d -m 0755 "$1"
INSTALL_DIR="$(cd -- "$1" && pwd -P)"

MANIFEST="${SCRIPT_DIR}/benchmark-manifest.json"
REQUIREMENTS="${SCRIPT_DIR}/runtime-requirements.txt"
ENV_PREFIX="${INSTALL_DIR}/env"
ENV_PYTHON="${ENV_PREFIX}/bin/python"
SOURCE_ROOT="${INSTALL_DIR}/calvin"
DATASET_ROOT="${INSTALL_DIR}/dataset/minimal_task_D_D"
EVAL_SEQUENCES="${INSTALL_DIR}/eval_sequences.json"

CALVIN_URL="${ROBOSCOPE_CALVIN_URL:-https://github.com/mees/calvin}"
CALVIN_COMMIT="${ROBOSCOPE_CALVIN_COMMIT:-fa03f01f19c65920e18cf37398a9ce859274af76}"
CALVIN_ENV_URL="${ROBOSCOPE_CALVIN_ENV_URL:-https://github.com/mees/calvin_env}"
CALVIN_ENV_COMMIT="${ROBOSCOPE_CALVIN_ENV_COMMIT:-797142c588c21e76717268b7b430958dbd13bf48}"
PYTHON_VERSION="3.8.20"
# The starVLA revision whose eval_sequences.json matches the SHA-256 the kit's
# CALVIN adapter asserts.
EVAL_SEQUENCES_URL="${ROBOSCOPE_CALVIN_EVAL_SEQUENCES_URL:-https://raw.githubusercontent.com/starVLA/starVLA/074ce6546f47caf401805ccf39e5776507a94790/examples/calvin/eval_files/eval_sequences.json}"
EVAL_SEQUENCES_SHA256="c3689911565037d399d779c655e9e943cbd5bfbfbf44045e2b3b1daa8b1a2317"

# Ignore any pip configuration on the build host. GPU hosts commonly ship an
# /etc/pip.conf that adds an unreachable NVIDIA index, which pip then retries for
# every package, and a host-level index must not decide what the environment
# resolves to.
export PIP_CONFIG_FILE=/dev/null
# CALVIN's dependencies still fork parallel compilers; one job at a time avoids
# the OOM killer reporting a bare "Error compiling objects".
MAX_JOBS="${ROBOSCOPE_BUILD_MAX_JOBS:-1}"
export MAX_JOBS

if [[ "$(uname -s):$(uname -m)" != "Linux:x86_64" ]]; then
  echo "CALVIN runtime builds require Linux x86_64" >&2
  exit 3
fi
[[ -f "${MANIFEST}" && -f "${REQUIREMENTS}" ]] || {
  echo "CALVIN build input is missing next to $0" >&2
  exit 3
}
# micromamba is a drop-in conda replacement, so accept either.
CONDA_BIN=""
for candidate in conda micromamba; do
  if command -v "${candidate}" >/dev/null 2>&1; then
    CONDA_BIN="$(command -v "${candidate}")"
    break
  fi
done
[[ -n "${CONDA_BIN}" ]] || {
  echo "conda or micromamba is required to build the CALVIN runtime" >&2
  exit 3
}

if [[ ! -x "${ENV_PYTHON}" ]]; then
  "${CONDA_BIN}" create --yes --quiet -c conda-forge \
    --prefix "${ENV_PREFIX}" "python=${PYTHON_VERSION}"
fi
ACTUAL_PYTHON_VERSION="$("${ENV_PYTHON}" -I -c 'import sys; print("%d.%d.%d" % sys.version_info[:3])')"
[[ "${ACTUAL_PYTHON_VERSION}" == "${PYTHON_VERSION}" ]] || {
  echo "CALVIN environment is Python ${ACTUAL_PYTHON_VERSION}, expected ${PYTHON_VERSION}: ${ENV_PREFIX}" >&2
  exit 5
}

if [[ ! -d "${SOURCE_ROOT}/.git" ]]; then
  git clone --quiet --no-checkout "${CALVIN_URL}" "${SOURCE_ROOT}"
fi
git -C "${SOURCE_ROOT}" checkout --quiet --detach "${CALVIN_COMMIT}"
ACTUAL_CALVIN_COMMIT="$(git -C "${SOURCE_ROOT}" rev-parse HEAD)"
[[ "${ACTUAL_CALVIN_COMMIT:0:${#CALVIN_COMMIT}}" == "${CALVIN_COMMIT}" ]] || {
  echo "CALVIN is at ${ACTUAL_CALVIN_COMMIT}, expected ${CALVIN_COMMIT}" >&2
  exit 5
}
git -C "${SOURCE_ROOT}" submodule update --init --recursive
# CALVIN's own submodule pointer does not name the calvin_env revision this
# environment is pinned to, so override the pointer rather than trusting it.
git -C "${SOURCE_ROOT}/calvin_env" remote set-url origin "${CALVIN_ENV_URL}"
git -C "${SOURCE_ROOT}/calvin_env" fetch --quiet origin
git -C "${SOURCE_ROOT}/calvin_env" checkout --quiet --detach "${CALVIN_ENV_COMMIT}"
ACTUAL_CALVIN_ENV_COMMIT="$(git -C "${SOURCE_ROOT}/calvin_env" rev-parse HEAD)"
[[ "${ACTUAL_CALVIN_ENV_COMMIT:0:${#CALVIN_ENV_COMMIT}}" == "${CALVIN_ENV_COMMIT}" ]] || {
  echo "calvin_env is at ${ACTUAL_CALVIN_ENV_COMMIT}, expected ${CALVIN_ENV_COMMIT}" >&2
  exit 5
}
git -C "${SOURCE_ROOT}/calvin_env" submodule update --init --recursive
# tacto is pinned by the calvin_env submodule pointer.
EXPECTED_TACTO_COMMIT="$(git -C "${SOURCE_ROOT}/calvin_env" ls-tree HEAD tacto | awk '{print $3}')"
ACTUAL_TACTO_COMMIT="$(git -C "${SOURCE_ROOT}/calvin_env/tacto" rev-parse HEAD)"
[[ "${ACTUAL_TACTO_COMMIT}" == "${EXPECTED_TACTO_COMMIT}" ]] || {
  echo "tacto is at ${ACTUAL_TACTO_COMMIT}, expected ${EXPECTED_TACTO_COMMIT}" >&2
  exit 5
}

# Pin the build toolchain to what CALVIN itself pins. calvin_models/requirements.txt
# asks for setuptools 57.5.0, and pyhash's setup.py still uses use_2to3, which
# setuptools removed in 58, so a modern setuptools cannot build it at all.
PYTHONNOUSERSITE=1 "${ENV_PYTHON}" -m pip install --no-cache-dir \
  "setuptools==57.5.0" "wheel==0.45.1" "cmake==4.4.3"

# CALVIN's own install.sh installs tacto, then calvin_env, then calvin_models.
PYTHONNOUSERSITE=1 "${ENV_PYTHON}" -m pip install --no-cache-dir --no-deps \
  --no-build-isolation -e "${SOURCE_ROOT}/calvin_env/tacto"
PYTHONNOUSERSITE=1 "${ENV_PYTHON}" -m pip install --no-cache-dir --no-deps \
  --no-build-isolation -e "${SOURCE_ROOT}/calvin_env"
PYTHONNOUSERSITE=1 "${ENV_PYTHON}" -m pip install --no-cache-dir --no-deps \
  --no-build-isolation -e "${SOURCE_ROOT}/calvin_models"

# The dependency set is complete. Install without resolving substitutes
# (opencv's current metadata would otherwise upgrade the pinned numpy), and use
# the CPU torch wheel because this simulator drives a separate Policy process.
PYTHONNOUSERSITE=1 "${ENV_PYTHON}" -m pip install --no-cache-dir \
  --no-deps --no-build-isolation \
  --extra-index-url https://download.pytorch.org/whl/cpu \
  --requirement "${REQUIREMENTS}"

# Assemble the minimal evaluation dataset. CALVIN's evaluate_sequence needs the
# task oracle, the language annotations and a merged environment config; the two
# YAML files ship in calvin_models/conf and the merged config is composed here
# from calvin_env's own conf tree so the scene, robot and cameras stay the
# upstream definitions rather than a hand-written copy.
CALVIN_ENV_CONF="${SOURCE_ROOT}/calvin_env/conf"
CALVIN_MODELS_CONF="${SOURCE_ROOT}/calvin_models/conf"
install -d -m 0755 \
  "${DATASET_ROOT}/validation/.hydra" \
  "${DATASET_ROOT}/annotations" \
  "${DATASET_ROOT}/callbacks/rollout/tasks"
install -m 0644 "${CALVIN_MODELS_CONF}/annotations/new_playtable_validation.yaml" \
  "${DATASET_ROOT}/annotations/new_playtable_validation.yaml"
install -m 0644 "${CALVIN_MODELS_CONF}/callbacks/rollout/tasks/new_playtable_tasks.yaml" \
  "${DATASET_ROOT}/callbacks/rollout/tasks/new_playtable_tasks.yaml"

CALVIN_ENV_CONF="${CALVIN_ENV_CONF}" MERGED_CONFIG="${DATASET_ROOT}/validation/.hydra/merged_config.yaml" \
  PYTHONNOUSERSITE=1 "${ENV_PYTHON}" -I - <<'PY'
import os
from pathlib import Path

from omegaconf import OmegaConf

conf_root = Path(os.environ["CALVIN_ENV_CONF"])
output = Path(os.environ["MERGED_CONFIG"])

# calvin_env ships the environment as hydra groups that interpolate into each
# other: play_table_env.yaml refers to ${cameras}, ${robot} and ${scene}, while
# panda.yaml and calvin_scene_D_eval.yaml refer back at each other. Put every
# group under one root and let OmegaConf resolve the whole thing, then keep only
# the env subtree, which is what hydra's merged config contains.
cameras = {
    "static": OmegaConf.load(conf_root / "cameras" / "cameras" / "static.yaml"),
    "gripper": OmegaConf.load(conf_root / "cameras" / "cameras" / "gripper.yaml"),
}
# CALVIN's recorded validation environment uses the long-finger Panda. The
# default panda.yaml has different gripper joints and TCP geometry: checkpoints
# still load, but episodes would run with the wrong robot.
robot = OmegaConf.merge(
    OmegaConf.load(conf_root / "robot" / "panda.yaml"),
    OmegaConf.load(conf_root / "robot" / "panda_longer_finger.yaml"),
)
del robot.defaults
root = OmegaConf.create(
    {
        "cameras": cameras,
        "robot": robot,
        "scene": OmegaConf.load(conf_root / "scene" / "calvin_scene_D_eval.yaml"),
        "env": OmegaConf.load(conf_root / "env" / "play_table_env.yaml"),
        # calvin_scene_D_eval.yaml interpolates ${data_path}; the kit replaces it
        # with the asset root at runtime, so seed a placeholder here.
        "data_path": "/calvin/assets",
        # play_table_env.yaml interpolates ${use_vr}; the kit passes use_vr=False
        # when it instantiates, but the config has to resolve first.
        "use_vr": False,
    }
)
resolved = OmegaConf.to_container(root, resolve=True)
merged = OmegaConf.create({"env": resolved["env"]})
# The kit overwrites scene_cfg.data_path itself; make the placeholder explicit so
# the generated config is valid on its own.
merged.env.scene_cfg.data_path = "/calvin/assets"
OmegaConf.save(merged, output)
print(f"wrote {output}")
PY

sequences_sha256() { sha256sum "$1" | awk '{print $1}'; }
if [[ ! -f "${EVAL_SEQUENCES}" || "$(sequences_sha256 "${EVAL_SEQUENCES}")" != "${EVAL_SEQUENCES_SHA256}" ]]; then
  curl --fail --location --silent --show-error \
    --output "${EVAL_SEQUENCES}.tmp" "${EVAL_SEQUENCES_URL}"
  mv -- "${EVAL_SEQUENCES}.tmp" "${EVAL_SEQUENCES}"
fi
ACTUAL_SEQUENCES_SHA256="$(sequences_sha256 "${EVAL_SEQUENCES}")"
[[ "${ACTUAL_SEQUENCES_SHA256}" == "${EVAL_SEQUENCES_SHA256}" ]] || {
  echo "CALVIN evaluation sequences are ${ACTUAL_SEQUENCES_SHA256}, expected ${EVAL_SEQUENCES_SHA256}" >&2
  exit 5
}
SEQUENCE_COUNT="$(
  PYTHONNOUSERSITE=1 "${ENV_PYTHON}" -I -c \
    'import json,sys; print(len(json.load(open(sys.argv[1]))))' \
    "${EVAL_SEQUENCES}"
)"
[[ "${SEQUENCE_COUNT}" == "1000" ]] || {
  echo "CALVIN evaluation sequence count is ${SEQUENCE_COUNT}, expected 1000" >&2
  exit 5
}

SOURCE_ROOT="${SOURCE_ROOT}" DATASET_ROOT="${DATASET_ROOT}" \
  env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 "${ENV_PYTHON}" -I - <<'PY'
import os
from pathlib import Path

from omegaconf import OmegaConf

source_root = Path(os.environ["SOURCE_ROOT"])
dataset_root = Path(os.environ["DATASET_ROOT"])

import hydra
import numpy
import pybullet
import pyhash
import torch
from pytorch_lightning import seed_everything

from calvin_agent.evaluation.evaluate_policy import evaluate_sequence
from calvin_env.envs.play_table_env import PlayTableSimEnv

from importlib.metadata import version

for distribution in ("torch", "pytorch-lightning", "hydra-core", "omegaconf", "numpy"):
    print(f"  {distribution}=={version(distribution)}")

merged = OmegaConf.load(dataset_root / "validation" / ".hydra" / "merged_config.yaml")
env_config = merged["env"]
for key in ("scene_cfg", "robot_cfg", "cameras"):
    if key not in env_config:
        raise SystemExit(f"merged config is missing env.{key}")
# The camera group keys are the camera names; CALVIN turns them into rgb_<name>
# observations, which is where rgb_static and rgb_gripper come from.
for key in ("static", "gripper"):
    if key not in env_config["cameras"]:
        raise SystemExit(f"merged config cameras are missing {key}")

for relative in ("calvin_env", "calvin_models", "calvin_env/tacto", "calvin_env/data"):
    if not (source_root / relative).is_dir():
        raise SystemExit(f"CALVIN tree is missing {relative}")

# Instantiate the environment the way the kit's worker does. This is the only
# check that proves the merged config, the asset tree and the dependency set work
# together: imports alone pass even when the scene cannot be built.
from hydra.core.global_hydra import GlobalHydra

env_config.scene_cfg.data_path = str(source_root / "calvin_env" / "data")
if not GlobalHydra.instance().is_initialized():
    hydra.initialize(".")
environment = hydra.utils.instantiate(
    env_config, show_gui=False, use_vr=False, use_scene_info=True
)
observation = environment.reset()
for key in ("rgb_static", "rgb_gripper"):
    shape = observation["rgb_obs"][key].shape
    if len(shape) != 3 or shape[2] != 3:
        raise SystemExit(f"CALVIN camera {key} returned shape {shape}")
if len(observation["robot_obs"]) != 15:
    raise SystemExit(
        f"CALVIN robot_obs has {len(observation['robot_obs'])} values, expected 15"
    )
action = numpy.zeros(7, dtype=numpy.float32)
action[6] = 1.0
environment.step(action)
print("CALVIN runtime imports, merged config, reset and step passed")
print(f"  cameras: {sorted(observation['rgb_obs'])}")
PY

install -m 0644 "${MANIFEST}" "${INSTALL_DIR}/benchmark-manifest.json"
echo "Built CALVIN ${CALVIN_COMMIT} with calvin_env ${CALVIN_ENV_COMMIT} in ${INSTALL_DIR}"

# The kit runs its own path check against the installed manifest. It needs a
# Python with roboscope-kit installed; the CALVIN environment does not need it.
if [[ -n "${KIT_PYTHON:-}" ]] || python3 -c 'import roboscope_kit' >/dev/null 2>&1; then
  KIT_PYTHON="${KIT_PYTHON:-python3}"
  "${KIT_PYTHON}" -c 'import roboscope_kit' || {
    echo "KIT_PYTHON=${KIT_PYTHON} cannot import roboscope_kit; install roboscope-kit into it" >&2
    exit 6
  }
  "${KIT_PYTHON}" -m roboscope_kit.runtime.benchmark --benchmark calvin --self-test \
    --benchmark-manifest "${INSTALL_DIR}/benchmark-manifest.json" \
    --output-dir "${INSTALL_DIR}/self-test"
else
  echo "Skipped the kit self-test: python3 cannot import roboscope_kit; set KIT_PYTHON to run it" >&2
fi

#!/usr/bin/env bash
set -Eeuo pipefail

# One-shot builder for the CALVIN benchmark runtime.
#
# The lost CALVIN environment was not recoverable from the 2026-09-02
# inventory: it existed on the shared disk only as artifacts/calvin-env.tar, so
# there was no record of its package versions. The restored 2026-09-30 build
# provides the committed dependency lock used here; it defines a new reproducible
# runtime, rather than claiming to recover the lost environment's bytes.
#
# What the inventory does pin exactly is the source tree, and every part of it
# matches upstream byte for byte:
#   calvin        mees/calvin at fa03f01, the main tip, 158 files identical
#   calvin_env    mees/calvin_env at 797142c, 282 of 287 files identical
#   tacto         the calvin_env submodule
# The five differing calvin_env files are unrecorded local edits and are not
# reconstructed here.
#
# The evaluation dataset is not needed either. CALVIN's
# get_env_state_for_initial_condition hard-codes the initial states and reads no
# episode data, so the only dataset inputs are three configuration files, two of
# which ship inside calvin_models/conf. This builder generates the third,
# validation/.hydra/merged_config.yaml, from calvin_env's own conf tree, and
# assembles the minimal dataset layout the manifest names.
#
# The evaluation sequence manifest is fetched from the starVLA revision whose
# SHA-256 the SDK adapter asserts, so the protocol file is pinned by checksum.
#
# Validation target. This runtime is meant to reproduce the CALVIN numbers the
# StarVLA community reports, not the lost environment's bytes. StarVLA publishes
# them in examples/calvin/README.md and the SDK adapter emits exactly the same
# quantities, avg_seq_len plus task_1..task_5, so a run is comparable directly:
#
#   qwenpi  qwen2.5-vl-3B-instruct-action  avg 3.576  90.9 / 79.5 / 69.6 / 62.2 / 55.4
#   qwenpi  qwen3-vl-4B-instruct           avg 3.472  87.7 / 75.2 / 67.4 / 61.8 / 55.1
#   qwengr00t qwen2.5-vl-3B-instruct       avg 3.697  91.7 / 81.9 / 72.7 / 65.3 / 58.1
#   qwengr00t qwen2.5-vl-3B-instruct-action avg 3.786 92.5 / 83.9 / 74.4 / 67.9 / 59.9
#   qwengr00t qwen3-vl-4B-instruct         avg 3.645  89.8 / 79.9 / 71.8 / 64.9 / 58.1
#   qwengr00t qwen3-vl-4B-instruct-action  avg 3.757  91.1 / 81.8 / 74.1 / 67.6 / 61.1
#
# Comparing model scores also requires matching the checkpoint, Policy action
# schedule and full 1000-sequence evaluation. Runtime preparation alone does not
# establish score parity.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
MOUNT_ROOT="${1:-/root/autodl-fs/benchmarks}"
BUILD_TMP="${TMPDIR:-/tmp}"
STAGING="$(mktemp -d "${BUILD_TMP}/.calvin-build.XXXXXX")"
trap 'rm -rf -- "${STAGING}"' EXIT

MANIFEST="${SCRIPT_DIR}/benchmark-image.json"
SOURCE_ROOT="${MOUNT_ROOT}/calvin/benchmark/calvin"
ENV_PREFIX="${STAGING}/calvin"
ENV_PYTHON="${ENV_PREFIX}/bin/python"
DATASET_ROOT="${MOUNT_ROOT}/calvin/smoke-assets/suite/minimal_task_D_D"
ENV_ARCHIVE="${MOUNT_ROOT}/calvin/artifacts/calvin-env.tar"

CALVIN_URL="${ROBOSCOPE_CALVIN_URL:-https://github.com/mees/calvin}"
CALVIN_COMMIT="${ROBOSCOPE_CALVIN_COMMIT:-fa03f01f19c65920e18cf37398a9ce859274af76}"
CALVIN_ENV_URL="${ROBOSCOPE_CALVIN_ENV_URL:-https://github.com/mees/calvin_env}"
CALVIN_ENV_COMMIT="${ROBOSCOPE_CALVIN_ENV_COMMIT:-797142c588c21e76717268b7b430958dbd13bf48}"
PYTHON_VERSION="3.8.20"
REQUIREMENTS="${SCRIPT_DIR}/runtime-requirements.txt"
# The starVLA revision whose eval_sequences.json matches the SHA-256 the SDK
# adapter asserts.
EVAL_SEQUENCES_URL="${ROBOSCOPE_CALVIN_EVAL_SEQUENCES_URL:-https://raw.githubusercontent.com/starVLA/starVLA/074ce6546f47caf401805ccf39e5776507a94790/examples/calvin/eval_files/eval_sequences.json}"
EVAL_SEQUENCES_SHA256="c3689911565037d399d779c655e9e943cbd5bfbfbf44045e2b3b1daa8b1a2317"

# Ignore any pip configuration on the build host. GPU images commonly ship a
# /etc/pip.conf that adds an unreachable NVIDIA index, which pip then retries for
# every package, and a host-level index must not decide what a runtime resolves
# to.
export PIP_CONFIG_FILE=/dev/null
# CALVIN's dependencies still fork parallel compilers; one job at a time avoids
# the OOM killer reporting a bare "Error compiling objects".
MAX_JOBS="${ROBOSCOPE_BUILD_MAX_JOBS:-1}"
export MAX_JOBS

# micromamba is a drop-in conda replacement and is what the available Linux
# builders ship, so accept either.
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

if [[ "$(uname -s):$(uname -m)" != "Linux:x86_64" ]]; then
  echo "CALVIN runtime builds require Linux x86_64" >&2
  exit 3
fi
if [[ "${EUID}" -ne 0 ]]; then
  echo "CALVIN runtime builds must run as root" >&2
  exit 3
fi
[[ -f "${MANIFEST}" && -f "${REQUIREMENTS}" ]] || {
  echo "CALVIN runtime input is missing: ${MANIFEST}" >&2
  exit 3
}

if [[ -e "${SOURCE_ROOT}" || -e "${ENV_ARCHIVE}" ]]; then
  echo "CALVIN runtime already exists; remove it for a clean rebuild: ${SOURCE_ROOT}" >&2
  exit 4
fi
if [[ -L "/root/benchmark-runtime/calvin/runtime/root" ]]; then
  echo "CALVIN compact runtime is linked at /root/benchmark-runtime/calvin; roll it back first" >&2
  exit 4
fi

install -d -m 0755 "${SOURCE_ROOT%/*}"
"${CONDA_BIN}" create --yes --quiet -c conda-forge \
  --prefix "${ENV_PREFIX}" "python=${PYTHON_VERSION}"

git clone --quiet --no-checkout "${CALVIN_URL}" "${SOURCE_ROOT}"
git -C "${SOURCE_ROOT}" checkout --quiet --detach "${CALVIN_COMMIT}"
ACTUAL_CALVIN_COMMIT="$(git -C "${SOURCE_ROOT}" rev-parse HEAD)"
[[ "${ACTUAL_CALVIN_COMMIT:0:${#CALVIN_COMMIT}}" == "${CALVIN_COMMIT}" ]] || {
  echo "CALVIN is at ${ACTUAL_CALVIN_COMMIT}, expected ${CALVIN_COMMIT}" >&2
  exit 5
}
git -C "${SOURCE_ROOT}" submodule update --init --recursive
# CALVIN's own submodule pointer does not name the calvin_env revision the lost
# tree carried. The inventory matches 797142c byte for byte across 282 of 287
# files, so override the pointer rather than trusting it.
git -C "${SOURCE_ROOT}/calvin_env" fetch --quiet origin
git -C "${SOURCE_ROOT}/calvin_env" checkout --quiet --detach "${CALVIN_ENV_COMMIT}"
ACTUAL_CALVIN_ENV_COMMIT="$(git -C "${SOURCE_ROOT}/calvin_env" rev-parse HEAD)"
[[ "${ACTUAL_CALVIN_ENV_COMMIT:0:${#CALVIN_ENV_COMMIT}}" == "${CALVIN_ENV_COMMIT}" ]] || {
  echo "calvin_env is at ${ACTUAL_CALVIN_ENV_COMMIT}, expected ${CALVIN_ENV_COMMIT}" >&2
  exit 5
}
git -C "${SOURCE_ROOT}/calvin_env" submodule update --init --recursive
EXPECTED_TACTO_COMMIT="$(git -C "${SOURCE_ROOT}/calvin_env" ls-tree HEAD tacto | awk '{print $3}')"
ACTUAL_TACTO_COMMIT="$(git -C "${SOURCE_ROOT}/calvin_env/tacto" rev-parse HEAD)"
# tacto is pinned by the calvin_env submodule pointer and matches the inventory.
[[ "${ACTUAL_TACTO_COMMIT}" == "${EXPECTED_TACTO_COMMIT}" ]] || {
  echo "tacto is at ${ACTUAL_TACTO_COMMIT}, expected ${EXPECTED_TACTO_COMMIT}" >&2
  exit 5
}

# Pin the build toolchain to what CALVIN itself pins. calvin_models/requirements.txt
# asks for setuptools 57.5.0, and pyhash's setup.py still uses use_2to3, which
# setuptools removed in 58, so a modern setuptools cannot build it at all.
PYTHONNOUSERSITE=1 "${ENV_PYTHON}" -m pip install --no-cache-dir \
  "setuptools==57.5.0" "wheel==0.45.1" "cmake==4.4.3"

# CALVIN's own install.sh, which the lost tree's three egg-info directories show
# was followed, installs tacto, then calvin_env, then calvin_models.
PYTHONNOUSERSITE=1 "${ENV_PYTHON}" -m pip install --no-cache-dir --no-deps \
  --no-build-isolation -e "${SOURCE_ROOT}/calvin_env/tacto"
PYTHONNOUSERSITE=1 "${ENV_PYTHON}" -m pip install --no-cache-dir --no-deps \
  --no-build-isolation -e "${SOURCE_ROOT}/calvin_env"
PYTHONNOUSERSITE=1 "${ENV_PYTHON}" -m pip install --no-cache-dir --no-deps \
  --no-build-isolation -e "${SOURCE_ROOT}/calvin_models"

# The recovered dependency set is complete. Install without resolving substitutes
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
        # calvin_scene_D_eval.yaml interpolates ${data_path}; the SDK replaces it
        # with the asset root at runtime, so seed a placeholder here.
        "data_path": "/calvin/assets",
        # play_table_env.yaml interpolates ${use_vr}; the SDK passes use_vr=False
        # when it instantiates, but the config has to resolve first.
        "use_vr": False,
    }
)
resolved = OmegaConf.to_container(root, resolve=True)
merged = OmegaConf.create({"env": resolved["env"]})
# The SDK overwrites scene_cfg.data_path itself; make the placeholder explicit so
# the committed config is valid on its own.
merged.env.scene_cfg.data_path = "/calvin/assets"
OmegaConf.save(merged, output)
print(f"wrote {output}")
PY

curl --fail --location --silent --show-error \
  --output "${DATASET_ROOT}/eval_sequences.json" "${EVAL_SEQUENCES_URL}"
ACTUAL_SEQUENCES_SHA256="$(sha256sum "${DATASET_ROOT}/eval_sequences.json" | awk '{print $1}')"
[[ "${ACTUAL_SEQUENCES_SHA256}" == "${EVAL_SEQUENCES_SHA256}" ]] || {
  echo "CALVIN evaluation sequences are ${ACTUAL_SEQUENCES_SHA256}, expected ${EVAL_SEQUENCES_SHA256}" >&2
  exit 5
}
SEQUENCE_COUNT="$(
  PYTHONNOUSERSITE=1 "${ENV_PYTHON}" -I -c \
    'import json,sys; print(len(json.load(open(sys.argv[1]))))' \
    "${DATASET_ROOT}/eval_sequences.json"
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

# Instantiate the environment the way the SDK worker does. This is the only check
# that proves the merged config, the asset tree and the dependency set work
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

install -d -m 0755 "${ENV_ARCHIVE%/*}"
tar --sort=name --mtime=@0 --owner=0 --group=0 --numeric-owner \
  -cf "${ENV_ARCHIVE}.tmp" \
  -C "${STAGING}" calvin
mv -- "${ENV_ARCHIVE}.tmp" "${ENV_ARCHIVE}"
tar --list --file "${ENV_ARCHIVE}" > "${STAGING}/archive-members.txt"
[[ "$(head -n 1 "${STAGING}/archive-members.txt")" == "calvin/" ]] || {
  echo "CALVIN archive does not start at the calvin root" >&2
  exit 5
}
grep -qx "calvin/bin/python" "${STAGING}/archive-members.txt" || {
  echo "CALVIN archive is missing calvin/bin/python" >&2
  exit 5
}

echo "Built CALVIN source tree at ${SOURCE_ROOT}"
echo "Built ${ENV_ARCHIVE}"
echo "CALVIN ${CALVIN_COMMIT} with calvin_env ${CALVIN_ENV_COMMIT}"

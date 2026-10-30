#!/usr/bin/env bash
set -Eeuo pipefail

# One-shot builder for the RoboCasa GR1 Tabletop benchmark runtime.
#
# The shared-disk runtime is the byte-exact output of this committed script: it
# creates the pinned conda environment and the pinned StarVLA source tree at
# their final shared paths. compact-shared-runtimes.sh then turns that expanded
# tree into the compact archive-plus-symlink layout, so this script never writes
# the archive or the symlink itself.
#
# Both inputs are committed next to this script:
#   conda-specs.txt           exact name-version-build strings recovered from the
#                             lost runtime's conda-meta directory
#   runtime-requirements.txt  exact pip versions recovered from the lost
#                             runtime's top-level *.dist-info directories
#
# The lost runtime installed robocasa and robosuite as editable checkouts under
# robocasa_runtime_deps/ rather than from an index. Both were located by matching
# the 2026-09-02 inventory file by file against upstream:
#   robocasa-gr1-tabletop-tasks   robocasa/robocasa-gr1-tabletop-tasks at its tip
#                                 4840e67, 249 files byte-identical. This fork's
#                                 robocasa/__init__.py widens
#                                 the numpy assertion to 1.26.4 and robosuite to
#                                 1.5.{0,1}; upstream robocasa 0.2.0 rejects both
#                                 and cannot be imported in this environment.
#   robosuite                     ARISE-Initiative/robosuite at tag v1.5.1, 1270
#                                 files byte-identical.
#
# The StarVLA tree is separate: the SDK worker imports
# examples/simBenchmarks/Robocasa_tabletop/eval_files/simulation_env.py from it.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
MOUNT_ROOT="${1:-/root/autodl-fs/benchmarks}"
RUNTIME_ROOT="${MOUNT_ROOT}/robocasa/runtime/root"
BUILD_TMP="${TMPDIR:-/tmp}"
STAGING="$(mktemp -d "${BUILD_TMP}/.robocasa-build.XXXXXX")"
trap 'rm -rf -- "${STAGING}"' EXIT
CONDA_SPECS="${SCRIPT_DIR}/conda-specs.txt"
REQUIREMENTS="${SCRIPT_DIR}/runtime-requirements.txt"
MANIFEST="${SCRIPT_DIR}/benchmark-image.json"
ENV_NAME="robocasa"
ENV_PYTHON="${RUNTIME_ROOT}/miniconda3/envs/${ENV_NAME}/bin/python"
STARVLA_ROOT="${RUNTIME_ROOT}/starVLA"
STARVLA_URL="${ROBOSCOPE_STARVLA_URL:-https://github.com/starVLA/starVLA}"
DEPS_ROOT="${RUNTIME_ROOT}/robocasa_runtime_deps"
ROBOCASA_REPO_URL="${ROBOSCOPE_ROBOCASA_REPO_URL:-https://github.com/robocasa/robocasa-gr1-tabletop-tasks}"
ROBOCASA_REPO_COMMIT="${ROBOSCOPE_ROBOCASA_REPO_COMMIT:-4840e671596f93ca03651524b9f72ffb1aadfeff}"
ROBOSUITE_REPO_URL="${ROBOSCOPE_ROBOSUITE_REPO_URL:-https://github.com/ARISE-Initiative/robosuite}"
ROBOSUITE_REPO_COMMIT="${ROBOSCOPE_ROBOSUITE_REPO_COMMIT:-a071383d53568ab798eb315c0e95357911be922d}"
LOCAL_ROOT="/root/benchmark-runtime/robocasa/runtime/root"

if [[ "$(uname -s):$(uname -m)" != "Linux:x86_64" ]]; then
  echo "RoboCasa runtime builds require Linux x86_64" >&2
  exit 3
fi
if [[ "${EUID}" -ne 0 ]]; then
  echo "RoboCasa runtime builds must run as root" >&2
  exit 3
fi
for required in "${CONDA_SPECS}" "${REQUIREMENTS}" "${MANIFEST}"; do
  [[ -f "${required}" ]] || {
    echo "RoboCasa runtime input is missing: ${required}" >&2
    exit 3
  }
done
grep -q '^python-3\.10\.' "${CONDA_SPECS}" || {
  echo "RoboCasa conda specs must pin CPython 3.10" >&2
  exit 3
}
for pin in "robocasa==0.2.0" "robosuite==1.5.1" "mujoco==" "numpy==1.26.4"; do
  grep -q "^${pin}" "${REQUIREMENTS}" || {
    echo "RoboCasa runtime lock must pin ${pin}" >&2
    exit 3
  }
done
# micromamba is a drop-in conda replacement and is what the available Linux
# builders ship, so accept either.
# Ignore any pip configuration on the build host. GPU images commonly ship a
# /etc/pip.conf that adds an unreachable NVIDIA index, which pip then retries
# for every package, and a host-level index must not decide what a pinned
# runtime resolves to. The explicit index arguments below are the whole story.
export PIP_CONFIG_FILE=/dev/null
# cuRobo compiles CUDA kernels for several architectures at once. Parallel
# cicc processes exhaust memory on a normal build host and the kernel OOM
# killer takes them down, which surfaces only as "Error compiling objects".
# One job at a time builds reliably; override for a larger host.
MAX_JOBS="${ROBOSCOPE_BUILD_MAX_JOBS:-1}"
export MAX_JOBS
CONDA_BIN=""
for candidate in conda micromamba; do
  if command -v "${candidate}" >/dev/null 2>&1; then
    CONDA_BIN="$(command -v "${candidate}")"
    break
  fi
done
[[ -n "${CONDA_BIN}" ]] || {
  echo "conda or micromamba is required to build the RoboCasa runtime" >&2
  exit 3
}

# conda-specs.txt keeps the exact conda-meta directory names, which use the
# legacy name-version-build form. conda understands that spelling but micromamba
# parses the whole string as a package name, so normalise to name=version=build
# before handing the specs to either solver. The recovered build strings
# resolve only from the defaults channel, so that is the channel the original
# environment was built from.
readarray -t CONDA_PACKAGES < <(python3 - "${CONDA_SPECS}" <<'INNER'
import re
import sys

packages = []
for line in open(sys.argv[1], encoding="utf-8"):
    line = line.strip()
    if not line or line.startswith("#"):
        continue
    match = re.match(r"^(.*)-([0-9][^-]*)-([^-]+)$", line)
    if not match:
        raise SystemExit(f"unparsable conda spec: {line}")
    name, version, build = match.groups()
    packages.append(f"{name}={version}={build}")
if not packages:
    raise SystemExit("no conda specs found")
print("\n".join(packages))
INNER
)
[[ "${#CONDA_PACKAGES[@]}" -gt 0 ]] || {
  echo "RoboCasa conda specs are empty" >&2
  exit 3
}

if [[ -e "${RUNTIME_ROOT}" ]]; then
  echo "RoboCasa runtime already exists; remove it for a clean rebuild: ${RUNTIME_ROOT}" >&2
  exit 4
fi
if [[ -L "${LOCAL_ROOT}" ]]; then
  echo "RoboCasa compact runtime is linked at ${LOCAL_ROOT}; roll it back first" >&2
  exit 4
fi

install -d -m 0755 "${RUNTIME_ROOT}"
"${CONDA_BIN}" create --yes --quiet -c defaults \
  --prefix "${RUNTIME_ROOT}/miniconda3/envs/${ENV_NAME}" "${CONDA_PACKAGES[@]}"
# --no-deps: the recovered lock is the complete set of distributions the lost
# environment had, so installing with dependency resolution can only substitute
# versions. RoboCasa proves the point: opencv-python 5.0.0.93 requires numpy>=2
# while the lock pins numpy 1.26.4, and RoboTwin pairs torchvision 0.19.1 with
# torch 2.4.1+cu121, which that wheel rejects. Both states are what ran in
# production, so they are reproduced rather than resolved.
# robocasa and robosuite come from the editable checkouts under robocasa_runtime_deps,
# so neither can be resolved from an index. Install the lock without both.
grep -v -e '^robocasa==' -e '^robosuite==' "${REQUIREMENTS}" > "${STAGING}/requirements.txt"
PYTHONNOUSERSITE=1 "${ENV_PYTHON}" -m pip install --no-cache-dir --no-deps \
  --requirement "${STAGING}/requirements.txt"
install -d -m 0755 "${DEPS_ROOT}"
git clone --quiet --no-checkout "${ROBOCASA_REPO_URL}" "${DEPS_ROOT}/robocasa-gr1-tabletop-tasks"
git -C "${DEPS_ROOT}/robocasa-gr1-tabletop-tasks" checkout --quiet --detach "${ROBOCASA_REPO_COMMIT}"
ACTUAL_ROBOCASA_COMMIT="$(git -C "${DEPS_ROOT}/robocasa-gr1-tabletop-tasks" rev-parse HEAD)"
# Accept an abbreviated expectation so the pin reads the way it is usually quoted.
[[ "${ACTUAL_ROBOCASA_COMMIT:0:${#ROBOCASA_REPO_COMMIT}}" == "${ROBOCASA_REPO_COMMIT}" ]] || {
  echo "robocasa-gr1-tabletop-tasks is at ${ACTUAL_ROBOCASA_COMMIT}, expected ${ROBOCASA_REPO_COMMIT}" >&2
  exit 5
}
git clone --quiet --no-checkout "${ROBOSUITE_REPO_URL}" \
  "${DEPS_ROOT}/robosuite"
git -C "${DEPS_ROOT}/robosuite" checkout --quiet --detach "${ROBOSUITE_REPO_COMMIT}"
ACTUAL_ROBOSUITE_SHA="$(git -C "${DEPS_ROOT}/robosuite" rev-parse HEAD)"
[[ "${ACTUAL_ROBOSUITE_SHA}" == "${ROBOSUITE_REPO_COMMIT}" ]] || {
  echo "robosuite is at ${ACTUAL_ROBOSUITE_SHA}, expected ${ROBOSUITE_REPO_COMMIT}" >&2
  exit 5
}
PYTHONNOUSERSITE=1 "${ENV_PYTHON}" -m pip install --no-cache-dir --no-deps \
  --no-build-isolation -e "${DEPS_ROOT}/robocasa-gr1-tabletop-tasks"
PYTHONNOUSERSITE=1 "${ENV_PYTHON}" -m pip install --no-cache-dir --no-deps \
  --no-build-isolation -e "${DEPS_ROOT}/robosuite"
ROBOCASA_VERSION="$(
  PYTHONNOUSERSITE=1 "${ENV_PYTHON}" -I -c \
    'from importlib.metadata import version; print(version("robocasa"))'
)"
[[ "${ROBOCASA_VERSION}" == "0.2.0" ]] || {
  echo "RoboCasa installed as ${ROBOCASA_VERSION}, expected 0.2.0" >&2
  exit 5
}

PYTHONNOUSERSITE=1 "${ENV_PYTHON}" -I \
  "${SCRIPT_DIR}/build-robocasa-assets.py" \
  "${DEPS_ROOT}/robocasa-gr1-tabletop-tasks" \
  "${ROBOSCOPE_ROBOCASA_ASSETS_DIR:-${STAGING}/assets}"

STARVLA_COMMIT="${ROBOSCOPE_STARVLA_COMMIT:-2e5f239bc0b1661d7d556bdba5071f3041544cc6}"
[[ "${STARVLA_COMMIT}" =~ ^[0-9a-f]{7,40}$ ]] || {
  echo "StarVLA commit must be a hex hash: ${STARVLA_COMMIT}" >&2
  exit 3
}
# Fetch the source archive rather than cloning: starVLA does not serve an
# arbitrary SHA over git fetch, and the runtime needs only the working tree.
curl --fail --location --silent --show-error \
  --output "${STAGING}/starvla.tar.gz" \
  "${STARVLA_URL}/archive/${STARVLA_COMMIT}.tar.gz"
tar -xzf "${STAGING}/starvla.tar.gz" -C "${STAGING}"
STARVLA_EXTRACT="$(find "${STAGING}" -maxdepth 1 -type d -name 'starVLA-*' | head -n 1)"
[[ -n "${STARVLA_EXTRACT}" ]] || {
  echo "StarVLA archive did not extract a starVLA directory" >&2
  exit 5
}
mv -- "${STARVLA_EXTRACT}" "${STARVLA_ROOT}"

for relative in \
  examples/simBenchmarks/Robocasa_tabletop/eval_files/simulation_env.py \
  examples/simBenchmarks/Robocasa_tabletop/eval_files/wrappers/multistep_wrapper.py
do
  [[ -f "${STARVLA_ROOT}/${relative}" ]] || {
    echo "StarVLA checkout is missing the RoboCasa evaluator: ${relative}" >&2
    exit 5
  }
done

# The lost tree's simulation_env.py was 13556 bytes. Every upstream revision from
# the file's creation on 2026-07-01 onwards carries 12962 bytes, and the file has
# exactly one commit in its history, so the production copy was hand-edited and
# that edit was never committed. It is unrecoverable, and this build therefore
# ships the upstream evaluator, which is the committed source of truth. Anyone
# comparing against historical RoboCasa results must know they differ.
EVALUATOR="${STARVLA_ROOT}/examples/simBenchmarks/Robocasa_tabletop/eval_files/simulation_env.py"
EVALUATOR_BYTES="$(stat -c '%s' "${EVALUATOR}")"
if [[ "${EVALUATOR_BYTES}" != "12962" ]]; then
  echo "RoboCasa evaluator is ${EVALUATOR_BYTES} bytes, expected 12962 for an upstream checkout" >&2
  exit 5
fi
echo "WARNING: the production RoboCasa evaluator was 13556 bytes, 594 bytes of" >&2
echo "WARNING: uncommitted local edits that no revision carries. This build uses" >&2
echo "WARNING: the upstream evaluator, so results are not comparable to the" >&2
echo "WARNING: historical RoboCasa numbers without re-validating them." >&2

env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 "${ENV_PYTHON}" -I - "${STARVLA_ROOT}" <<'PY'
import sys
from pathlib import Path

benchmark_root = Path(sys.argv[1])
sys.path.insert(0, str(benchmark_root))

import mujoco
import numpy
import robocasa
import robosuite

from importlib.metadata import version

for distribution, expected in (
    ("robocasa", "0.2.0"),
    ("robosuite", "1.5.1"),
    ("mujoco", "3.2.6"),
    ("numpy", "1.26.4"),
):
    actual = version(distribution)
    if actual != expected:
        raise SystemExit(f"RoboCasa runtime {distribution} is {actual}, expected {expected}")

module = __import__(
    "examples.simBenchmarks.Robocasa_tabletop.eval_files.simulation_env",
    fromlist=["SimulationConfig"],
)
for name in ("MultiStepConfig", "SimulationConfig", "VideoConfig", "_create_single_env"):
    if not hasattr(module, name):
        raise SystemExit(f"RoboCasa evaluator is missing {name}")

print("RoboCasa runtime imports passed")
PY

echo "Built RoboCasa runtime at ${RUNTIME_ROOT} from StarVLA ${STARVLA_COMMIT}"

#!/usr/bin/env bash
set -Eeuo pipefail

# Build the RoboCasa GR1 Tabletop benchmark runtime under <install-dir>:
#
#   <install-dir>/
#     benchmark-manifest.json   this directory's manifest; runtime paths are relative to it
#     env/                      conda environment (CPython 3.10), runtime.python = env/bin/python
#     robocasa_runtime_deps/    editable robocasa and robosuite checkouts, plus the kitchen
#                               and GR1 assets inside the robocasa checkout
#     starVLA/                  pinned StarVLA tree, runtime.benchmark_root
#
# Usage: build-runtime.sh <install-dir>
# Then:  python -m roboscope_kit.runtime.benchmark --benchmark robocasa \
#          --benchmark-manifest <install-dir>/benchmark-manifest.json ...
#
# Environment:
#   KIT_PYTHON                     Python with roboscope-kit installed (default python3); runs the
#                                  final self-test. The RoboCasa environment does not need the kit.
#   ROBOSCOPE_ROBOCASA_ASSETS_DIR  optional download cache for the asset ZIPs listed in
#                                  asset-inputs.json; ZIPs already there are verified, not refetched.
#   ROBOSCOPE_BUILD_MAX_JOBS       parallel compile jobs (default 1).
#
# Re-running on an existing <install-dir> reuses every finished step: the environment, both
# checkouts (verified against their pins), the extracted assets and the StarVLA tree.
#
# Both pin files are committed next to this script:
#   conda-specs.txt           exact name-version-build strings recovered from the
#                             original runtime's conda-meta directory
#   runtime-requirements.txt  exact pip versions recovered from the original
#                             runtime's top-level *.dist-info directories
#
# The original runtime installed robocasa and robosuite as editable checkouts under
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
# The StarVLA tree is separate: the kit's worker imports
# examples/simBenchmarks/Robocasa_tabletop/eval_files/simulation_env.py from it.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
[[ $# -eq 1 ]] || {
  echo "usage: $0 <install-dir>" >&2
  exit 2
}
install -d -m 0755 "$1"
INSTALL_DIR="$(cd -- "$1" && pwd -P)"

CONDA_SPECS="${SCRIPT_DIR}/conda-specs.txt"
REQUIREMENTS="${SCRIPT_DIR}/runtime-requirements.txt"
MANIFEST="${SCRIPT_DIR}/benchmark-manifest.json"
ASSET_INPUTS="${SCRIPT_DIR}/asset-inputs.json"
ENV_PREFIX="${INSTALL_DIR}/env"
ENV_PYTHON="${ENV_PREFIX}/bin/python"
STARVLA_ROOT="${INSTALL_DIR}/starVLA"
STARVLA_URL="${ROBOSCOPE_STARVLA_URL:-https://github.com/starVLA/starVLA}"
STARVLA_COMMIT="${ROBOSCOPE_STARVLA_COMMIT:-2e5f239bc0b1661d7d556bdba5071f3041544cc6}"
DEPS_ROOT="${INSTALL_DIR}/robocasa_runtime_deps"
ROBOCASA_ROOT="${DEPS_ROOT}/robocasa-gr1-tabletop-tasks"
ROBOCASA_REPO_URL="${ROBOSCOPE_ROBOCASA_REPO_URL:-https://github.com/robocasa/robocasa-gr1-tabletop-tasks}"
ROBOCASA_REPO_COMMIT="${ROBOSCOPE_ROBOCASA_REPO_COMMIT:-4840e671596f93ca03651524b9f72ffb1aadfeff}"
ROBOSUITE_ROOT="${DEPS_ROOT}/robosuite"
ROBOSUITE_REPO_URL="${ROBOSCOPE_ROBOSUITE_REPO_URL:-https://github.com/ARISE-Initiative/robosuite}"
ROBOSUITE_REPO_COMMIT="${ROBOSCOPE_ROBOSUITE_REPO_COMMIT:-a071383d53568ab798eb315c0e95357911be922d}"
KIT_PYTHON="${KIT_PYTHON:-python3}"

if [[ "$(uname -s):$(uname -m)" != "Linux:x86_64" ]]; then
  echo "RoboCasa runtime builds require Linux x86_64" >&2
  exit 3
fi
for required in "${CONDA_SPECS}" "${REQUIREMENTS}" "${MANIFEST}" "${ASSET_INPUTS}"; do
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
# The robosuite pin is a tag's commit, so it must be the full SHA.
[[ "${ROBOSUITE_REPO_COMMIT}" =~ ^[0-9a-f]{40}$ ]] || {
  echo "robosuite pin must be a 40-character commit SHA: ${ROBOSUITE_REPO_COMMIT}" >&2
  exit 3
}
[[ "${STARVLA_COMMIT}" =~ ^[0-9a-f]{7,40}$ ]] || {
  echo "StarVLA commit must be a hex hash: ${STARVLA_COMMIT}" >&2
  exit 3
}
# Fail before the long build rather than at the final self-test.
"${KIT_PYTHON}" -c 'import roboscope_kit' || {
  echo "KIT_PYTHON=${KIT_PYTHON} cannot import roboscope_kit; install roboscope-kit into it" >&2
  exit 3
}
# Ignore any pip configuration on the build host. GPU hosts commonly ship a
# /etc/pip.conf that adds an unreachable NVIDIA index, which pip then retries
# for every package, and a host-level index must not decide what a pinned
# runtime resolves to. The explicit index arguments below are the whole story.
export PIP_CONFIG_FILE=/dev/null
# Parallel compiler processes exhaust memory on a normal build host and the
# kernel OOM killer takes them down, which surfaces only as "Error compiling
# objects". One job at a time builds reliably; override for a larger host.
MAX_JOBS="${ROBOSCOPE_BUILD_MAX_JOBS:-1}"
export MAX_JOBS
# micromamba is a drop-in conda replacement, so accept either.
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

# Stage inside the install directory: the asset ZIPs are about 4 GB.
STAGING="$(mktemp -d "${INSTALL_DIR}/.robocasa-build.XXXXXX")"
trap 'rm -rf -- "${STAGING}"' EXIT

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

if [[ ! -x "${ENV_PYTHON}" ]]; then
  "${CONDA_BIN}" create --yes --quiet -c defaults --prefix "${ENV_PREFIX}" "${CONDA_PACKAGES[@]}"
fi
# --no-deps: the recovered lock is the complete set of distributions the original
# environment had, so installing with dependency resolution can only substitute
# versions. opencv-python 5.0.0.93 requires numpy>=2 while the lock pins numpy
# 1.26.4; that state is what ran, so it is reproduced rather than resolved.
# robocasa and robosuite come from the editable checkouts under robocasa_runtime_deps,
# so neither can be resolved from an index. Install the lock without both.
grep -v -e '^robocasa==' -e '^robosuite==' "${REQUIREMENTS}" > "${STAGING}/requirements.txt"
PYTHONNOUSERSITE=1 "${ENV_PYTHON}" -m pip install --no-cache-dir --no-deps \
  --requirement "${STAGING}/requirements.txt"

# Clone a pinned checkout once, then insist it is still at the pin on every run.
# An abbreviated pin is accepted so it reads the way it is usually quoted.
pinned_checkout() {
  local url="$1" commit="$2" target="$3" actual
  if [[ ! -d "${target}/.git" ]]; then
    git clone --quiet --no-checkout "${url}" "${target}"
    git -C "${target}" checkout --quiet --detach "${commit}"
  fi
  actual="$(git -C "${target}" rev-parse HEAD)"
  [[ "${actual:0:${#commit}}" == "${commit}" ]] || {
    echo "${target} is at ${actual}, expected ${commit}" >&2
    exit 5
  }
}
install -d -m 0755 "${DEPS_ROOT}"
pinned_checkout "${ROBOCASA_REPO_URL}" "${ROBOCASA_REPO_COMMIT}" "${ROBOCASA_ROOT}"
pinned_checkout "${ROBOSUITE_REPO_URL}" "${ROBOSUITE_REPO_COMMIT}" "${ROBOSUITE_ROOT}"
PYTHONNOUSERSITE=1 "${ENV_PYTHON}" -m pip install --no-cache-dir --no-deps \
  --no-build-isolation -e "${ROBOCASA_ROOT}"
PYTHONNOUSERSITE=1 "${ENV_PYTHON}" -m pip install --no-cache-dir --no-deps \
  --no-build-isolation -e "${ROBOSUITE_ROOT}"
ROBOCASA_VERSION="$(
  PYTHONNOUSERSITE=1 "${ENV_PYTHON}" -I -c \
    'from importlib.metadata import version; print(version("robocasa"))'
)"
[[ "${ROBOCASA_VERSION}" == "0.2.0" ]] || {
  echo "RoboCasa installed as ${ROBOCASA_VERSION}, expected 0.2.0" >&2
  exit 5
}

# build-robocasa-assets.py records asset-inputs.json in the checkout once every
# archive is verified and extracted, so an identical record means the assets are done.
ASSET_RECORD="${ROBOCASA_ROOT}/roboscope-assets-provenance.json"
if ! python3 - "${ASSET_INPUTS}" "${ASSET_RECORD}" <<'PY'
import json
import sys
from pathlib import Path

record = Path(sys.argv[2])
same = record.is_file() and json.loads(record.read_text()) == json.loads(Path(sys.argv[1]).read_text())
raise SystemExit(0 if same else 1)
PY
then
  PYTHONNOUSERSITE=1 "${ENV_PYTHON}" -I \
    "${SCRIPT_DIR}/build-robocasa-assets.py" \
    "${ROBOCASA_ROOT}" \
    "${ROBOSCOPE_ROBOCASA_ASSETS_DIR:-${STAGING}/assets}"
fi

if [[ ! -d "${STARVLA_ROOT}" ]]; then
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
fi

for relative in \
  examples/simBenchmarks/Robocasa_tabletop/eval_files/simulation_env.py \
  examples/simBenchmarks/Robocasa_tabletop/eval_files/wrappers/multistep_wrapper.py
do
  [[ -f "${STARVLA_ROOT}/${relative}" ]] || {
    echo "StarVLA tree is missing the RoboCasa evaluator: ${relative}" >&2
    exit 5
  }
done

# The original tree's simulation_env.py was 13556 bytes. Every upstream revision
# from the file's creation on 2026-07-01 onwards carries 12962 bytes, and the file
# has exactly one commit in its history, so the original copy was hand-edited and
# that edit was never committed. It is unrecoverable, and this build therefore
# ships the upstream evaluator, which is the committed source of truth. Anyone
# comparing against historical RoboCasa results must know they differ.
EVALUATOR="${STARVLA_ROOT}/examples/simBenchmarks/Robocasa_tabletop/eval_files/simulation_env.py"
EVALUATOR_BYTES="$(stat -c '%s' "${EVALUATOR}")"
if [[ "${EVALUATOR_BYTES}" != "12962" ]]; then
  echo "RoboCasa evaluator is ${EVALUATOR_BYTES} bytes, expected 12962 for an upstream checkout" >&2
  exit 5
fi
echo "WARNING: the original RoboCasa evaluator was 13556 bytes, 594 bytes of" >&2
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

install -m 0644 "${MANIFEST}" "${INSTALL_DIR}/benchmark-manifest.json"
echo "Built RoboCasa runtime in ${INSTALL_DIR} from StarVLA ${STARVLA_COMMIT}"

"${KIT_PYTHON}" -m roboscope_kit.runtime.benchmark --benchmark robocasa --self-test \
  --benchmark-manifest "${INSTALL_DIR}/benchmark-manifest.json" \
  --output-dir "${INSTALL_DIR}/self-test"

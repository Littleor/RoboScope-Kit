#!/usr/bin/env bash
set -Eeuo pipefail

# Build the robomimic v0.1 rollout environment under <install-dir>:
#
#   <install-dir>/
#     benchmark-manifest.json   this directory's manifest; runtime paths are relative to it
#     python/                   the uv-managed CPython 3.11 the environment runs on
#     env/                      venv with exactly runtime-requirements.txt
#     pip-freeze.txt            what env/ actually contains
#
# Usage: build-runtime.sh <install-dir>
# Then:  python -m roboscope_kit.runtime.benchmark --benchmark robomimic --suite ph_low_dim \
#          --benchmark-manifest <install-dir>/benchmark-manifest.json ...
#
# The environment pairs robomimic v0.4.0 with the robosuite v1.4.1 branch on the DeepMind
# MuJoCo 2.3.7 bindings. The robomimic v0.1 model zoo was trained on the mujoco-py based
# robosuite offline_study branch, which needs MuJoCo 2.0 and mujoco-py 2.0 and does not
# build on Python 3.11; see ROBOMIMIC_EVALUATION_PROTOCOL.md for that boundary.
#
# The build regenerates the 50 seeded initial states of every task through the kit
# worker's own reset path and rejects an environment whose digests differ from the
# constants in the kit's robomimic task table. Re-running the script on an existing
# <install-dir> reuses the venv and re-applies the lock.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
KIT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd -P)"
[[ $# -eq 1 ]] || {
  echo "usage: $0 <install-dir>" >&2
  exit 2
}
install -d -m 0755 "$1"
INSTALL_DIR="$(cd -- "$1" && pwd -P)"

MANIFEST="${SCRIPT_DIR}/benchmark-manifest.json"
LOCK="${SCRIPT_DIR}/runtime-requirements.txt"
KIT_SOURCE="${KIT_ROOT}/packages/roboscope-kit/src"
PYTHON_INSTALL_DIR="${INSTALL_DIR}/python"
VENV="${INSTALL_DIR}/env"
PYTHON="${VENV}/bin/python"
PYTHON_VERSION="3.11"
ROBOMIMIC_URL="https://github.com/ARISE-Initiative/robomimic.git"
# robomimic v0.4.0
ROBOMIMIC_COMMIT="7c66e7a41b5d9dcc905b1a68346bfee1b49b79c9"
# Head of robosuite's v1.4.1 branch, which robomimic v0.4.0 documents for its datasets.
ROBOSUITE_COMMIT="c848ca848020d0c4ccdd10c5056bd06f2a195ba2"

if [[ "$(uname -s):$(uname -m)" != "Linux:x86_64" ]]; then
  echo "robomimic runtime builds require Linux x86_64" >&2
  exit 3
fi
if [[ ! -f "${MANIFEST}" || ! -f "${LOCK}" || ! -d "${KIT_SOURCE}/roboscope_kit" ]]; then
  echo "robomimic manifest, runtime lock or kit source is missing" >&2
  exit 3
fi
for pin in "mujoco==2.3.7" "torch==" "numpy==" \
  "robomimic @ git+${ROBOMIMIC_URL}@${ROBOMIMIC_COMMIT}" \
  "robosuite @ git+https://github.com/ARISE-Initiative/robosuite.git@${ROBOSUITE_COMMIT}"; do
  grep -qF "${pin}" "${LOCK}" || {
    echo "robomimic runtime lock must pin ${pin}" >&2
    exit 3
  }
done
command -v git >/dev/null 2>&1 || {
  echo "git is required to install the pinned robomimic and robosuite commits" >&2
  exit 3
}
command -v uv >/dev/null 2>&1 || {
  echo "uv 0.11.14 is required to build the robomimic runtime" >&2
  exit 3
}
UV_BIN="$(command -v uv)"
if [[ "$("${UV_BIN}" --version)" != "uv 0.11.14 "* ]]; then
  echo "uv 0.11.14 is required to build the robomimic runtime" >&2
  exit 3
fi

# Keep the interpreter inside <install-dir> so the environment does not depend
# on the user's uv cache.
export UV_PYTHON_INSTALL_DIR="${PYTHON_INSTALL_DIR}"
"${UV_BIN}" python install --no-bin --no-progress "${PYTHON_VERSION}"
if [[ ! -x "${PYTHON}" ]]; then
  "${UV_BIN}" venv \
    --python "${PYTHON_VERSION}" \
    --managed-python \
    --no-python-downloads \
    --relocatable \
    "${VENV}"
fi
# Every package, including the torch CUDA wheels, comes from PyPI.
"${UV_BIN}" pip install \
  --python "${PYTHON}" \
  --no-progress \
  --index-url https://pypi.org/simple \
  --requirements "${LOCK}"
"${UV_BIN}" pip freeze --python "${PYTHON}" > "${INSTALL_DIR}/pip-freeze.txt"

env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 "${PYTHON}" -I - "${ROBOMIMIC_COMMIT}" "${ROBOSUITE_COMMIT}" <<'PY'
import json
import sys
from importlib.metadata import distribution, version

robomimic_commit, robosuite_commit = sys.argv[1:3]
for name, locked in (
    ("mujoco", "2.3.7"),
    ("numpy", "1.26.4"),
    ("torch", "2.4.1"),
    ("robomimic", "0.4.0"),
    ("robosuite", "1.4.1"),
):
    actual = version(name)
    if actual != locked:
        raise SystemExit(f"robomimic runtime {name} is {actual}, expected {locked}")
for name, commit in (("robomimic", robomimic_commit), ("robosuite", robosuite_commit)):
    direct_url = json.loads(distribution(name).read_text("direct_url.json") or "{}")
    if direct_url.get("vcs_info", {}).get("commit_id") != commit:
        raise SystemExit(f"robomimic runtime {name} was not installed from commit {commit}")

import importlib.util

import robomimic.utils.file_utils  # noqa: F401
import robosuite

# The upstream rollout entry point used for the native comparison.
if importlib.util.find_spec("robomimic.scripts.run_trained_agent") is None:
    raise SystemExit("robomimic.scripts.run_trained_agent is not installed")

if robosuite.__version__ != "1.4.1":
    raise SystemExit(f"robosuite reports version {robosuite.__version__}, expected 1.4.1")
print("robomimic runtime versions verified")
PY

# Regenerate every task's seeded initial states with the kit worker's reset path.
env PYTHONPATH="${KIT_SOURCE}" CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 \
  "${PYTHON}" -m roboscope_kit.runtime.benchmarks.robomimic.initial_states

install -m 0644 "${MANIFEST}" "${INSTALL_DIR}/benchmark-manifest.json"
echo "Built robomimic ph_low_dim in ${INSTALL_DIR}"

# The kit runs its own check against the installed manifest. It needs a Python with
# roboscope-kit installed; the robomimic environment does not need it.
if [[ -n "${KIT_PYTHON:-}" ]] || python3 -c 'import roboscope_kit' >/dev/null 2>&1; then
  KIT_PYTHON="${KIT_PYTHON:-python3}"
  "${KIT_PYTHON}" -c 'import roboscope_kit' || {
    echo "KIT_PYTHON=${KIT_PYTHON} cannot import roboscope_kit; install roboscope-kit into it" >&2
    exit 6
  }
  rm -rf -- "${INSTALL_DIR}/self-test"
  "${KIT_PYTHON}" -m roboscope_kit.runtime.benchmark --benchmark robomimic --self-test \
    --benchmark-manifest "${INSTALL_DIR}/benchmark-manifest.json" \
    --output-dir "${INSTALL_DIR}/self-test"
else
  echo "Skipped the kit self-test: python3 cannot import roboscope_kit; set KIT_PYTHON to run it" >&2
fi

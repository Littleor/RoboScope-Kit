#!/usr/bin/env bash
set -Eeuo pipefail

# Build the LIBERO benchmark environment under <install-dir>:
#
#   <install-dir>/
#     benchmark-manifest.json   this directory's manifest; runtime paths are relative to it
#     env/                      conda prefix: CPython 3.10.20 with exactly runtime-requirements.txt
#     LIBERO/                   Lifelong-Robot-Learning/LIBERO at SOURCE_REF; bddl_files,
#                               init_files and assets are part of that checkout
#     config/config.yaml        LIBERO path config written by the kit
#     self-test/                output of the kit's --self-test
#
# Usage: build-runtime.sh <install-dir>
# Then:  python -m roboscope_kit.runtime.benchmark --benchmark libero \
#          --benchmark-manifest <install-dir>/benchmark-manifest.json ...
#
# Requires Linux x86_64, conda or micromamba, git, network access, and KIT_PYTHON
# (default python3): a Python 3.11+ that imports roboscope_kit, used to write and
# check the LIBERO config. Optional input:
# LIBERO_URL, a mirror of the LIBERO repository. Evaluation needs no LIBERO
# demonstration dataset.
#
# Re-running on an existing <install-dir> keeps the checkout (it must be at
# SOURCE_REF) and the environment, and re-applies the lock. The conda prefix and
# config.yaml record absolute paths, so do not move <install-dir>.
#
# Pins: earlier LIBERO environments were never built by a committed script, so
# only their interpreter, CPython 3.10, is on record. SOURCE_REF is the head of
# upstream LIBERO master (2025-03-15), chosen for this script. The interpreter
# and package lock reuse the recovered LIBERO+ ones, a fork of LIBERO on the same
# simulator stack; see conda-specs.txt and runtime-requirements.txt.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
[[ $# -eq 1 ]] || {
  echo "usage: $0 <install-dir>" >&2
  exit 2
}
install -d -m 0755 "$1"
INSTALL_DIR="$(cd -- "$1" && pwd -P)"

MANIFEST="${SCRIPT_DIR}/benchmark-manifest.json"
CONDA_SPECS="${SCRIPT_DIR}/conda-specs.txt"
LOCK="${SCRIPT_DIR}/runtime-requirements.txt"
ENV_PREFIX="${INSTALL_DIR}/env"
ENV_PYTHON="${ENV_PREFIX}/bin/python"
SOURCE_ROOT="${INSTALL_DIR}/LIBERO"
CONFIG_PATH="${INSTALL_DIR}/config/config.yaml"
SOURCE_URL="${LIBERO_URL:-https://github.com/Lifelong-Robot-Learning/LIBERO}"
SOURCE_REF="8f1084e3132a39270c3a13ebe37270a43ece2a01"
TORCH_INDEX_URL="https://download.pytorch.org/whl/cpu"
KIT_PYTHON="${KIT_PYTHON:-python3}"

fail() {
  echo "$1" >&2
  exit "${2:-3}"
}

[[ "$(uname -s):$(uname -m)" == "Linux:x86_64" ]] || fail "LIBERO runtime builds require Linux x86_64"
for required in "${MANIFEST}" "${CONDA_SPECS}" "${LOCK}"; do
  [[ -f "${required}" ]] || fail "LIBERO build input is missing: ${required}"
done
grep -q '^python-3\.10\.20-' "${CONDA_SPECS}" || fail "LIBERO conda specs must pin CPython 3.10.20"
for pin in "libero==0.1.0" "bddl==1.0.1" "robosuite==1.4.0" "mujoco==" "numpy==1.24.4" "torch=="; do
  grep -q "^${pin}" "${LOCK}" || fail "LIBERO runtime lock must pin ${pin}"
done
"${KIT_PYTHON}" -c 'import roboscope_kit' 2>/dev/null ||
  fail "KIT_PYTHON=${KIT_PYTHON} cannot import roboscope_kit; install roboscope-kit into it or set KIT_PYTHON"
CONDA_BIN="$(command -v conda || command -v micromamba || true)"
[[ -n "${CONDA_BIN}" ]] || fail "conda or micromamba is required to build the LIBERO runtime"

# A host pip configuration (GPU hosts often add an unreachable extra index) must
# not decide what a pinned runtime resolves to.
export PIP_CONFIG_FILE=/dev/null

if [[ -d "${SOURCE_ROOT}" ]]; then
  ACTUAL_REF="$(git -C "${SOURCE_ROOT}" rev-parse HEAD)"
  [[ "${ACTUAL_REF}" == "${SOURCE_REF}" ]] ||
    fail "${SOURCE_ROOT} is at ${ACTUAL_REF}, expected ${SOURCE_REF}; remove it to rebuild" 4
else
  STAGING="$(mktemp -d "${INSTALL_DIR}/.build.XXXXXX")"
  trap 'rm -rf -- "${STAGING}"' EXIT
  git clone --quiet --no-checkout "${SOURCE_URL}" "${STAGING}/LIBERO"
  git -C "${STAGING}/LIBERO" checkout --quiet --detach "${SOURCE_REF}"
  mv -- "${STAGING}/LIBERO" "${SOURCE_ROOT}"
fi
for relative in setup.py libero/libero/__init__.py libero/libero/benchmark/__init__.py \
  libero/libero/bddl_files libero/libero/init_files libero/libero/assets/scenes \
  libero/libero/assets/textures libero/libero/assets/serving_region.xml; do
  [[ -e "${SOURCE_ROOT}/${relative}" ]] || fail "LIBERO source tree is missing ${relative}" 5
done

# conda-specs.txt keeps conda-meta's name-version-build spelling. conda accepts
# it but micromamba reads the whole string as a package name, so pass
# name=version=build. The build strings come from conda-forge.
CONDA_PACKAGES=()
while IFS= read -r line; do
  [[ -z "${line}" || "${line}" == \#* ]] && continue
  [[ "${line}" =~ ^(.*)-([0-9][^-]*)-([^-]+)$ ]] || fail "unparsable conda spec: ${line}"
  CONDA_PACKAGES+=("${BASH_REMATCH[1]}=${BASH_REMATCH[2]}=${BASH_REMATCH[3]}")
done < "${CONDA_SPECS}"
if [[ ! -x "${ENV_PYTHON}" ]]; then
  "${CONDA_BIN}" create --yes --quiet -c conda-forge --prefix "${ENV_PREFIX}" "${CONDA_PACKAGES[@]}"
fi
# Install the source tree first: its setup.py declares libero 0.1.0, which PyPI
# never published, so pip then treats that pin of the lock as satisfied.
PYTHONNOUSERSITE=1 "${ENV_PYTHON}" -m pip install --no-cache-dir --no-deps "${SOURCE_ROOT}"
# --no-deps: the lock is a complete distribution set, so dependency resolution
# could only substitute versions. torch 2.5.1+cpu, which LIBERO uses to load the
# initial states, only exists on the PyTorch CPU index.
PYTHONNOUSERSITE=1 "${ENV_PYTHON}" -m pip install --no-cache-dir --no-deps \
  --extra-index-url "${TORCH_INDEX_URL}" --requirement "${LOCK}"

env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 "${ENV_PYTHON}" -I - <<'PY'
from importlib.metadata import version

import bddl
import robosuite
import torch

for distribution, expected in (
    ("libero", "0.1.0"),
    ("bddl", "1.0.1"),
    ("robosuite", "1.4.0"),
    ("numpy", "1.24.4"),
    ("torch", "2.5.1+cpu"),
):
    actual = version(distribution)
    if actual != expected:
        raise SystemExit(f"LIBERO runtime {distribution} is {actual}, expected {expected}")
print("LIBERO runtime imports passed")
PY

# Write the LIBERO path config with the kit, then import LIBERO and the kit's
# worker in the LIBERO interpreter exactly as an evaluation worker does, and
# check both resolve to this install.
KIT_IMPORT_ROOT="$(
  "${KIT_PYTHON}" - "${CONFIG_PATH}" "${SOURCE_ROOT}" <<'PY'
import sys
from pathlib import Path

from roboscope_kit.runtime.benchmark.kit_path import kit_import_root
from roboscope_kit.runtime.benchmarks.libero.adapter import write_libero_config

write_libero_config(Path(sys.argv[1]), Path(sys.argv[2]))
print(kit_import_root())
PY
)"
env -u PYTHONPATH \
  PYTHONPATH="${SOURCE_ROOT}:${KIT_IMPORT_ROOT}" \
  LIBERO_HOME="${SOURCE_ROOT}" \
  LIBERO_CONFIG_PATH="${CONFIG_PATH%/*}" \
  PYTHONDONTWRITEBYTECODE=1 \
  PYTHONNOUSERSITE=1 \
  "${ENV_PYTHON}" - "${CONFIG_PATH}" "${SOURCE_ROOT}" "${KIT_IMPORT_ROOT}" <<'PY'
import sys
from pathlib import Path

from roboscope_kit.runtime.benchmarks.libero.adapter import validate_libero_runtime

validate_libero_runtime(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]))
print("LIBERO config and kit worker imports passed")
PY

install -m 0644 "${MANIFEST}" "${INSTALL_DIR}/benchmark-manifest.json"
rm -rf -- "${INSTALL_DIR}/self-test"
"${KIT_PYTHON}" -m roboscope_kit.runtime.benchmark --benchmark libero --self-test \
  --benchmark-manifest "${INSTALL_DIR}/benchmark-manifest.json" \
  --output-dir "${INSTALL_DIR}/self-test"
echo "Built LIBERO ${SOURCE_REF} in ${INSTALL_DIR}"

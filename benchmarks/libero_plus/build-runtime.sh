#!/usr/bin/env bash
set -Eeuo pipefail

# Build the LIBERO+ benchmark environment under <install-dir>:
#
#   <install-dir>/
#     benchmark-manifest.json   this directory's manifest; runtime paths are relative to it
#     env/                      conda prefix: CPython 3.10.20 with exactly runtime-requirements.txt
#     LIBERO-plus/              sylvestf/LIBERO-plus at SOURCE_REF, with libero/libero/assets
#                               from the pinned Hugging Face assets.zip
#     config/config.yaml        LIBERO path config written by the kit
#     self-test/                output of the kit's --self-test
#
# Usage: build-runtime.sh <install-dir>
# Then:  python -m roboscope_kit.runtime.benchmark --benchmark libero_plus \
#          --benchmark-manifest <install-dir>/benchmark-manifest.json ...
#
# Requires Linux x86_64, conda or micromamba, git, curl, unzip, network access,
# room for the 6.4 GB assets.zip and its extracted tree, and KIT_PYTHON (default
# python3): a Python 3.11+ that imports roboscope_kit, used to write and check the
# LIBERO config.
# Optional inputs:
#   LIBERO_PLUS_ASSETS_ZIP  an already downloaded assets.zip (6.4 GB); its SHA-256
#                           must still match ASSETS_SHA256
#   LIBERO_PLUS_URL         a mirror of the LIBERO-plus repository
#
# Re-running on an existing <install-dir> keeps the checkout (it must be at
# SOURCE_REF), the assets and the environment, and re-applies the lock. The conda
# prefix and config.yaml record absolute paths, so do not move <install-dir>.
#
# The recovered lock pins libero==0.1.0 while PyPI only publishes 0.1.1, and the
# recorded dist-info kept a direct_url.json, so that package was installed from
# the LIBERO-plus source tree, whose setup.py declares 0.1.0. This script
# installs it the same way. The original environment also carried a hand-written
# 000-libero-plus-path.pth whose contents nothing records; the kit's LIBERO+
# adapter puts the source tree on PYTHONPATH for every worker, so it is not
# recreated.

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
SOURCE_ROOT="${INSTALL_DIR}/LIBERO-plus"
CONFIG_PATH="${INSTALL_DIR}/config/config.yaml"
SOURCE_URL="${LIBERO_PLUS_URL:-https://github.com/sylvestf/LIBERO-plus}"
SOURCE_REF="4976dc30028e805ff8094b55501d532c48fec182"
ASSETS_REPO="Sylvest/LIBERO-plus"
ASSETS_REVISION="dd2bd61b7d9a6fef1abc52d606e983b41886a149"
ASSETS_SHA256="96764a4bfbdaea98d4411598caeab235458318fe0f549611b93d1a323027b3cf"
ASSETS_URL="https://huggingface.co/datasets/${ASSETS_REPO}/resolve/${ASSETS_REVISION}/assets.zip"
TORCH_INDEX_URL="https://download.pytorch.org/whl/cpu"
KIT_PYTHON="${KIT_PYTHON:-python3}"

fail() {
  echo "$1" >&2
  exit "${2:-3}"
}

[[ "$(uname -s):$(uname -m)" == "Linux:x86_64" ]] || fail "LIBERO+ runtime builds require Linux x86_64"
for required in "${MANIFEST}" "${CONDA_SPECS}" "${LOCK}"; do
  [[ -f "${required}" ]] || fail "LIBERO+ build input is missing: ${required}"
done
grep -q '^python-3\.10\.20-' "${CONDA_SPECS}" || fail "LIBERO+ conda specs must pin CPython 3.10.20"
for pin in "libero==0.1.0" "bddl==1.0.1" "robosuite==1.4.0" "mujoco==" "numpy==1.24.4"; do
  grep -q "^${pin}" "${LOCK}" || fail "LIBERO+ runtime lock must pin ${pin}"
done
"${KIT_PYTHON}" -c 'import roboscope_kit' 2>/dev/null ||
  fail "KIT_PYTHON=${KIT_PYTHON} cannot import roboscope_kit; install roboscope-kit into it or set KIT_PYTHON"
CONDA_BIN="$(command -v conda || command -v micromamba || true)"
[[ -n "${CONDA_BIN}" ]] || fail "conda or micromamba is required to build the LIBERO+ runtime"

# A host pip configuration (GPU hosts often add an unreachable extra index) must
# not decide what a pinned runtime resolves to.
export PIP_CONFIG_FILE=/dev/null

# Stage downloads inside <install-dir> so the 6.4 GB assets do not depend on the
# size of /tmp, and moves into place stay on one filesystem.
STAGING="$(mktemp -d "${INSTALL_DIR}/.build.XXXXXX")"
trap 'rm -rf -- "${STAGING}"' EXIT

if [[ -d "${SOURCE_ROOT}" ]]; then
  ACTUAL_REF="$(git -C "${SOURCE_ROOT}" rev-parse HEAD)"
  [[ "${ACTUAL_REF}" == "${SOURCE_REF}" ]] ||
    fail "${SOURCE_ROOT} is at ${ACTUAL_REF}, expected ${SOURCE_REF}; remove it to rebuild" 4
else
  git clone --quiet --no-checkout "${SOURCE_URL}" "${STAGING}/LIBERO-plus"
  git -C "${STAGING}/LIBERO-plus" checkout --quiet --detach "${SOURCE_REF}"
  mv -- "${STAGING}/LIBERO-plus" "${SOURCE_ROOT}"
fi
for relative in setup.py libero/libero/__init__.py libero/libero/bddl_files libero/libero/init_files; do
  [[ -e "${SOURCE_ROOT}/${relative}" ]] || fail "LIBERO+ source tree is missing ${relative}" 5
done

# libero/libero/assets is not in the repository. The upstream README says to
# unzip the Hugging Face assets.zip straight into libero/libero, but every entry
# of the shipped zip sits under a machine-specific prefix
# (inspire/hdd/project/.../LIBERO-plus-0/assets), so a plain unzip produces an
# inspire/ tree. Extract to scratch, then move the assets directory into place.
ASSETS_DIR="${SOURCE_ROOT}/libero/libero/assets"
if [[ ! -d "${ASSETS_DIR}" ]]; then
  ASSETS_ZIP="${LIBERO_PLUS_ASSETS_ZIP:-}"
  if [[ -n "${ASSETS_ZIP}" ]]; then
    [[ -f "${ASSETS_ZIP}" ]] || fail "LIBERO_PLUS_ASSETS_ZIP does not exist: ${ASSETS_ZIP}"
  else
    ASSETS_ZIP="${STAGING}/assets.zip"
    curl --fail --location --silent --show-error --output "${ASSETS_ZIP}" "${ASSETS_URL}"
  fi
  [[ "$(sha256sum "${ASSETS_ZIP}" | awk '{print $1}')" == "${ASSETS_SHA256}" ]] ||
    fail "LIBERO+ assets.zip does not match revision ${ASSETS_REVISION} (SHA-256 ${ASSETS_SHA256})" 5
  unzip -q -o "${ASSETS_ZIP}" -d "${STAGING}/assets-extract"
  EXTRACTED="$(find "${STAGING}/assets-extract" -mindepth 1 -maxdepth 12 -type d -name assets -print -quit)"
  [[ -n "${EXTRACTED}" ]] || fail "LIBERO+ assets.zip contains no assets directory" 5
  mv -- "${EXTRACTED}" "${ASSETS_DIR}"
fi
for entry in articulated_objects new_objects scenes stable_hope_objects stable_scanned_objects \
  textures turbosquid_objects serving_region.xml; do
  [[ -e "${ASSETS_DIR}/${entry}" ]] || fail "LIBERO+ assets directory is missing ${entry}" 5
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
# Install the source tree first: it provides libero==0.1.0, which PyPI never
# published, so pip then treats that pin of the lock as satisfied.
PYTHONNOUSERSITE=1 "${ENV_PYTHON}" -m pip install --no-cache-dir --no-deps "${SOURCE_ROOT}"
# --no-deps: the lock is the complete distribution set of the original
# environment, so dependency resolution could only substitute versions.
# torch 2.5.1+cpu only exists on the PyTorch CPU index.
PYTHONNOUSERSITE=1 "${ENV_PYTHON}" -m pip install --no-cache-dir --no-deps \
  --extra-index-url "${TORCH_INDEX_URL}" --requirement "${LOCK}"

env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 "${ENV_PYTHON}" -I - <<'PY'
from importlib.metadata import version

import bddl
import robosuite

for distribution, expected in (
    ("libero", "0.1.0"),
    ("bddl", "1.0.1"),
    ("robosuite", "1.4.0"),
    ("numpy", "1.24.4"),
):
    actual = version(distribution)
    if actual != expected:
        raise SystemExit(f"LIBERO+ runtime {distribution} is {actual}, expected {expected}")
print("LIBERO+ runtime imports passed")
PY

# Write the LIBERO path config with the kit, then import LIBERO and the kit's
# worker in the LIBERO+ interpreter exactly as an evaluation worker does, and
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
print("LIBERO+ config and kit worker imports passed")
PY

install -m 0644 "${MANIFEST}" "${INSTALL_DIR}/benchmark-manifest.json"
"${KIT_PYTHON}" -m roboscope_kit.runtime.benchmark --benchmark libero_plus --self-test \
  --benchmark-manifest "${INSTALL_DIR}/benchmark-manifest.json" \
  --output-dir "${INSTALL_DIR}/self-test"
echo "Built LIBERO+ ${SOURCE_REF} in ${INSTALL_DIR}"

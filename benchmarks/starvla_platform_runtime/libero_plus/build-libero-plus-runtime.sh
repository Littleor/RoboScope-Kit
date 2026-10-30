#!/usr/bin/env bash
set -Eeuo pipefail

# One-shot builder for the LIBERO+ benchmark runtime.
#
# This script produces both shared-disk products the LIBERO+ launcher consumes:
#   the pinned conda environment at liberoplus/runtime/root/miniconda3, which
#   compact-shared-runtimes.sh later turns into the compact archive-plus-symlink
#   layout, and the LIBERO-plus source tree archive at
#   liberoplus/archives/LIBERO-plus.tar, which prepare-runtime.sh extracts into
#   every instance-local runtime root.
#
# Inputs committed next to this script:
#   conda-specs.txt           exact name-version-build strings recovered from the
#                             lost runtime's conda-meta directory
#   runtime-requirements.txt  exact pip versions recovered from the lost
#                             runtime's top-level *.dist-info directories
#
# The recovered lock pins libero==0.1.0 while PyPI only publishes 0.1.1, and the
# lost dist-info kept a direct_url.json, so that package was installed from the
# LIBERO-plus source tree itself, whose setup.py declares version 0.1.0. This
# script installs it the same way instead of substituting the PyPI release.
#
# The lost environment also carried a hand-written 000-libero-plus-path.pth that
# pointed at the shared source tree. Nothing records its contents, and the live
# prepare-runtime.sh already puts LIBERO_HOME on PYTHONPATH for every run, so
# this builder does not recreate it.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
MOUNT_ROOT="${1:-/root/autodl-fs/benchmarks}"
RUNTIME_ROOT="${MOUNT_ROOT}/liberoplus/runtime/root"
CONDA_SPECS="${SCRIPT_DIR}/conda-specs.txt"
REQUIREMENTS="${SCRIPT_DIR}/runtime-requirements.txt"
MANIFEST="${SCRIPT_DIR}/benchmark-image.json"
ENV_NAME="libero-plus"
ENV_PYTHON="${RUNTIME_ROOT}/miniconda3/envs/${ENV_NAME}/bin/python"
SOURCE_ARCHIVE="${MOUNT_ROOT}/liberoplus/archives/LIBERO-plus.tar"
SOURCE_URL="${ROBOSCOPE_LIBERO_PLUS_URL:-https://github.com/sylvestf/LIBERO-plus}"
SOURCE_REF="${ROBOSCOPE_LIBERO_PLUS_REF:-4976dc30028e805ff8094b55501d532c48fec182}"
ASSETS_SOURCE="${ROBOSCOPE_LIBERO_PLUS_ASSETS:-Sylvest/LIBERO-plus}"
ASSETS_REVISION="dd2bd61b7d9a6fef1abc52d606e983b41886a149"
ASSETS_SHA256="96764a4bfbdaea98d4411598caeab235458318fe0f549611b93d1a323027b3cf"
LOCAL_ROOT="/root/benchmark-runtime/libero_plus/runtime/root"

if [[ "$(uname -s):$(uname -m)" != "Linux:x86_64" ]]; then
  echo "LIBERO+ runtime builds require Linux x86_64" >&2
  exit 3
fi
if [[ "${EUID}" -ne 0 ]]; then
  echo "LIBERO+ runtime builds must run as root" >&2
  exit 3
fi
for required in "${CONDA_SPECS}" "${REQUIREMENTS}" "${MANIFEST}"; do
  [[ -f "${required}" ]] || {
    echo "LIBERO+ runtime input is missing: ${required}" >&2
    exit 3
  }
done
grep -q '^python-3\.10\.' "${CONDA_SPECS}" || {
  echo "LIBERO+ conda specs must pin CPython 3.10" >&2
  exit 3
}
for pin in "libero==0.1.0" "bddl==1.0.1" "robosuite==1.4.0" "mujoco==" "numpy==1.24.4"; do
  grep -q "^${pin}" "${REQUIREMENTS}" || {
    echo "LIBERO+ runtime lock must pin ${pin}" >&2
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
  echo "conda or micromamba is required to build the LIBERO+ runtime" >&2
  exit 3
}
TORCH_INDEX_URL="https://download.pytorch.org/whl/cpu"

# conda-specs.txt keeps the exact conda-meta directory names, which use the
# legacy name-version-build form. conda understands that spelling but micromamba
# parses the whole string as a package name, so normalise to name=version=build
# before handing the specs to either solver. The build strings come from
# conda-forge, which is also the only channel carrying the checksum-pinned zstd
# build that compact-shared-runtimes.sh installs.
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
  echo "LIBERO+ conda specs are empty" >&2
  exit 3
}

if [[ -e "${RUNTIME_ROOT}" ]]; then
  echo "LIBERO+ runtime already exists; remove it for a clean rebuild: ${RUNTIME_ROOT}" >&2
  exit 4
fi
if [[ -L "${LOCAL_ROOT}" ]]; then
  echo "LIBERO+ compact runtime is linked at ${LOCAL_ROOT}; roll it back first" >&2
  exit 4
fi

BUILD_TMP="${TMPDIR:-/tmp}"
STAGING="$(mktemp -d "${BUILD_TMP}/liberoplus-build.XXXXXX")"
trap 'rm -rf -- "${STAGING}"' EXIT

git clone --quiet --no-checkout "${SOURCE_URL}" "${STAGING}/LIBERO-plus"
git -C "${STAGING}/LIBERO-plus" checkout --quiet --detach "${SOURCE_REF}"
ACTUAL_SOURCE_REF="$(git -C "${STAGING}/LIBERO-plus" rev-parse HEAD)"

for relative in setup.py libero/libero/__init__.py; do
  [[ -e "${STAGING}/LIBERO-plus/${relative}" ]] || {
    echo "LIBERO+ source tree is missing ${relative}" >&2
    exit 5
  }
done
for directory in bddl_files init_files; do
  [[ -d "${STAGING}/LIBERO-plus/libero/libero/${directory}" ]] || {
    echo "LIBERO+ source tree is missing libero/libero/${directory}" >&2
    exit 5
  }
done

# libero/libero/assets is not part of the repository. The upstream README ships it
# as assets.zip in the Sylvest/LIBERO-plus Hugging Face dataset and tells you to
# unzip it straight into libero/libero, but the shipped zip does not match that
# instruction: every entry sits under
# inspire/hdd/project/embodied-multimodality/public/<author>/libero_new/release/dataset/LIBERO-plus-0/assets,
# so a plain unzip produces an inspire/ tree instead of assets/. Extract to a
# scratch directory, relocate the assets directory, and verify the layout that
# prepare-runtime.sh's source_complete check and the upstream README both name.
ASSETS_URL="https://huggingface.co/datasets/${ASSETS_SOURCE}/resolve/${ASSETS_REVISION}/assets.zip"
ASSETS_ZIP="${ROBOSCOPE_LIBERO_PLUS_ASSETS_ZIP:-}"
if [[ ! -d "${STAGING}/LIBERO-plus/libero/libero/assets" ]]; then
  ASSETS_EXTRACT="${STAGING}/assets-extract"
  install -d -m 0755 "${ASSETS_EXTRACT}"
  if [[ -n "${ASSETS_ZIP}" ]]; then
    [[ -f "${ASSETS_ZIP}" ]] || {
      echo "LIBERO+ assets zip does not exist: ${ASSETS_ZIP}" >&2
      exit 3
    }
  else
    curl --fail --location --silent --show-error \
      --output "${STAGING}/assets.zip" "${ASSETS_URL}"
    ASSETS_ZIP="${STAGING}/assets.zip"
  fi
  [[ "$(sha256sum "${ASSETS_ZIP}" | awk '{print $1}')" == "${ASSETS_SHA256}" ]] || {
    echo "LIBERO+ assets do not match pinned revision ${ASSETS_REVISION}" >&2
    exit 5
  }
  unzip -q -o "${ASSETS_ZIP}" -d "${ASSETS_EXTRACT}"
  # The archive nests assets under a machine-specific prefix, so locate it rather
  # than assuming the README's flat layout.
  ASSETS_SOURCE_DIR=""
  while IFS= read -r candidate; do
    ASSETS_SOURCE_DIR="$(dirname -- "${candidate}")"
    break
  done < <(find "${ASSETS_EXTRACT}" -mindepth 1 -maxdepth 12 -type d -name assets -print)
  [[ -n "${ASSETS_SOURCE_DIR}" ]] || {
    echo "LIBERO+ assets zip contains no assets directory" >&2
    exit 5
  }
  mv -- "${ASSETS_SOURCE_DIR}/assets" "${STAGING}/LIBERO-plus/libero/libero/assets"
  rm -rf -- "${ASSETS_EXTRACT}"
fi
for directory in bddl_files init_files assets; do
  [[ -d "${STAGING}/LIBERO-plus/libero/libero/${directory}" ]] || {
    echo "LIBERO+ source tree is missing libero/libero/${directory}" >&2
    exit 5
  }
done
for entry in \
  articulated_objects \
  new_objects \
  scenes \
  stable_hope_objects \
  stable_scanned_objects \
  textures \
  turbosquid_objects \
  serving_region.xml
do
  [[ -e "${STAGING}/LIBERO-plus/libero/libero/assets/${entry}" ]] || {
    echo "LIBERO+ assets directory is missing ${entry}" >&2
    exit 5
  }
done

install -d -m 0755 "${RUNTIME_ROOT}"
"${CONDA_BIN}" create --yes --quiet -c conda-forge \
  --prefix "${RUNTIME_ROOT}/miniconda3/envs/${ENV_NAME}" "${CONDA_PACKAGES[@]}"
# Install the source tree before the lock: it provides libero==0.1.0, which PyPI
# never published, so installing the lock first cannot resolve that pin. Once the
# source install has registered 0.1.0, pip treats the pin as already satisfied.
PYTHONNOUSERSITE=1 "${ENV_PYTHON}" -m pip install --no-cache-dir --no-deps "${STAGING}/LIBERO-plus"
# --no-deps: the recovered lock is the complete set of distributions the lost
# environment had, so installing with dependency resolution can only substitute
# versions. RoboCasa proves the point: opencv-python 5.0.0.93 requires numpy>=2
# while the lock pins numpy 1.26.4, and RoboTwin pairs torchvision 0.19.1 with
# torch 2.4.1+cu121, which that wheel rejects. Both states are what ran in
# production, so they are reproduced rather than resolved.
# torch 2.5.1+cpu only exists on the PyTorch CPU index.
PYTHONNOUSERSITE=1 "${ENV_PYTHON}" -m pip install --no-cache-dir --no-deps \
  --extra-index-url "${TORCH_INDEX_URL}" \
  --requirement "${REQUIREMENTS}"

env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 "${ENV_PYTHON}" -I - "${STAGING}/LIBERO-plus" <<'PY'
import sys
from pathlib import Path

source_root = Path(sys.argv[1])
sys.path.insert(0, str(source_root))

import bddl
import libero
import numpy
import robosuite

from importlib.metadata import version

for distribution, expected in (
    ("libero", "0.1.0"),
    ("bddl", "1.0.1"),
    ("robosuite", "1.4.0"),
    ("numpy", "1.24.4"),
):
    actual = version(distribution)
    if actual != expected:
        raise SystemExit(f"LIBERO+ runtime {distribution} is {actual}, expected {expected}")

for relative in ("libero/libero/bddl_files", "libero/libero/init_files", "libero/libero/assets"):
    if not (source_root / relative).is_dir():
        raise SystemExit(f"LIBERO+ source tree is missing {relative}")

print("LIBERO+ runtime imports passed")
PY

install -d -m 0755 "${SOURCE_ARCHIVE%/*}"
tar --sort=name --mtime=@0 --owner=0 --group=0 --numeric-owner \
  -cf "${SOURCE_ARCHIVE}.tmp" \
  -C "${STAGING}" LIBERO-plus
mv -- "${SOURCE_ARCHIVE}.tmp" "${SOURCE_ARCHIVE}"
# The archive normalises every mtime to the epoch for reproducibility, so a
# filesystem comparison would report every entry as differing. Read the whole
# archive instead, which validates its structure, and assert the entries the
# launcher depends on are present.
tar --list --file "${SOURCE_ARCHIVE}" > "${STAGING}/archive-members.txt"
[[ "$(head -n 1 "${STAGING}/archive-members.txt")" == "LIBERO-plus/" ]] || {
  echo "LIBERO+ archive does not start at the LIBERO-plus root" >&2
  exit 5
}
for entry in \
  LIBERO-plus/setup.py \
  LIBERO-plus/libero/libero/__init__.py \
  LIBERO-plus/libero/libero/bddl_files/ \
  LIBERO-plus/libero/libero/init_files/ \
  LIBERO-plus/libero/libero/assets/
do
  grep -qx "${entry}" "${STAGING}/archive-members.txt" || {
    echo "LIBERO+ archive is missing ${entry}" >&2
    exit 5
  }
done

echo "Built LIBERO+ runtime at ${RUNTIME_ROOT}"
echo "Built ${SOURCE_ARCHIVE} from LIBERO-plus ${ACTUAL_SOURCE_REF}"

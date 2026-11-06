#!/usr/bin/env bash
set -Eeuo pipefail

# Build the LIBERO-Pro benchmark environment under <install-dir>:
#
#   <install-dir>/
#     benchmark-manifest.json   this directory's manifest; runtime paths are relative to it
#     env/                      conda prefix: CPython 3.11.18 with exactly runtime-requirements.txt
#     LIBERO-PRO/               Zxy-MLlab/LIBERO-PRO at SOURCE_REF (its tree carries the assets and the
#                               base LIBERO suites), plus the BDDL and initial-state files of the 16
#                               released single-perturbation suites from the pinned Hugging Face dataset
#     config/config.yaml        LIBERO path config written by the kit
#     self-test/                output of the kit's --self-test
#
# Usage: build-runtime.sh <install-dir>
# Then:  python -m roboscope_kit.runtime.benchmark --benchmark libero_pro \
#          --benchmark-manifest <install-dir>/benchmark-manifest.json ...
#
# Requires Linux x86_64, conda or micromamba, git, curl, network access, about 2 GB of disk, and
# KIT_PYTHON (default python3): a Python 3.11+ that imports roboscope_kit, used to write and check the
# LIBERO config. Optional inputs:
#   LIBERO_PRO_URL        a mirror of the LIBERO-PRO repository; it must serve SOURCE_REF
#   LIBERO_PRO_FILES_URL  a mirror of the Hugging Face dataset revision: <url>/<path> must serve every
#                         path of task-files.sha256; the checksums still apply
#
# Re-running on an existing <install-dir> keeps the checkout (it must be at SOURCE_REF) and the
# environment, re-checks every task file, and re-applies the lock. The conda prefix and config.yaml
# record absolute paths, so do not move <install-dir>.
#
# Pins: SOURCE_REF is the head of upstream LIBERO-PRO master (2026-03-23). Upstream documents
# Python 3.8 and torch 1.11; this runtime is CPython 3.11.18, the kit's minimum supported interpreter,
# with the LIBERO package lock carried over and only the changes CPython 3.11 needs (see
# runtime-requirements.txt). The upstream repository does not contain the perturbed suites: their
# files come from the zhouxueyang/LIBERO-Pro dataset at FILES_REVISION, each checked against
# task-files.sha256. The dataset's Environment (*_env) suites were never released, and its seven-case
# extension under bddl_files/0*_* needs a parser this source revision does not have, so neither is
# downloaded.

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
TASK_FILES="${SCRIPT_DIR}/task-files.sha256"
ENV_PREFIX="${INSTALL_DIR}/env"
ENV_PYTHON="${ENV_PREFIX}/bin/python"
SOURCE_ROOT="${INSTALL_DIR}/LIBERO-PRO"
CONFIG_PATH="${INSTALL_DIR}/config/config.yaml"
SOURCE_URL="${LIBERO_PRO_URL:-https://github.com/Zxy-MLlab/LIBERO-PRO}"
SOURCE_REF="eafdb809426b13153aa1e4c42d6601844217dfec"
FILES_REPO="zhouxueyang/LIBERO-Pro"
FILES_REVISION="c86fc3b8293185a6f373677018ff3e37f8391602"
FILES_URL="${LIBERO_PRO_FILES_URL:-https://huggingface.co/datasets/${FILES_REPO}/resolve/${FILES_REVISION}}"
TORCH_INDEX_URL="https://download.pytorch.org/whl/cpu"
KIT_PYTHON="${KIT_PYTHON:-python3}"

fail() {
  echo "$1" >&2
  exit "${2:-3}"
}

[[ "$(uname -s):$(uname -m)" == "Linux:x86_64" ]] || fail "LIBERO-Pro runtime builds require Linux x86_64"
for required in "${MANIFEST}" "${CONDA_SPECS}" "${LOCK}" "${TASK_FILES}"; do
  [[ -f "${required}" ]] || fail "LIBERO-Pro build input is missing: ${required}"
done
grep -q '^python-3\.11\.18-' "${CONDA_SPECS}" || fail "LIBERO-Pro conda specs must pin CPython 3.11.18"
for pin in "bddl==1.0.1" "robosuite==1.4.0" "mujoco==" "numpy==1.24.4" "torch=="; do
  grep -q "^${pin}" "${LOCK}" || fail "LIBERO-Pro runtime lock must pin ${pin}"
done
"${KIT_PYTHON}" -c 'import roboscope_kit' 2>/dev/null ||
  fail "KIT_PYTHON=${KIT_PYTHON} cannot import roboscope_kit; install roboscope-kit into it or set KIT_PYTHON"
CONDA_BIN="$(command -v conda || command -v micromamba || true)"
[[ -n "${CONDA_BIN}" ]] || fail "conda or micromamba is required to build the LIBERO-Pro runtime"

# A host pip configuration (GPU hosts often add an unreachable extra index) must
# not decide what a pinned runtime resolves to.
export PIP_CONFIG_FILE=/dev/null

STAGING="$(mktemp -d "${INSTALL_DIR}/.build.XXXXXX")"
trap 'rm -rf -- "${STAGING}"' EXIT

if [[ -d "${SOURCE_ROOT}" ]]; then
  ACTUAL_REF="$(git -C "${SOURCE_ROOT}" rev-parse HEAD)"
  [[ "${ACTUAL_REF}" == "${SOURCE_REF}" ]] ||
    fail "${SOURCE_ROOT} is at ${ACTUAL_REF}, expected ${SOURCE_REF}; remove it to rebuild" 4
else
  # The repository is about 1 GB with history; fetch only the pinned commit.
  git init --quiet "${STAGING}/LIBERO-PRO"
  git -C "${STAGING}/LIBERO-PRO" fetch --quiet --depth 1 "${SOURCE_URL}" "${SOURCE_REF}"
  git -C "${STAGING}/LIBERO-PRO" checkout --quiet --detach FETCH_HEAD
  mv -- "${STAGING}/LIBERO-PRO" "${SOURCE_ROOT}"
fi
for relative in setup.py libero/libero/__init__.py libero/libero/benchmark/__init__.py \
  libero/libero/benchmark/libero_suite_task_map.py libero/libero/assets/scenes \
  libero/libero/assets/textures libero/libero/assets/serving_region.xml; do
  [[ -e "${SOURCE_ROOT}/${relative}" ]] || fail "LIBERO-PRO source tree is missing ${relative}" 5
done

# Place the released task files of the perturbed suites. A file already in place is kept only if it
# matches its checksum; anything else is downloaded again from the pinned revision.
FILES_ROOT="${SOURCE_ROOT}/libero/libero"
TASK_SUMS="${STAGING}/task-files.sha256"
grep -Ev '^(#|$)' "${TASK_FILES}" >"${TASK_SUMS}"
[[ "$(wc -l <"${TASK_SUMS}")" -eq 320 ]] || fail "task-files.sha256 must list 320 files (16 suites x 10 tasks x 2)"
while read -r digest relative; do
  [[ "${relative}" =~ ^(bddl_files|init_files)/libero_(goal|spatial|10|object)_(object|swap|lan|task)/[A-Za-z0-9_]+\.(bddl|pruned_init)$ ]] ||
    fail "unexpected path in task-files.sha256: ${relative}"
  target="${FILES_ROOT}/${relative}"
  if [[ -f "${target}" ]] && [[ "$(sha256sum "${target}" | awk '{print $1}')" == "${digest}" ]]; then
    continue
  fi
  download="${STAGING}/files/${relative}"
  mkdir -p -- "${download%/*}"
  curl --fail --location --silent --show-error --retry 5 --output "${download}" "${FILES_URL}/${relative}"
  [[ "$(sha256sum "${download}" | awk '{print $1}')" == "${digest}" ]] ||
    fail "LIBERO-Pro ${relative} does not match revision ${FILES_REVISION} (SHA-256 ${digest})" 5
  mkdir -p -- "${target%/*}"
  mv -f -- "${download}" "${target}"
done <"${TASK_SUMS}"
(cd -- "${FILES_ROOT}" && sha256sum --check --quiet --strict "${TASK_SUMS}") ||
  fail "LIBERO-Pro task files do not match task-files.sha256" 5

# conda-specs.txt keeps conda-meta's name-version-build spelling. conda accepts
# it but micromamba reads the whole string as a package name, so pass
# name=version=build. The build strings come from conda-forge.
CONDA_PACKAGES=()
while IFS= read -r line; do
  [[ -z "${line}" || "${line}" == \#* ]] && continue
  [[ "${line}" =~ ^(.*)-([0-9][^-]*)-([^-]+)$ ]] || fail "unparsable conda spec: ${line}"
  CONDA_PACKAGES+=("${BASH_REMATCH[1]}=${BASH_REMATCH[2]}=${BASH_REMATCH[3]}")
done <"${CONDA_SPECS}"
if [[ ! -x "${ENV_PYTHON}" ]]; then
  "${CONDA_BIN}" create --yes --quiet -c conda-forge --override-channels --prefix "${ENV_PREFIX}" \
    "${CONDA_PACKAGES[@]}"
fi
# The source tree's setup.py declares libero 0.1.0, which PyPI never published.
PYTHONNOUSERSITE=1 "${ENV_PYTHON}" -m pip install --no-cache-dir --no-deps "${SOURCE_ROOT}"
# --no-deps: the lock is a complete distribution set, so dependency resolution
# could only substitute versions. torch 2.5.1+cpu, which LIBERO uses to load the
# initial states, only exists on the PyTorch CPU index.
PYTHONNOUSERSITE=1 "${ENV_PYTHON}" -m pip install --no-cache-dir --no-deps \
  --extra-index-url "${TORCH_INDEX_URL}" --requirement "${LOCK}"

env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 "${ENV_PYTHON}" -I - <<'PY'
import sys
from importlib.metadata import version

import bddl
import matplotlib
import robosuite
import torch

if sys.version_info[:3] != (3, 11, 18):
    raise SystemExit(f"LIBERO-Pro runtime is Python {sys.version.split()[0]}, expected 3.11.18")
for distribution, expected in (
    ("libero", "0.1.0"),
    ("bddl", "1.0.1"),
    ("robosuite", "1.4.0"),
    ("mujoco", "3.2.3"),
    ("numpy", "1.24.4"),
    ("torch", "2.5.1+cpu"),
):
    actual = version(distribution)
    if actual != expected:
        raise SystemExit(f"LIBERO-Pro runtime {distribution} is {actual}, expected {expected}")
print("LIBERO-Pro runtime imports passed")
PY

# Write the LIBERO path config with the kit, then import LIBERO and the kit's
# worker in the LIBERO-Pro interpreter exactly as an evaluation worker does, and
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
mapfile -t SUITES < <(
  "${KIT_PYTHON}" - <<'PY'
from roboscope_kit.runtime.benchmarks.libero_pro import LIBERO_PRO_SUITES

print("\n".join(sorted(LIBERO_PRO_SUITES)))
PY
)
[[ "${#SUITES[@]}" -eq 16 ]] || fail "the kit lists ${#SUITES[@]} LIBERO-Pro suites, expected 16"
# Besides the import check, every released suite must be registered in this source revision with the
# 10 tasks whose files were placed above, and each task's initial states must load with 50 states.
env -u PYTHONPATH \
  PYTHONPATH="${SOURCE_ROOT}:${KIT_IMPORT_ROOT}" \
  LIBERO_HOME="${SOURCE_ROOT}" \
  LIBERO_CONFIG_PATH="${CONFIG_PATH%/*}" \
  PYTHONDONTWRITEBYTECODE=1 \
  PYTHONNOUSERSITE=1 \
  "${ENV_PYTHON}" - "${CONFIG_PATH}" "${SOURCE_ROOT}" "${KIT_IMPORT_ROOT}" "${SUITES[@]}" <<'PY'
import sys
from pathlib import Path

from roboscope_kit.runtime.benchmarks.libero.adapter import validate_libero_runtime

validate_libero_runtime(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]))

import contextlib
import io

from libero.libero import benchmark

suites = benchmark.get_benchmark_dict()
for name in sys.argv[4:]:
    # LIBERO prints a warning on every path lookup because no demonstration dataset is installed.
    with contextlib.redirect_stdout(io.StringIO()):
        suite = suites[name]()
        bddl_files = [Path(suite.get_task_bddl_file_path(task_id)) for task_id in range(suite.get_num_tasks())]
        state_counts = [len(suite.get_task_init_states(task_id)) for task_id in range(suite.get_num_tasks())]
    if len(bddl_files) != 10:
        raise SystemExit(f"LIBERO-Pro suite {name} has {len(bddl_files)} tasks, expected 10")
    missing = [str(path) for path in bddl_files if not path.is_file()]
    if missing:
        raise SystemExit(f"LIBERO-Pro suite {name} has no BDDL file {missing[0]}")
    if state_counts != [50] * 10:
        raise SystemExit(f"LIBERO-Pro suite {name} has {state_counts} initial states per task, expected 50")
print(f"LIBERO-Pro config, kit worker imports, and {len(sys.argv) - 4} suites passed")
PY

install -m 0644 "${MANIFEST}" "${INSTALL_DIR}/benchmark-manifest.json"
rm -rf -- "${INSTALL_DIR}/self-test"
"${KIT_PYTHON}" -m roboscope_kit.runtime.benchmark --benchmark libero_pro --self-test \
  --benchmark-manifest "${INSTALL_DIR}/benchmark-manifest.json" \
  --output-dir "${INSTALL_DIR}/self-test"
echo "Built LIBERO-Pro ${SOURCE_REF} with task files ${FILES_REVISION} in ${INSTALL_DIR}"

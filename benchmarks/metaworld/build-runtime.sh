#!/usr/bin/env bash
set -Eeuo pipefail

# Build the MetaWorld MT50 benchmark environment under <install-dir>:
#
#   <install-dir>/
#     benchmark-manifest.json   this directory's manifest; runtime paths are relative to it
#     python/                   the uv-managed CPython 3.12 the environment runs on
#     env/                      venv with exactly runtime-requirements.txt
#     pip-freeze.txt            what env/ actually contains
#
# Usage: build-runtime.sh <install-dir>
# Then:  python -m roboscope_kit.runtime.benchmark --benchmark metaworld \
#          --benchmark-manifest <install-dir>/benchmark-manifest.json ...
#
# The build verifies the frozen MT50(seed=42) train-task digest against the
# constant the kit's MetaWorld adapter asserts, so an environment whose task
# goals differ is rejected here rather than at evaluation time. Re-running the
# script on an existing <install-dir> reuses the venv and re-applies the lock.

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
INSTRUCTIONS="${KIT_ROOT}/packages/roboscope-kit/src/roboscope_kit/runtime/benchmarks/metaworld/instructions.py"
PYTHON_INSTALL_DIR="${INSTALL_DIR}/python"
VENV="${INSTALL_DIR}/env"
PYTHON="${VENV}/bin/python"
PYTHON_VERSION="3.12"
FROZEN_PREFIX="FROZEN_MT50_TASKS_SHA256 = \""

if [[ "$(uname -s):$(uname -m)" != "Linux:x86_64" ]]; then
  echo "MetaWorld runtime builds require Linux x86_64" >&2
  exit 3
fi
if [[ ! -f "${MANIFEST}" || ! -f "${LOCK}" || ! -f "${INSTRUCTIONS}" ]]; then
  echo "MetaWorld manifest, runtime lock or instruction constants are missing" >&2
  exit 3
fi
for pin in "metaworld==3.1.1" "mujoco==3.3.0" "gymnasium==" "numpy=="; do
  grep -q "^${pin}" "${LOCK}" || {
    echo "MetaWorld runtime lock must pin ${pin}" >&2
    exit 3
  }
done
if ! command -v uv >/dev/null 2>&1; then
  echo "uv 0.11.14 is required to build the MetaWorld runtime" >&2
  exit 3
fi
UV_BIN="$(command -v uv)"
if [[ "$("${UV_BIN}" --version)" != "uv 0.11.14 "* ]]; then
  echo "uv 0.11.14 is required to build the MetaWorld runtime" >&2
  exit 3
fi
FROZEN_DIGEST="$(
  sed -n "s/^${FROZEN_PREFIX}\\([0-9a-f]\\{64\\}\\)\".*/\\1/p" "${INSTRUCTIONS}"
)"
if [[ ! "${FROZEN_DIGEST}" =~ ^[0-9a-f]{64}$ ]]; then
  echo "Unable to read the frozen MT50 train-task digest from ${INSTRUCTIONS}" >&2
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
"${UV_BIN}" pip install \
  --python "${PYTHON}" \
  --no-progress \
  --requirements "${LOCK}"
"${UV_BIN}" pip freeze --python "${PYTHON}" > "${INSTALL_DIR}/pip-freeze.txt"

env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 "${PYTHON}" -I - "${FROZEN_DIGEST}" <<'PY'
import hashlib
import sys
from collections import defaultdict
from importlib.metadata import version

import metaworld
from metaworld.env_dict import ALL_V3_ENVIRONMENTS

expected = sys.argv[1]
for distribution, locked in (
    ("metaworld", "3.1.1"),
    ("mujoco", "3.3.0"),
    ("gymnasium", "1.1.1"),
    ("numpy", "2.1.3"),
    ("scipy", "1.14.1"),
    ("pillow", "11.0.0"),
):
    actual = version(distribution)
    if actual != locked:
        raise SystemExit(f"MetaWorld runtime {distribution} is {actual}, expected {locked}")

tasks = list(metaworld.MT50(seed=42).train_tasks)
grouped: dict[str, list] = defaultdict(list)
for task in tasks:
    grouped[task.env_name].append(task)
if len(tasks) != 2500 or len(grouped) != 50 or any(len(v) != 50 for v in grouped.values()):
    raise SystemExit("MT50(seed=42) did not generate 50 tasks x 50 training goals")
if set(grouped) != set(ALL_V3_ENVIRONMENTS):
    raise SystemExit("MT50(seed=42) does not cover the Farama task list")

digest = hashlib.sha256()
for task in tasks:
    digest.update(task.env_name.encode("utf-8"))
    digest.update(b"\0")
    digest.update(task.data)
    digest.update(b"\0")
actual_digest = digest.hexdigest()
if actual_digest != expected:
    raise SystemExit(
        f"MT50(seed=42) train-task digest {actual_digest} does not match the committed {expected}"
    )
print(f"MT50(seed=42) train-task digest verified: {actual_digest}")
PY

install -m 0644 "${MANIFEST}" "${INSTALL_DIR}/benchmark-manifest.json"
echo "Built MetaWorld MT50 in ${INSTALL_DIR}"
echo "Frozen MT50 train-task digest: ${FROZEN_DIGEST}"

# The kit runs its own import check against the installed manifest. It needs a
# Python with roboscope-kit installed; the MetaWorld environment does not need it.
if [[ -n "${KIT_PYTHON:-}" ]] || python3 -c 'import roboscope_kit' >/dev/null 2>&1; then
  KIT_PYTHON="${KIT_PYTHON:-python3}"
  "${KIT_PYTHON}" -c 'import roboscope_kit' || {
    echo "KIT_PYTHON=${KIT_PYTHON} cannot import roboscope_kit; install roboscope-kit into it" >&2
    exit 6
  }
  "${KIT_PYTHON}" -m roboscope_kit.runtime.benchmark --benchmark metaworld --self-test \
    --benchmark-manifest "${INSTALL_DIR}/benchmark-manifest.json" \
    --output-dir "${INSTALL_DIR}/self-test"
else
  echo "Skipped the kit self-test: python3 cannot import roboscope_kit; set KIT_PYTHON to run it" >&2
fi

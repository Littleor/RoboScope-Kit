#!/usr/bin/env bash
set -Eeuo pipefail

# One-shot builder for the MetaWorld MT50 benchmark runtime.
#
# The shared-disk runtime is the byte-exact output of this committed script: it
# creates the pinned venv at its final path, verifies the frozen MT50(seed=42)
# train-task digest against the committed constant, and packs both the venv and
# its managed CPython into /root/autodl-fs/benchmarks/metaworld/artifacts.
# Preparation never pip-installs; it only extracts this archive.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
KIT_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd -P)"
MOUNT_ROOT="${1:-/root/autodl-fs/benchmarks}"
RUNTIME_ROOT="${MOUNT_ROOT}/metaworld"
ARTIFACT_DIR="${RUNTIME_ROOT}/artifacts"
ARCHIVE="${ARTIFACT_DIR}/metaworld-env.tar"
LOCK="${SCRIPT_DIR}/runtime-requirements.txt"
INSTRUCTIONS="${KIT_ROOT}/packages/roboscope-kit/src/roboscope_kit/runtime/benchmarks/metaworld/instructions.py"
LOCAL_ROOT="/root/benchmark-runtime/metaworld"
PYTHON_INSTALL_DIR="${LOCAL_ROOT}/python"
PYTHON="${LOCAL_ROOT}/metaworld/bin/python"
PYTHON_VERSION="3.12"
FROZEN_PREFIX="FROZEN_MT50_TASKS_SHA256 = \""

if [[ "$(uname -s):$(uname -m)" != "Linux:x86_64" ]]; then
  echo "MetaWorld runtime builds require Linux x86_64" >&2
  exit 3
fi
if [[ "${EUID}" -ne 0 ]]; then
  echo "MetaWorld runtime builds must run as root" >&2
  exit 3
fi
if [[ ! -f "${LOCK}" || ! -f "${INSTRUCTIONS}" ]]; then
  echo "MetaWorld runtime lock or instruction constants are missing" >&2
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
if [[ -e "${LOCAL_ROOT}/metaworld" ]]; then
  echo "MetaWorld runtime already exists; remove it for a clean rebuild: ${LOCAL_ROOT}/metaworld" >&2
  exit 4
fi

install -d -m 0755 "${LOCAL_ROOT}" "${PYTHON_INSTALL_DIR}" "${ARTIFACT_DIR}"
UV_PYTHON_INSTALL_DIR="${PYTHON_INSTALL_DIR}" \
  "${UV_BIN}" python install --no-bin --no-progress "${PYTHON_VERSION}"
UV_PYTHON_INSTALL_DIR="${PYTHON_INSTALL_DIR}" \
  "${UV_BIN}" venv \
    --python "${PYTHON_VERSION}" \
    --managed-python \
    --no-python-downloads \
    --relocatable \
    "${LOCAL_ROOT}/metaworld"
UV_PYTHON_INSTALL_DIR="${PYTHON_INSTALL_DIR}" \
  "${UV_BIN}" pip install \
    --python "${PYTHON}" \
    --no-progress \
    --requirements "${LOCK}"

"${UV_BIN}" pip freeze --python "${PYTHON}" > "${ARTIFACT_DIR}/metaworld-pip-freeze.txt"

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

tar --sort=name --mtime=@0 --owner=0 --group=0 --numeric-owner \
  -cf "${ARCHIVE}.tmp" \
  -C "${LOCAL_ROOT}" metaworld python
mv -- "${ARCHIVE}.tmp" "${ARCHIVE}"
echo "Built ${ARCHIVE}"
echo "Frozen MT50 train-task digest: ${FROZEN_DIGEST}"

#!/usr/bin/env bash
set -Eeuo pipefail

LOCAL_ROOT="${1:?local runtime root is required}"
CONTROL_PYTHON="${ROBOSCOPE_CONTROL_PYTHON:?ROBOSCOPE_CONTROL_PYTHON is required}"
ARCHIVE="/root/autodl-fs/benchmarks/libero/archives/libero-runtime.tar.gz"
LIBERO_PYTHON="${LOCAL_ROOT}/libero/bin/python"
LIBERO_HOME="${LOCAL_ROOT}/LIBERO"
CONFIG_PATH="${LOCAL_ROOT}/config/config.yaml"

runtime_complete() {
  [[ -x "${LIBERO_PYTHON}" ]] &&
    [[ -f "${LIBERO_HOME}/libero/libero/__init__.py" ]] &&
    [[ -d "${LIBERO_HOME}/libero/libero/bddl_files" ]] &&
    [[ -d "${LIBERO_HOME}/libero/libero/init_files" ]] &&
    [[ -d "${LIBERO_HOME}/libero/libero/assets" ]]
}

if [[ ! -x "${CONTROL_PYTHON}" ]]; then
  echo "Control Python is not executable: ${CONTROL_PYTHON}" >&2
  exit 11
fi
CONTROL_RUNTIME_ROOT="$(cd -- "$(dirname -- "${CONTROL_PYTHON}")/../.." && pwd -P)"
EXPECTED_SDK_IMPORT_ROOT="${CONTROL_RUNTIME_ROOT}/packages/roboscope-kit/src"
if [[ ! -f "${ARCHIVE}" ]]; then
  echo "LIBERO runtime archive is missing: ${ARCHIVE}" >&2
  exit 12
fi
if ! runtime_complete; then
  tar -xzf "${ARCHIVE}" -C "${LOCAL_ROOT}"
fi
if ! runtime_complete; then
  echo "LIBERO archive did not produce a complete runtime under ${LOCAL_ROOT}" >&2
  exit 13
fi

SDK_IMPORT_ROOT="$(
  env -u PYTHONPATH "${CONTROL_PYTHON}" -I - "${CONFIG_PATH}" "${LIBERO_HOME}" <<'PY'
import sys
from pathlib import Path

from roboscope_kit.runtime.benchmarks.libero.adapter import (
    SDK_IMPORT_ROOT,
    write_libero_config,
)

write_libero_config(Path(sys.argv[1]), Path(sys.argv[2]))
print(SDK_IMPORT_ROOT)
PY
)"
if [[ "${SDK_IMPORT_ROOT}" != "${EXPECTED_SDK_IMPORT_ROOT}" ]]; then
  echo "Control SDK import root is ${SDK_IMPORT_ROOT}, expected ${EXPECTED_SDK_IMPORT_ROOT}" >&2
  exit 14
fi

env \
  PYTHONPATH="${LIBERO_HOME}:${SDK_IMPORT_ROOT}" \
  LIBERO_HOME="${LIBERO_HOME}" \
  LIBERO_CONFIG_PATH="${CONFIG_PATH%/*}" \
  PYTHONDONTWRITEBYTECODE=1 \
  "${LIBERO_PYTHON}" - "${CONFIG_PATH}" "${LIBERO_HOME}" "${SDK_IMPORT_ROOT}" <<'PY'
import sys
from pathlib import Path

from roboscope_kit.runtime.benchmarks.libero.adapter import validate_libero_runtime

validate_libero_runtime(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]))
PY

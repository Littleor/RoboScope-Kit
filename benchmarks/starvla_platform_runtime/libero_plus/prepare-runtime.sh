#!/usr/bin/env bash
set -Eeuo pipefail

LOCAL_ROOT="${1:?local runtime root is required}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
MANIFEST="${SCRIPT_DIR}/benchmark-image.json"
CONTROL_PYTHON="${ROBOSCOPE_CONTROL_PYTHON:?ROBOSCOPE_CONTROL_PYTHON is required}"
ARCHIVE="/root/autodl-fs/benchmarks/liberoplus/archives/LIBERO-plus.tar"

if [[ ! -x "${CONTROL_PYTHON}" ]]; then
  echo "Control Python is not executable: ${CONTROL_PYTHON}" >&2
  exit 11
fi
CONTROL_RUNTIME_ROOT="$(cd -- "$(dirname -- "${CONTROL_PYTHON}")/../.." && pwd -P)"
EXPECTED_SDK_IMPORT_ROOT="${CONTROL_RUNTIME_ROOT}/packages/roboscope-kit/src"
if [[ ! -f "${MANIFEST}" ]]; then
  echo "LIBERO+ benchmark manifest is missing: ${MANIFEST}" >&2
  exit 12
fi

RUNTIME_PATHS="$("${CONTROL_PYTHON}" -I - "${MANIFEST}" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    runtime = json.load(handle)["runtime"]
print(runtime["python"], runtime["benchmark_root"], runtime["config_path"], sep="\t")
PY
)"
IFS=$'\t' read -r LIBERO_PYTHON LIBERO_HOME CONFIG_PATH <<< "${RUNTIME_PATHS}"
SOURCE_ROOT="${LOCAL_ROOT}/LIBERO-plus"
if [[ "${LIBERO_HOME}" != "${SOURCE_ROOT}" ]]; then
  echo "LIBERO+ manifest benchmark root does not match ${SOURCE_ROOT}: ${LIBERO_HOME}" >&2
  exit 13
fi
EXPECTED_CONFIG_PATH="${LOCAL_ROOT}/config/config.yaml"
if [[ "${CONFIG_PATH}" != "${EXPECTED_CONFIG_PATH}" ]]; then
  echo "LIBERO+ manifest config path does not match ${EXPECTED_CONFIG_PATH}: ${CONFIG_PATH}" >&2
  exit 14
fi
if [[ ! -x "${LIBERO_PYTHON}" ]]; then
  echo "LIBERO+ Python is not executable: ${LIBERO_PYTHON}" >&2
  exit 15
fi

source_complete() {
  [[ -f "${LIBERO_HOME}/setup.py" ]] &&
    [[ -f "${LIBERO_HOME}/libero/libero/__init__.py" ]] &&
    [[ -d "${LIBERO_HOME}/libero/libero/bddl_files" ]] &&
    [[ -d "${LIBERO_HOME}/libero/libero/init_files" ]] &&
    [[ -d "${LIBERO_HOME}/libero/libero/assets" ]]
}

if [[ ! -f "${ARCHIVE}" ]]; then
  echo "LIBERO+ archive is missing: ${ARCHIVE}" >&2
  exit 16
fi
if ! source_complete; then
  tar -xf "${ARCHIVE}" -C "${LOCAL_ROOT}"
fi
if ! source_complete; then
  echo "LIBERO+ archive did not produce a complete runtime under ${LOCAL_ROOT}" >&2
  exit 17
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
  exit 18
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

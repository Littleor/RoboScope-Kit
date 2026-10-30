#!/usr/bin/env bash
set -Eeuo pipefail

LOCAL_ROOT="${1:?RoboCasa local runtime root is required}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
MANIFEST="${SCRIPT_DIR}/benchmark-image.json"
CONTROL_PYTHON="${ROBOSCOPE_CONTROL_PYTHON:?ROBOSCOPE_CONTROL_PYTHON is required}"

RUNTIME_PATHS="$("${CONTROL_PYTHON}" -I - "${MANIFEST}" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    runtime = json.load(handle)["runtime"]
print(runtime["python"], runtime["benchmark_root"], sep="\t")
PY
)"
IFS=$'\t' read -r ROBOCASA_PYTHON ROBOCASA_ROOT <<< "${RUNTIME_PATHS}"
[[ -x "${ROBOCASA_PYTHON}" ]] || {
  echo "RoboCasa Python is not executable: ${ROBOCASA_PYTHON}" >&2
  exit 31
}
[[ -d "${ROBOCASA_ROOT}" ]] || {
  echo "RoboCasa benchmark root is missing: ${ROBOCASA_ROOT}" >&2
  exit 32
}

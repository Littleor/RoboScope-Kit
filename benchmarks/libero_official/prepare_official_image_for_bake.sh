#!/usr/bin/env bash
set -Eeuo pipefail

TARGET_ROOT="/root"

usage() {
  cat <<'EOF'
Usage: prepare-for-bake.sh [--target-root <absolute-path>]

Remove per-run and per-user state before saving a verified official base image.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --target-root)
      TARGET_ROOT="${2:?--target-root requires an absolute path}"
      shift 2
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ "${TARGET_ROOT}" != /* || "${TARGET_ROOT}" == "/" ]]; then
  echo "Target root must be an absolute directory other than /" >&2
  exit 3
fi

"${TARGET_ROOT}/resetenv.sh"

rm -rf -- \
  "${TARGET_ROOT}/testresult" \
  "${TARGET_ROOT}/testvideo" \
  "${TARGET_ROOT}/roboscope-checkpoint" \
  "${TARGET_ROOT}/benchmark" \
  "${TARGET_ROOT}/.cache/pip" \
  "${TARGET_ROOT}/.cache/uv"
rm -f -- \
  "${TARGET_ROOT}/testlog.txt" \
  "${TARGET_ROOT}/.bash_history" \
  "${TARGET_ROOT}/.python_history"

WORKSPACE="${TARGET_ROOT%/}/../workspace"
if [[ "${TARGET_ROOT}" == "/root" && -d "/workspace" ]]; then
  WORKSPACE="/workspace"
fi
if [[ -d "${WORKSPACE}" ]]; then
  find "${WORKSPACE}" -mindepth 1 -maxdepth 1 -exec rm -rf -- {} +
fi

if [[ "${TARGET_ROOT}" == "/root" && -f /etc/environment ]]; then
  sed -i \
    -e '/^[[:space:]]*ROBOSCOPE_POLICY_COMMAND_JSON=/d' \
    -e '/^[[:space:]]*ROBOSCOPE_POLICY_CWD=/d' \
    -e '/^[[:space:]]*ROBOSCOPE_POLICY_CONFIG_PATH=/d' \
    -e '/^[[:space:]]*ROBOSCOPE_BENCHMARK_RUNTIME=/d' \
    /etc/environment
fi

if [[ -e "${TARGET_ROOT}/roboscope-checkpoint" ]]; then
  echo "Failed to remove embedded user checkpoint" >&2
  exit 4
fi
for path in benchmark testresult testvideo testlog.txt; do
  if [[ -e "${TARGET_ROOT}/${path}" ]]; then
    echo "Failed to remove stale benchmark state: ${TARGET_ROOT}/${path}" >&2
    exit 4
  fi
done
if [[ -n "$(find "${WORKSPACE}" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]]; then
  echo "Failed to clear user workspace: ${WORKSPACE}" >&2
  exit 4
fi

echo "Official image user and runtime state cleared"

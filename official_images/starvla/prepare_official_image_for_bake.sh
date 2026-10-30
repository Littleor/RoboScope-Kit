#!/usr/bin/env bash
set -Eeuo pipefail

TARGET_ROOT="/root"

usage() {
  cat <<'EOF'
Usage: prepare-for-bake.sh [--target-root <absolute-path>]

Remove per-run and per-user state before saving a verified official image.
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

# AutoDL attaches the shared volume at this canonical path on each instance.
# An acceptance run may redirect the user-facing link to private local storage;
# that instance-specific target must not survive an image bake.
if [[ "${TARGET_ROOT}" == "/root" && -L /root/autodl-fs ]]; then
  ln -s /autodl-fs/data /root/.roboscope-autodl-fs.$$.tmp
  mv -Tf -- /root/.roboscope-autodl-fs.$$.tmp /root/autodl-fs
fi

rm -rf -- \
  "${TARGET_ROOT}/testresult" \
  "${TARGET_ROOT}/testvideo" \
  "${TARGET_ROOT}/roboscope-checkpoint" \
  "${TARGET_ROOT}/roboscope-kit" \
  "${TARGET_ROOT}/benchmark" \
  "${TARGET_ROOT}/benchmark-runtime" \
  "${TARGET_ROOT}/benchmark-cache" \
  "${TARGET_ROOT}/benchmark-home" \
  "${TARGET_ROOT}/.cache/huggingface" \
  "${TARGET_ROOT}/.cache/modelscope" \
  "${TARGET_ROOT}/.cache/pip" \
  "${TARGET_ROOT}/.cache/uv" \
  "${TARGET_ROOT}/autodl-tmp"/roboscope-* \
  "${TARGET_ROOT}"/roboscope-official-starvla-image-*
rm -f -- \
  "${TARGET_ROOT}/testlog.txt" \
  "${TARGET_ROOT}"/starvla-policy-*.log \
  "${TARGET_ROOT}/.bash_history" \
  "${TARGET_ROOT}/.python_history"

WORKSPACE="${TARGET_ROOT%/}/../workspace"
if [[ "${TARGET_ROOT}" == "/root" && -d "/workspace" ]]; then
  WORKSPACE="/workspace"
fi
if [[ -d "${WORKSPACE}" ]]; then
  find "${WORKSPACE}" -mindepth 1 -maxdepth 1 -exec rm -rf -- {} +
else
  mkdir -p "${WORKSPACE}"
fi
DEFAULT_POLICY_CONFIG="${TARGET_ROOT%/}/../opt/roboscope/defaults/roboscope-policy.json"
if [[ ! -f "${DEFAULT_POLICY_CONFIG}" ]]; then
  echo "Missing official default Policy config: ${DEFAULT_POLICY_CONFIG}" >&2
  exit 4
fi
install -m 0644 "${DEFAULT_POLICY_CONFIG}" "${WORKSPACE}/roboscope-policy.json"

if [[ "${TARGET_ROOT}" == "/root" && -f /etc/environment ]]; then
  sed -i \
    -e '/^[[:space:]]*ROBOSCOPE_POLICY_COMMAND_JSON=/d' \
    -e '/^[[:space:]]*ROBOSCOPE_POLICY_CWD=/d' \
    -e '/^[[:space:]]*ROBOSCOPE_POLICY_CONFIG_PATH=/d' \
    -e '/^[[:space:]]*ROBOSCOPE_BENCHMARK_RUNTIME=/d' \
    -e '/^[[:space:]]*ROBOSCOPE_INTERNAL_BENCHMARK_RUNTIME=/d' \
    -e '/^[[:space:]]*ROBOSCOPE_PLATFORM_ENTRYPOINT=/d' \
    -e '/^[[:space:]]*ROBOSCOPE_PLATFORM_RESET=/d' \
    /etc/environment
fi

if [[ -e "${TARGET_ROOT}/roboscope-checkpoint" ]]; then
  echo "Failed to remove embedded user checkpoint" >&2
  exit 4
fi
if [[ -e "${TARGET_ROOT}/roboscope-kit" ]]; then
  echo "Failed to remove stale RoboScope Kit checkout" >&2
  exit 4
fi
for path in testlog.txt testresult testvideo; do
  if [[ -e "${TARGET_ROOT}/${path}" ]]; then
    echo "Failed to remove stale benchmark output: ${TARGET_ROOT}/${path}" >&2
    exit 4
  fi
done
for path in benchmark benchmark-runtime benchmark-cache benchmark-home; do
  if [[ -e "${TARGET_ROOT}/${path}" ]]; then
    echo "Failed to remove extracted Benchmark runtime state: ${TARGET_ROOT}/${path}" >&2
    exit 4
  fi
done
unexpected_workspace_entry="$(find "${WORKSPACE}" -mindepth 1 -maxdepth 1 ! -name roboscope-policy.json -print -quit)"
if [[ -n "${unexpected_workspace_entry}" ]]; then
  echo "Failed to clear user workspace entry: ${unexpected_workspace_entry}" >&2
  exit 4
fi
if ! cmp -s "${DEFAULT_POLICY_CONFIG}" "${WORKSPACE}/roboscope-policy.json"; then
  echo "Failed to restore official default Policy config" >&2
  exit 4
fi

echo "StarVLA official image user state cleared and default Policy restored"

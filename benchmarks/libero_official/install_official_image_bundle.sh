#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
TARGET_ROOT="/root"
DEFAULT_PYTHON=""
LIBERO_PYTHON=""
SKIP_SELF_TEST=0

usage() {
  cat <<'EOF'
Usage: install.sh [--target-root <absolute-path>] [--default-python <path>]
                  [--libero-python <path>] [--skip-self-test]

Install one official base-image bundle. The production image uses the defaults and
must run the hermetic self-test before it is saved as an AutoDL image.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --target-root)
      TARGET_ROOT="${2:?--target-root requires an absolute path}"
      shift 2
      ;;
    --default-python)
      DEFAULT_PYTHON="${2:?--default-python requires a path}"
      shift 2
      ;;
    --libero-python)
      LIBERO_PYTHON="${2:?--libero-python requires a path}"
      shift 2
      ;;
    --skip-self-test)
      SKIP_SELF_TEST=1
      shift
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
if [[ -z "${LIBERO_PYTHON}" ]]; then
  LIBERO_PYTHON="${TARGET_ROOT}/miniconda3/envs/libero/bin/python"
fi
if [[ -z "${DEFAULT_PYTHON}" ]]; then
  DEFAULT_PYTHON="${TARGET_ROOT}/miniconda3/bin/python"
fi

resolve_python() {
  if [[ "$1" == */* ]]; then
    printf '%s\n' "$1"
    return
  fi
  command -v "$1"
}

DEFAULT_PYTHON="$(resolve_python "${DEFAULT_PYTHON}")"
LIBERO_PYTHON="$(resolve_python "${LIBERO_PYTHON}")"
if [[ ! -x "${DEFAULT_PYTHON}" ]]; then
  echo "Default Python is not executable: ${DEFAULT_PYTHON}" >&2
  exit 4
fi
if [[ ! -x "${LIBERO_PYTHON}" ]]; then
  echo "LIBERO Python is not executable: ${LIBERO_PYTHON}" >&2
  exit 4
fi

sha256_check() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum -c "$1" >/dev/null
    return
  fi
  shasum -a 256 -c "$1" >/dev/null
}

(
  cd "${SCRIPT_DIR}"
  sha256_check "payload.sha256"
)
echo "Bundle payload checksums passed"

WHEEL_COUNT="$(find "${SCRIPT_DIR}/wheels" -maxdepth 1 -type f -name '*.whl' | wc -l | tr -d ' ')"
if [[ "${WHEEL_COUNT}" -ne 1 ]]; then
  echo "Expected exactly one SDK wheel, found ${WHEEL_COUNT}" >&2
  exit 5
fi
WHEEL_PATH="$(find "${SCRIPT_DIR}/wheels" -maxdepth 1 -type f -name '*.whl')"

TARGET_KIT_ROOT="${TARGET_ROOT}/roboscope-kit"
TARGET_METADATA_ROOT="${TARGET_ROOT}/.roboscope-official-image"
IMAGE_ROOT="$(cd -- "${TARGET_ROOT}/.." && pwd)"
CONTROL_RUNTIME_ROOT="${IMAGE_ROOT%/}/opt/roboscope/control-runtime"
install -d \
  "${TARGET_KIT_ROOT}/packages" \
  "${TARGET_KIT_ROOT}/benchmarks/libero_official" \
  "${TARGET_ROOT}/roboscope_assets" \
  "${TARGET_METADATA_ROOT}"

rm -rf -- \
  "${TARGET_KIT_ROOT}/packages/roboscope-kit" \
  "${TARGET_KIT_ROOT}/benchmarks/libero_official/starvla_runtime"
cp -a "${SCRIPT_DIR}/evaluation-kit/packages/roboscope-kit" "${TARGET_KIT_ROOT}/packages/"
cp -a \
  "${SCRIPT_DIR}/evaluation-kit/benchmarks/libero_official/starvla_runtime" \
  "${TARGET_KIT_ROOT}/benchmarks/libero_official/"

OFFICIAL_SOURCE="${SCRIPT_DIR}/evaluation-kit/benchmarks/libero_official"
install -m 0755 "${OFFICIAL_SOURCE}/startbenchmark.sh" "${TARGET_ROOT}/startbenchmark.sh"
install -m 0755 "${OFFICIAL_SOURCE}/resetenv.sh" "${TARGET_ROOT}/resetenv.sh"
install -m 0644 "${OFFICIAL_SOURCE}/roboscope_libero_runner.py" "${TARGET_ROOT}/roboscope_libero_runner.py"
install -m 0644 "${OFFICIAL_SOURCE}/roboscope_libero_autotune_eval.py" "${TARGET_ROOT}/roboscope_libero_autotune_eval.py"
install -m 0644 "${OFFICIAL_SOURCE}/base_assets.json" "${TARGET_ROOT}/roboscope_assets/base_assets.json"

"${DEFAULT_PYTHON}" -m pip install --no-deps --force-reinstall "${WHEEL_PATH}"
"${LIBERO_PYTHON}" -m pip install --no-deps --force-reinstall "${WHEEL_PATH}"

CONTROL_METADATA="$("${DEFAULT_PYTHON}" - "${SCRIPT_DIR}/manifest.json" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    manifest = json.load(handle)
print(manifest["evaluation_kit_commit"])
print(manifest["sdk"]["version"])
PY
)"
mapfile -t CONTROL_METADATA_LINES <<< "${CONTROL_METADATA}"
if [[ "${#CONTROL_METADATA_LINES[@]}" -ne 2 ]]; then
  echo "Official image manifest has incomplete Control Runtime metadata" >&2
  exit 5
fi
"${SCRIPT_DIR}/install-control-runtime.sh" \
  --source-root "${SCRIPT_DIR}/evaluation-kit" \
  --artifacts-dir "${SCRIPT_DIR}/control-runtime-artifacts" \
  --target-root "${CONTROL_RUNTIME_ROOT}" \
  --kit-commit "${CONTROL_METADATA_LINES[0]}" \
  --sdk-version "${CONTROL_METADATA_LINES[1]}"

install -m 0644 "${SCRIPT_DIR}/manifest.json" "${TARGET_METADATA_ROOT}/manifest.json"
install -m 0644 "${SCRIPT_DIR}/runtime.sha256" "${TARGET_METADATA_ROOT}/runtime.sha256"
install -m 0755 "${SCRIPT_DIR}/verify.sh" "${TARGET_METADATA_ROOT}/verify.sh"
install -m 0755 "${SCRIPT_DIR}/prepare-for-bake.sh" "${TARGET_METADATA_ROOT}/prepare-for-bake.sh"
install -m 0755 \
  "${SCRIPT_DIR}/install-control-runtime.sh" \
  "${TARGET_METADATA_ROOT}/verify-control-runtime.sh"

VERIFY_ARGS=(
  --target-root "${TARGET_ROOT}"
  --default-python "${DEFAULT_PYTHON}"
  --libero-python "${LIBERO_PYTHON}"
)
if [[ "${SKIP_SELF_TEST}" == "1" ]]; then
  VERIFY_ARGS+=(--skip-self-test)
fi
"${TARGET_METADATA_ROOT}/verify.sh" "${VERIFY_ARGS[@]}"

if [[ "${SKIP_SELF_TEST}" != "1" ]]; then
  "${TARGET_METADATA_ROOT}/prepare-for-bake.sh" --target-root "${TARGET_ROOT}"
fi

echo "Official image payload installed and verified under ${TARGET_ROOT}"

#!/usr/bin/env bash
set -Eeuo pipefail

TARGET_ROOT="/root"
DEFAULT_PYTHON=""
LIBERO_PYTHON=""
SKIP_SELF_TEST=0

usage() {
  cat <<'EOF'
Usage: verify_official_image.sh [--target-root <absolute-path>]
                                [--default-python <path>]
                                [--libero-python <path>]
                                [--skip-self-test]
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

TARGET_KIT_ROOT="${TARGET_ROOT}/roboscope-kit"
TARGET_METADATA_ROOT="${TARGET_ROOT}/.roboscope-official-image"
IMAGE_ROOT="$(cd -- "${TARGET_ROOT}/.." && pwd)"
CONTROL_RUNTIME_ROOT="${IMAGE_ROOT%/}/opt/roboscope/control-runtime"
MANIFEST="${TARGET_METADATA_ROOT}/manifest.json"
RUNTIME_CHECKSUMS="${TARGET_METADATA_ROOT}/runtime.sha256"

for path in \
  "${MANIFEST}" \
  "${RUNTIME_CHECKSUMS}" \
  "${TARGET_METADATA_ROOT}/verify-control-runtime.sh" \
  "${TARGET_ROOT}/startbenchmark.sh" \
  "${TARGET_ROOT}/resetenv.sh" \
  "${TARGET_ROOT}/roboscope_libero_runner.py" \
  "${TARGET_ROOT}/roboscope_libero_autotune_eval.py" \
  "${TARGET_ROOT}/roboscope_assets/base_assets.json" \
  "${TARGET_KIT_ROOT}/packages/roboscope-kit" \
  "${TARGET_KIT_ROOT}/benchmarks/libero_official/starvla_runtime"; do
  if [[ ! -e "${path}" ]]; then
    echo "Missing official image path: ${path}" >&2
    exit 4
  fi
done

if command -v sha256sum >/dev/null 2>&1; then
  (cd "${TARGET_ROOT}" && sha256sum -c "${RUNTIME_CHECKSUMS}" >/dev/null)
else
  (cd "${TARGET_ROOT}" && shasum -a 256 -c "${RUNTIME_CHECKSUMS}" >/dev/null)
fi
echo "Installed runtime checksums passed"

TRUE_COMMAND="$(type -P true)"
ROBOSCOPE_PLATFORM_ENTRYPOINT="${TRUE_COMMAND}" "${TARGET_ROOT}/startbenchmark.sh"
echo "Platform Benchmark delegation verified"

CONTROL_METADATA="$("${DEFAULT_PYTHON}" - "${MANIFEST}" "${CONTROL_RUNTIME_ROOT}" <<'PY'
import json
import sys

manifest_path, control_runtime_root = sys.argv[1:3]
with open(manifest_path, encoding="utf-8") as handle:
    manifest = json.load(handle)
control_runtime = manifest["control_runtime"]
if control_runtime.get("root") != control_runtime_root:
    raise SystemExit("Official image manifest has an invalid Control Runtime root")
if control_runtime.get("python") != f"{control_runtime_root}/control-python/bin/python":
    raise SystemExit("Official image manifest has an invalid Control Runtime python")
artifacts = manifest.get("control_runtime_artifacts", {})
if control_runtime.get("python_version") != artifacts.get("python", {}).get("version"):
    raise SystemExit("Official image manifest Control Runtime python version mismatch")
print(manifest["evaluation_kit_commit"])
print(manifest["sdk"]["version"])
PY
)"
mapfile -t CONTROL_METADATA_LINES <<< "${CONTROL_METADATA}"
if [[ "${#CONTROL_METADATA_LINES[@]}" -ne 2 ]]; then
  echo "Official image manifest has incomplete Control Runtime metadata" >&2
  exit 4
fi
KIT_COMMIT="${CONTROL_METADATA_LINES[0]}"
SDK_VERSION="${CONTROL_METADATA_LINES[1]}"

"${TARGET_METADATA_ROOT}/verify-control-runtime.sh" \
  --verify-only \
  --target-root "${CONTROL_RUNTIME_ROOT}" \
  --kit-commit "${KIT_COMMIT}" \
  --sdk-version "${SDK_VERSION}"

verify_sdk() {
  local python_path="$1"
  "${python_path}" - "${SDK_VERSION}" <<'PY'
import importlib.metadata
import sys

import roboscope_kit

expected = sys.argv[1]
actual = importlib.metadata.version("roboscope-kit")
if actual != expected:
    raise SystemExit(f"SDK version mismatch: expected {expected}, got {actual}")
print(f"{sys.executable}: roboscope-kit {actual}")
PY
}

verify_sdk "${DEFAULT_PYTHON}"
verify_sdk "${LIBERO_PYTHON}"

PYTHONPATH="${TARGET_ROOT}:${TARGET_KIT_ROOT}/packages/roboscope-kit/src:${TARGET_KIT_ROOT}" \
  "${LIBERO_PYTHON}" - <<'PY'
from roboscope_libero_autotune_eval import build_autotune_args
from roboscope_kit.runtime.benchmarks.libero import adapter, worker
from benchmarks.libero_official.starvla_runtime import libero_server

assert build_autotune_args
assert adapter and worker and libero_server
print("official base runtime imports passed")
PY

if [[ "${SKIP_SELF_TEST}" != "1" ]]; then
  if [[ "${TARGET_ROOT}" != "/root" ]]; then
    echo "The image self-test must run against the production /root layout" >&2
    exit 5
  fi
  for path in \
    "${TARGET_ROOT}/starVLA" \
    "${TARGET_ROOT}/LIBERO" \
    "${TARGET_ROOT}/miniconda3/envs/starVLA"; do
    if [[ ! -d "${path}" ]]; then
      echo "Missing pre-baked official base-image dependency: ${path}" >&2
      exit 6
    fi
  done
  ROBOSCOPE_LIBERO_RUNNER="${TARGET_ROOT}/roboscope_libero_runner.py" \
  ROBOSCOPE_SELF_TEST_PYTHON="${LIBERO_PYTHON}" \
  LIBERO_PYTHON="${LIBERO_PYTHON}" \
  ROBOSCOPE_KIT_ROOT="${TARGET_KIT_ROOT}" \
    "${TARGET_ROOT}/startbenchmark.sh" --self-test
fi

"${DEFAULT_PYTHON}" - "${MANIFEST}" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    manifest = json.load(handle)
print(
    "Official image verified: "
    f"evaluation_kit={manifest['evaluation_kit_commit']} sdk={manifest['sdk']['version']}"
)
PY
